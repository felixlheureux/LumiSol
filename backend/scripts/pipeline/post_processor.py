import os
import numpy as np
import rasterio
import rasterio.features
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union
from pathlib import Path
from tqdm import tqdm
from buildingregulariser import regularize_geodataframe

class PostProcessor:
    def __init__(self, config):
        self.config = config
        self.pixel_res = 0.2  # 20cm per pixel
        
        # Calculate dilation: 0.5m target
        self.dilation_meters = config.get("MASK_EROSION", 0.5)
        self.dilation_pixels = self.dilation_meters / self.pixel_res

    def regularize_geometry(self, polygons, crs=None):
        """
        Wraps the Building-Regulariser library.
        Handles both Pixel (None CRS) and Metric (Projected CRS) data.
        """
        if not polygons: return []

        gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
        
        # Adjust parameters based on units (Meters vs Pixels)
        # If CRS is None, we assume we are in Pixel Space (Visualizer)
        is_pixel_space = crs is None
        
        # Critical Tuning for Building-Regulariser
        # simplify_tolerance: removes stair-case pixels before squaring.
        # Should be ~2-3x the resolution.
        tol = 1.0 if is_pixel_space else 0.2
        
        try:
            reg_gdf = regularize_geodataframe(
                gdf,
                simplify_tolerance=tol,
                parallel_threshold=2.0,
                allow_45_degree=True, # Critical for L-shapes/Bay windows
                allow_circles=False,  # Turn off for roofs (usually)
                num_cores=1           # Keep 1 for single-image debug to avoid overhead
            )
            return reg_gdf.geometry.tolist()
        except Exception as e:
            print(f"⚠️ Regularization failed: {e}")
            return polygons # Fallback to raw dilation

    def process_single_mask(self, binary_mask):
        """
        Used by Visualizer (Pixel Space)
        """
        # 1. Vectorize (Raster -> Polygons)
        shapes = rasterio.features.shapes(
            binary_mask.astype(np.uint8), 
            mask=binary_mask.astype(bool)
        )

        dilated_polygons = []
        for geom, val in shapes:
            if val == 1: 
                poly = shape(geom)
                # 2. Dilation (Strategy A)
                # Join_style=2 (Mitre) preserves sharp corners
                dilated = poly.buffer(self.dilation_pixels, join_style=2)
                dilated_polygons.append(dilated)
        
        # 3. Robust Regularization (Using Library)
        return self.regularize_geometry(dilated_polygons, crs=None)

    def polygons_to_image(self, polygons, shape):
        """
        Draws the polished polygons back onto a black image for visualization.
        """
        vis_image = np.zeros(shape, dtype=np.uint8)
        
        for poly in polygons:
            if poly.is_empty: continue
            
            # Convert Shapely coords -> OpenCV format (int32)
            coords = np.array(poly.exterior.coords, dtype=np.int32)
            coords = coords.reshape((-1, 1, 2))
            
            # Fill the polygon (White)
            import cv2 # Import locally to avoid top-level dependency if not needed elsewhere
            cv2.fillPoly(vis_image, [coords], 255)
            
        return vis_image

    def vectorize_and_recover(self, prediction_mask, transform):
        """
        Converts raster mask -> Polygons -> Dilated Polygons (+0.5m).
        """
        # 1. Vectorize (Raster -> Raw Polygons)
        # mask=prediction_mask ensures we only vectorize 'True' pixels
        shapes = rasterio.features.shapes(
            prediction_mask.astype(np.uint8), 
            mask=prediction_mask.astype(bool), 
            transform=transform
        )

        polygons = []
        for geom, val in shapes:
            if val == 1: # Class 1 = Building
                poly = shape(geom)
                
                # 2. Dilation (Strategy A)
                # We buffer by +0.5m (dilation_meters) to reverse the training erosion.
                # join_style=2 (Mitre) preserves sharp corners better than Round (1).
                recovered_poly = poly.buffer(self.dilation_meters, join_style=2)
                
                # Simplify slightly to remove pixel-stair-stepping before regularization
                # tolerance=0.1m removes tiny jitter without losing shape
                clean_poly = recovered_poly.simplify(0.1, preserve_topology=True)
                
                polygons.append(clean_poly)
        
        return polygons

    def calculate_solar_attributes(self, polygon, height_map, transform):
        """
        Extracts Z (Height) from the nDSM for this polygon.
        """
        # Create a window reading the nDSM for this specific polygon bounds
        try:
            bounds = polygon.bounds
            window = rasterio.windows.from_bounds(*bounds, transform=transform)
            
            # Read height data
            # Note: We need the original nDSM file passed here, or a crop
            # This is a placeholder for the logic:
            # roi = height_map.read(1, window=window)
            # mean_height = np.nanmean(roi)
            mean_height = 10.0 # Mock value for now
            
            return {
                "area_sqm": polygon.area,
                "mean_height": mean_height,
                "azimuth": 180 # Placeholder: South
            }
        except:
            return {"area_sqm": 0, "mean_height": 0}

    def run(self, input_dir=None):
        print("🔧 Starting Post-Processing Pipeline...")
        
        if input_dir is None:
            input_dir = f"{self.config['OUTPUT_DIR']}/inference_results" # Define this path
            
        output_file = self.config["OUTPUT_VECTOR_PATH"].replace(".gpkg", "_final.gpkg")
        
        # 1. Load Inference Results
        # Assuming you saved your model predictions as TIFs with georeferencing
        # If you saved PNGs, you must reload the original TIF transform!
        pred_files = list(Path(input_dir).glob("*.tif"))
        
        if not pred_files:
            print("❌ No inference result TIFs found. Run inference first.")
            return

        all_buildings = []

        for p_file in tqdm(pred_files):
            with rasterio.open(p_file) as src:
                mask = src.read(1) # Your AI output (0 or 1)
                transform = src.transform
                crs = src.crs
                
                # A. Vectorize & Dilate
                raw_polys = self.vectorize_and_recover(mask, transform)
                
                # B. Regularize (Batch)
                # Use the new library-based regularization
                reg_polys = self.regularize_geometry(raw_polys, crs=crs)
                
                # C. Attribute
                for poly in reg_polys:
                    attrs = self.calculate_solar_attributes(poly, None, transform)
                    
                    record = {'geometry': poly}
                    record.update(attrs)
                    all_buildings.append(record)

        # 2. Save Final Layer
        if all_buildings:
            gdf = gpd.GeoDataFrame(all_buildings, crs=crs)
            gdf.to_file(output_file, driver="GPKG")
            print(f"✅ Saved {len(gdf)} solar-ready buildings to {output_file}")
        else:
            print("⚠️ No buildings detected.")
