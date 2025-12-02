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

class PostProcessor:
    def __init__(self, config):
        self.config = config
        self.erosion_distance = config.get("MASK_EROSION", 0.5)
        # Calculate dilation in PIXELS
        # Strategy A: We eroded by 0.5m. We must dilate by 0.5m.
        # Resolution is fixed at 0.2m/pixel (from DataGenerator)
        self.pixel_resolution = 0.2 
        self.dilation_pixels = self.erosion_distance / self.pixel_resolution

    def process_single_mask(self, binary_mask):
        """
        Pipeline: Raster -> Vector -> Dilate -> Regularize -> Raster (for Vis)
        """
        # 1. Vectorize (Raster -> Polygons)
        # We use a distinct mask=binary_mask to only grab the "1"s
        shapes = rasterio.features.shapes(
            binary_mask.astype(np.uint8), 
            mask=binary_mask.astype(bool)
        )

        refined_polygons = []
        
        for geom, val in shapes:
            if val == 1: 
                poly = shape(geom)
                
                # 2. Dilation (Strategy A)
                # Expands the polygon outward by ~2.5 pixels (0.5m)
                dilated = poly.buffer(self.dilation_pixels, join_style=2) # 2=Mitre (Sharp corners)
                
                # 3. Regularization (Simple "Squaring")
                # For visualization, we use the Oriented Bounding Box (OBB) 
                # as a proxy for complex regularization.
                # In production, use 'Building-Regulariser' library here.
                regularized = dilated.minimum_rotated_rectangle
                
                # Only accept regularization if it doesn't deviate too much (IoU check)
                # If the building is L-shaped, OBB is bad, so we keep the dilated version.
                if dilated.intersection(regularized).area / regularized.area > 0.85:
                    refined_polygons.append(regularized)
                else:
                    refined_polygons.append(dilated.simplify(0.5)) # Fallback: simple smoothing

        return refined_polygons

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
                # We buffer by +0.5m to reverse the training erosion.
                # join_style=2 (Mitre) preserves sharp corners better than Round (1).
                recovered_poly = poly.buffer(self.erosion_distance, join_style=2)
                
                # Simplify slightly to remove pixel-stair-stepping before regularization
                # tolerance=0.1m removes tiny jitter without losing shape
                clean_poly = recovered_poly.simplify(0.1, preserve_topology=True)
                
                polygons.append(clean_poly)
        
        return polygons

    def regularize_polygon(self, polygon):
        """
        Force-squares the polygon edges. 
        Implements a simplified version of the 'Building-Regulariser' logic.
        """
        if polygon.is_empty: return polygon
        
        # Calculate the Oriented Bounding Box (OBB)
        # This gives us the "Dominant Angle" of the building
        obb = polygon.minimum_rotated_rectangle
        
        # For simple rectangular roofs, the OBB is often the best regularization
        # For complex L-shapes, we would need a more complex 'rectification' lib
        # But OBB is a strong baseline for solar analysis (azimuth/tilt).
        
        # Calculate Intersection over Union to see if OBB is a good fit
        iou = polygon.intersection(obb).area / obb.area
        
        # If the building is mostly rectangular (>90% fit), snap to the rectangle
        if iou > 0.90:
            return obb
        
        # Otherwise, return the dilated (but irregular) polygon
        # (Real implementation of 'Building-Regulariser' would go here for L-shapes)
        return polygon

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
                
                # B. Regularize & Attribute
                for poly in raw_polys:
                    reg_poly = self.regularize_polygon(poly)
                    attrs = self.calculate_solar_attributes(reg_poly, None, transform)
                    
                    record = {'geometry': reg_poly}
                    record.update(attrs)
                    all_buildings.append(record)

        # 2. Save Final Layer
        if all_buildings:
            gdf = gpd.GeoDataFrame(all_buildings, crs=crs)
            gdf.to_file(output_file, driver="GPKG")
            print(f"✅ Saved {len(gdf)} solar-ready buildings to {output_file}")
        else:
            print("⚠️ No buildings detected.")
