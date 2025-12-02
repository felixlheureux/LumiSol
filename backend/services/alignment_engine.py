import os
import numpy as np
import rasterio
from shapely.geometry import shape, box
from samgeo import SamGeo
from scripts.pipeline.config import CONFIG
import leafmap

class AlignmentEngine:
    def __init__(self):
        print("🛰️  Initializing SAM-Geo Aligner...")
        # Initialize SAM with a lightweight model for speed (vit_b is ~300MB)
        # Using 'mps' for Mac acceleration if available
        self.sam = SamGeo(
            model_type="vit_b", 
            device="mps" if os.uname().sysname == "Darwin" else "cuda",
            automatic=False # We will provide prompts
        )
        
        self.cache_dir = os.path.join(CONFIG["CACHE_DIR"], "sat_chips")
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_satellite_chip(self, lat, lon, zoom=20):
        """
        Downloads a georeferenced satellite image (Google/Esri) for the location.
        """
        filename = f"sat_{lat}_{lon}.tif"
        output_path = os.path.join(self.cache_dir, filename)
        
        if os.path.exists(output_path):
            return output_path

        # Download tile using Leafmap (defaulting to Google Satellite)
        # We grab a small area around the point
        try:
            leafmap.map_tiles_to_geotiff(
                output=output_path,
                lat=lat,
                lon=lon,
                zoom=zoom,
                source="Google Satellite", # Or "Esri World Imagery"
                width=512, # Match your pipeline size
                height=512
            )
            return output_path
        except Exception as e:
            print(f"⚠️ Failed to download satellite image: {e}")
            return None

    def calculate_shift(self, lat, lon, gov_polygon_meters):
        """
        Returns the (d_lat, d_lon) shift needed to align LiDAR to Satellite.
        """
        # 1. Get Visual Data
        sat_path = self.get_satellite_chip(lat, lon)
        if not sat_path: return 0, 0

        # 2. Prepare Prompt (Centroid of Gov Polygon)
        # We need the centroid in Lat/Lon to prompt SAM on the geotiff
        # Assuming gov_polygon_meters is in EPSG:32188
        
        with rasterio.open(sat_path) as src:
            # Convert Gov Polygon Centroid (Meters) -> Lat/Lon -> Pixels
            from pyproj import Transformer
            to_web = Transformer.from_crs("EPSG:32188", src.crs, always_xy=True)
            
            cx, cy = gov_polygon_meters.centroid.x, gov_polygon_meters.centroid.y
            wx, wy = to_web.transform(cx, cy) # To Satellite CRS (usually Web Mercator or LatLon)
            
            # Get Pixel Coords for Prompt
            py, px = src.index(wx, wy)
            point_prompt = [[px, py]] # [X, Y] format for SAM? Check docs (usually Col, Row)

        # 3. Run SAM (Prompted Segmentation)
        self.sam.set_image(sat_path)
        # point_coords expects [[x, y]], point_labels expects [1] (Foreground)
        self.sam.predict(point_coords=point_prompt, point_labels=[1])
        
        # 4. Extract Visual Centroid
        # self.sam.masks is a list of masks. We take the highest score one.
        if len(self.sam.masks) == 0:
            return 0, 0 # Detection failed
            
        # Simplified: Save vector and load it back
        temp_vec = os.path.join(self.cache_dir, "temp.gpkg")
        self.sam.save_prediction(temp_vec)
        
        try:
            # Load the SAM Polygon
            import geopandas as gpd
            sam_gdf = gpd.read_file(temp_vec)
            if sam_gdf.empty: return 0, 0
            
            visual_poly = sam_gdf.geometry.iloc[0]
            
            # 5. Calculate Offset
            # Visual Centroid
            vx, vy = visual_poly.centroid.x, visual_poly.centroid.y
            
            # Reference Centroid (The input Lat/Lon we prompted with)
            # wx, wy are the coordinates of the Gov Centroid in the TIF's CRS
            
            # Shift = Visual - Reference
            d_x = vx - wx
            d_y = vy - wy
            
            # Ideally: Return the Vector Shift in Lat/Lon
            return d_x, d_y
            
        except:
            return 0, 0
