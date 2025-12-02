import os
import json
import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
from pyproj import Transformer
from scripts.pipeline.data_generator import DataGenerator
from scripts.pipeline.config import CONFIG

class GeoEngine:
    def __init__(self):
        print("🗺️  Initializing GeoEngine...")
        
        # 1. Load the Tile Index (Fast lookup)
        # We use the JSON list because it maps filenames to locations
        if not os.path.exists(CONFIG["OUTPUT_TILES_PATH"]):
            raise FileNotFoundError("Run 'lumisol.py filter' first to generate tile index.")
            
        with open(CONFIG["OUTPUT_TILES_PATH"]) as f:
            self.tiles_meta = json.load(f)
            
        # 2. Setup Transformers (Lat/Lon <-> Metric)
        # Assuming LiDAR is MTM Zone 8 (EPSG:32188) - Adjust if needed
        self.crs_metric = "EPSG:32188"
        self.crs_web = "EPSG:4326"
        self.to_metric = Transformer.from_crs(self.crs_web, self.crs_metric, always_xy=True)
        self.to_web = Transformer.from_crs(self.crs_metric, self.crs_web, always_xy=True)
        
        # Reuse your existing geometry logic
        self.datagen = DataGenerator(CONFIG)

    def find_tile(self, lat, lon):
        """Finds the local LiDAR file containing the point."""
        mx, my = self.to_metric.transform(lon, lat)
        point = Point(mx, my)
        
        # Simple scan (Optimization: Use an R-Tree if you have >1000 tiles)
        for tile in self.tiles_meta:
            # We construct the path
            path = os.path.join(CONFIG["CACHE_DIR"], tile["filename"])
            if not os.path.exists(path): continue
            
            # Check bounds quickly
            with rasterio.open(path) as src:
                if box(*src.bounds).contains(point):
                    return path, mx, my
        
        return None, mx, my

    def get_patch(self, lat, lon):
        """
        Extracts a 512x512 geometric chip centered on the lat/lon.
        """
        lidar_path, cx, cy = self.find_tile(lat, lon)
        if not lidar_path:
            raise ValueError("No LiDAR data found for this location.")

        # Match training resolution (0.2m/px)
        chip_size_px = CONFIG["CHIP_SIZE"]
        chip_meters = chip_size_px * 0.2
        
        with rasterio.open(lidar_path) as src:
            # Calculate window
            minx, maxx = cx - chip_meters/2, cx + chip_meters/2
            miny, maxy = cy - chip_meters/2, cy + chip_meters/2
            
            window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            
            # Read & Resample (Bilinear)
            height_data = src.read(
                1, window=window, 
                out_shape=(chip_size_px, chip_size_px), 
                resampling=rasterio.enums.Resampling.bilinear,
                boundless=True # Handle edge cases with padding
            )
            
            # Generate the 3-Channel Tensor (Height/Slope/Roughness)
            tensor = self.datagen.calculate_geometry(height_data)
            
            # Metadata for reconstruction
            meta = {
                "transform": rasterio.transform.from_bounds(minx, miny, maxx, maxy, chip_size_px, chip_size_px),
                "crs": src.crs,
                "bounds": [[minx, miny], [maxx, maxy]]
            }
            
            return tensor, meta
