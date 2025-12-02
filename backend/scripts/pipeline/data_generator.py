import os
import json
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import box, shape
from pathlib import Path
import cv2
import scipy.ndimage
import random
import traceback
from tqdm import tqdm

class DataGenerator:
    def __init__(self, config):
        self.config = config

    def setup_dirs(self):
        for d in ["images", "masks", "debug_vis"]:
            Path(f"{self.config['OUTPUT_DIR']}/{d}").mkdir(parents=True, exist_ok=True)

    def calculate_geometry(self, height_array):
        # Ensure float64 to prevent overflow in gradient/slope calc
        height_array = height_array.astype(np.float64)

        # 1. Apply Gaussian Smoothing (As promised in README)
        # sigma=0.5 pixels helps reduce Aliasing/Lidar noise
        smoothed_height = scipy.ndimage.gaussian_filter(height_array, sigma=0.5)

        # 2. Slope (Use smoothed data)
        dy, dx = np.gradient(smoothed_height)
        slope = np.sqrt(dx**2 + dy**2)
        slope_norm = np.clip(slope, 0, 1.0) * 255
        
        # 3. Roughness (Use smoothed data)
        # Laplace filter highlights edges/ridges
        curvature = scipy.ndimage.laplace(smoothed_height)
        roughness_norm = np.clip(np.abs(curvature), 0, 1.0) * 255
        
        # 4. Height (Use Raw or Smoothed depending on preference, Raw is usually fine for absolute height)
        # CRITICAL: 20m clip is too low for Montreal urban density (3-4 story plexes + downtown). 
        # Suggest increasing to 50m.
        height_norm = np.clip(height_array, 0, 50) / 50.0 * 255
        
        return np.dstack([
            height_norm.astype(np.uint8),
            slope_norm.astype(np.uint8),
            roughness_norm.astype(np.uint8)
        ])

    def save_debug_visualization(self, uid, geo_tensor, mask):
        # Visualize Height Channel (Red)
        height_vis = cv2.applyColorMap(geo_tensor[:,:,0], cv2.COLORMAP_JET)
        
        # Visualize Mask (Random Colors)
        mask_vis = np.zeros_like(height_vis)
        unique_ids = np.unique(mask)
        for i in unique_ids:
            if i == 0: continue
            np.random.seed(int(i)) 
            color = np.random.randint(50, 255, 3).tolist()
            mask_vis[mask == i] = color
        
        # Stack: Geometry | Mask
        debug_img = np.hstack([height_vis, mask_vis])
        cv2.imwrite(f"{self.config['OUTPUT_DIR']}/debug_vis/{uid}.jpg", debug_img)

    def run(self, tile_list=None):
        self.setup_dirs()
        
        print("1. Loading Selected Tiles...")
        # We NO LONGER load the full vector file here.
        # We will load per-tile chunks inside the loop. to process
        selected_tiles = []
        if tile_list:
            print(f"   Using provided tile list ({len(tile_list)} tiles).")
            selected_tiles = tile_list
        else:
            if not os.path.exists(self.config['OUTPUT_TILES_PATH']):
                print(f"❌ Selected tiles list not found: {self.config['OUTPUT_TILES_PATH']}")
                return
            
            with open(self.config['OUTPUT_TILES_PATH']) as f:
                selected_tiles = json.load(f)
            print(f"   Found {len(selected_tiles)} tiles to process.")

        total_chips_generated = 0
        
        for tile_info in tqdm(selected_tiles):
            filename = tile_info['filename']
            lidar_path = os.path.join(self.config['CACHE_DIR'], filename)
            
            if not os.path.exists(lidar_path):
                print(f"   ⚠️ File not found in cache: {filename}")
                continue

            try:
                # LOAD VECTOR CHUNK
                tile_id = tile_info.get('id', filename.replace('lidar_', '').replace('.tif', ''))
                chunk_path = os.path.join(self.config['CACHE_DIR'], "vectors", f"vectors_{tile_id}.gpkg")
                
                if not os.path.exists(chunk_path):
                    print(f"   ⚠️ Vector chunk not found: {chunk_path}")
                    continue
                
                buildings_in_tile = gpd.read_file(chunk_path)
                
                # Force MTM Zone 8 (Montreal standard metric grid) or UTM
                # EPSG:32188 is MTM Zone 8 (Quebec)
                METRIC_CRS = "EPSG:32188" 
                
                if buildings_in_tile.crs.to_string() != METRIC_CRS:
                    buildings_in_tile = buildings_in_tile.to_crs(METRIC_CRS)
                
                # Ensure instance_id exists (it should be in the chunk, but re-verify)
                if 'instance_id' not in buildings_in_tile.columns:
                     buildings_in_tile['instance_id'] = range(1, len(buildings_in_tile) + 1)

                with rasterio.open(lidar_path) as src_lidar:
                    # Find buildings in this specific tile
                    # Since we loaded the chunk, 'buildings_in_tile' ALREADY contains only relevant buildings.
                    # We can skip the spatial query against the full dataset.
                    
                    if buildings_in_tile.empty: continue

                    # Sample 50 buildings
                    sample_buildings = buildings_in_tile.sample(n=min(50, len(buildings_in_tile)), random_state=self.config['RANDOM_SEED'])
                    
                    for _, b in sample_buildings.iterrows():
                        # Check reprojection
                        geom = b.geometry
                        # If LiDAR is not in METRIC_CRS, we need to transform.
                        # Usually LiDAR is in MTM8 too. Let's assume it is or handle it.
                        if src_lidar.crs.to_string() != METRIC_CRS:
                             from rasterio.warp import transform_geom
                             geom_dict = transform_geom(METRIC_CRS, src_lidar.crs, geom.__geo_interface__)
                             geom = shape(geom_dict)
                        
                        # ... rest of logic uses 'geom' ...
                        cx, cy = geom.centroid.x, geom.centroid.y
                        
                        # Define Optimized Zoom
                        TARGET_RESOLUTION = 0.2  # 20cm per pixel
                        CHIP_SIZE_METERS = self.config['CHIP_SIZE'] * TARGET_RESOLUTION
                        
                        # 1. Calculate the window in METERS (Physical Space)
                        minx = cx - CHIP_SIZE_METERS / 2
                        maxx = cx + CHIP_SIZE_METERS / 2
                        miny = cy - CHIP_SIZE_METERS / 2
                        maxy = cy + CHIP_SIZE_METERS / 2
                        
                        # 2. Convert Meters -> Source Pixels
                        lidar_window = window_from_bounds(minx, miny, maxx, maxy, transform=src_lidar.transform)
                        
                        # 3. Read and RESAMPLE to 512x512
                        height = src_lidar.read(
                            1,
                            window=lidar_window,
                            out_shape=(self.config['CHIP_SIZE'], self.config['CHIP_SIZE']),
                            resampling=Resampling.bilinear,
                            boundless=True
                        )
                        
                        # GEOMETRIC FEATURES
                        geo_tensor = self.calculate_geometry(height)
                        
                        # MASK GENERATION
                        # Update transform for the new resampled grid
                        win_transform = transform_from_bounds(minx, miny, maxx, maxy, self.config['CHIP_SIZE'], self.config['CHIP_SIZE'])
                        bounds = (minx, miny, maxx, maxy)
                        box_geom = box(*bounds)
                        
                        # Intersection check for mask generation
                        # We need to check which buildings from the chunk intersect this specific chip window
                        # The chunk is already small, so this is fast.
                        precise = buildings_in_tile[buildings_in_tile.intersects(box_geom)]
                        
                        if precise.empty: continue
                        
                        # Erosion & Burning
                        shapes_to_burn = []
                        for g, i in zip(precise.geometry, precise.instance_id):
                            # Ensure geometry is in same CRS as window for burning? 
                            # Rasterize expects coords in the transform's CRS.
                            # If src_lidar is MTM8 and vectors are MTM8, we are good.
                            eroded = g.buffer(-self.config['MASK_EROSION'])
                            if not eroded.is_empty: shapes_to_burn.append((eroded, i))
                        
                        if not shapes_to_burn: continue

                        mask = rasterize(shapes_to_burn, out_shape=(self.config['CHIP_SIZE'], self.config['CHIP_SIZE']), transform=win_transform, fill=0, dtype='int32')
                        
                        local_mask = np.zeros_like(mask, dtype='int32')
                        uids = np.unique(mask[mask != 0])
                        for i, uid in enumerate(uids): local_mask[mask == uid] = i + 1

                        # SAVE
                        # Use tile ID (from filename if id not present) + building ID
                        # tile_id already defined above
                        uid = f"t{tile_id}_b{b['instance_id']}"
                        
                        cv2.imwrite(f"{self.config['OUTPUT_DIR']}/images/{uid}.png", geo_tensor)
                        cv2.imwrite(f"{self.config['OUTPUT_DIR']}/masks/{uid}.png", local_mask.astype(np.uint16))
                        
                        self.save_debug_visualization(uid, geo_tensor, local_mask)
                        total_chips_generated += 1

            except Exception as e:
                print(f"❌ Error processing tile {filename}: {e}")
                traceback.print_exc()
                continue

        print(f"Done. Generated {total_chips_generated} geometric chips.")
