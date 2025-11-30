import os
import json
import requests
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import box, shape
from pathlib import Path
from tqdm import tqdm
import cv2
import scipy.ndimage
import random

# --- CONFIGURATION ---
VECTOR_PATH = "data/raw/referentiel_batiments.gpkg" 
LIDAR_INDEX_PATH = "data/raw/lidar_index.geojson"

CACHE_DIR = "data/raw/cache"
OUTPUT_DIR = "data/processed/mask2former_lidar_only"

CHIP_SIZE = 512 
MAX_TILES = 15        
MASK_EROSION = 0.5    
RANDOM_SEED = 42

# Approximate Bounding Box for Montreal Island (Lat/Lon EPSG:4326)
# This ensures we grab tiles from the city center
MONTREAL_BBOX = box(-73.9, 45.4, -73.5, 45.7)

def setup_dirs():
    for d in ["images", "masks", "debug_vis"]:
        Path(f"{OUTPUT_DIR}/{d}").mkdir(parents=True, exist_ok=True)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

def download_file(url, local_filename):
    local_path = os.path.join(CACHE_DIR, local_filename)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 10240: return local_path
    
    print(f"   ⬇️ Downloading: {local_filename}...")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            if r.status_code != 200: 
                print(f"   ❌ HTTP Error {r.status_code} for {url}")
                return None
            temp_path = local_path + ".tmp"
            with open(temp_path, 'wb') as f, tqdm(
                desc=local_filename,
                total=int(r.headers.get('content-length', 0)),
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(8192): 
                    size = f.write(chunk)
                    bar.update(size)
            if os.path.getsize(temp_path) < 1024: 
                os.remove(temp_path)
                return None
            os.rename(temp_path, local_path)
            return local_path
    except Exception as e: 
        print(f"   ❌ Error: {e}")
        return None

def load_lidar_index(path):
    print("2. Loading LiDAR Index...")
    with open(path) as f: data = json.load(f)
    # Check if CRS exists, default to EPSG:4326 (Lat/Lon) which is standard for GeoJSON
    crs = data.get('crs', {}).get('properties', {}).get('name', 'EPSG:4326')
    return gpd.GeoDataFrame.from_features(data['features'], crs=crs)

def calculate_geometry(height_array):
    # 1. Slope
    dy, dx = np.gradient(height_array)
    slope = np.sqrt(dx**2 + dy**2)
    slope_norm = np.clip(slope, 0, 1.0) * 255
    
    # 2. Roughness
    curvature = scipy.ndimage.laplace(height_array)
    roughness_norm = np.clip(np.abs(curvature), 0, 1.0) * 255
    
    # 3. Height
    height_norm = np.clip(height_array, 0, 20) / 20.0 * 255
    
    return np.dstack([
        height_norm.astype(np.uint8),
        slope_norm.astype(np.uint8),
        roughness_norm.astype(np.uint8)
    ])

def generate_data():
    setup_dirs()
    print("1. Loading Vectors...")
    buildings = gpd.read_file(VECTOR_PATH)
    buildings['instance_id'] = range(1, len(buildings) + 1)
    
    lidar_index = load_lidar_index(LIDAR_INDEX_PATH)
    
    # Project to Metric CRS (e.g. MTM Zone 8 EPSG:32188) for accurate sampling
    # We use the vector data's CRS as the target
    target_crs = buildings.crs
    if lidar_index.crs != target_crs:
        print("   Reprojecting LiDAR index to match vectors...")
        lidar_index = lidar_index.to_crs(target_crs)
    
    print("3. Calculating Density per LiDAR Tile...")
    # Spatial Join: Count buildings per LiDAR tile
    lidar_with_counts = gpd.sjoin(lidar_index, buildings, how="inner", predicate="intersects")
    tile_counts = lidar_with_counts.groupby(lidar_with_counts.index).size().reset_index(name='count')
    
    # --- NEW: Montreal Specific Selection ---
    # Convert Montreal BBOX to Target CRS
    montreal_poly = gpd.GeoSeries([MONTREAL_BBOX], crs="EPSG:4326").to_crs(target_crs).iloc[0]
    
    # Find tiles that intersect Montreal
    montreal_indices = lidar_index[lidar_index.intersects(montreal_poly)].index.tolist()
    
    # Filter Montreal tiles that actually have buildings (from our counts)
    valid_montreal = list(set(montreal_indices) & set(tile_counts['index_right'].tolist()))
    
    print(f"   Found {len(valid_montreal)} valid tiles covering Montreal.")

    # --- Stratified Sampling ---
    high = tile_counts[tile_counts['count'] > 2000]['index_right'].tolist()
    med = tile_counts[(tile_counts['count'] >= 200) & (tile_counts['count'] <= 2000)]['index_right'].tolist()
    low = tile_counts[tile_counts['count'] < 200]['index_right'].tolist()
    
    print(f"   High: {len(high)} | Med: {len(med)} | Low: {len(low)}")
    
    random.seed(RANDOM_SEED)
    selected_indices = set()
    
    # 1. Force 3 Montreal Tiles
    if valid_montreal:
        selected_indices.update(random.sample(valid_montreal, min(3, len(valid_montreal))))
        print(f"   ✅ Added {len(selected_indices)} Montreal tiles.")
        
    # 2. Fill rest with Stratified Random
    remaining_needed = MAX_TILES - len(selected_indices)
    if remaining_needed > 0:
        target_per_cat = max(1, remaining_needed // 3)
        
        # Helper to pick new tiles
        def pick_new(pool, n):
            available = list(set(pool) - selected_indices)
            return random.sample(available, min(len(available), n))

        selected_indices.update(pick_new(high, target_per_cat))
        selected_indices.update(pick_new(med, target_per_cat))
        selected_indices.update(pick_new(low, target_per_cat))
        
        # Final Top-up
        still_needed = MAX_TILES - len(selected_indices)
        if still_needed > 0:
            all_avail = set(tile_counts['index_right'].tolist()) - selected_indices
            selected_indices.update(random.sample(list(all_avail), min(len(all_avail), still_needed)))
            
    print(f"   🎯 Final Selection: {len(selected_indices)} tiles.")

    processed_tiles = 0
    total_chips = 0
    
    # Filter the main index
    # We convert set to list to index into dataframe
    selected_lidar_rows = lidar_index.loc[list(selected_indices)]
    
    for idx, lidar_row in tqdm(selected_lidar_rows.iterrows(), total=len(selected_lidar_rows)):
        
        # Get URL (MHC is priority)
        lidar_url = lidar_row.get('MHC') or lidar_row.get('MNS') or lidar_row.get('MNT')
        if not lidar_url: continue
        
        # Use the original index ID for filename to avoid collisions
        original_id = idx 
        lidar_path = download_file(lidar_url, f"lidar_{original_id}.tif")
        if not lidar_path: continue
        
        processed_tiles += 1

        try:
            with rasterio.open(lidar_path) as src_lidar:
                # Find buildings in this specific tile
                tile_box = box(*src_lidar.bounds)
                
                possible_matches = list(buildings.sindex.intersection(tile_box.bounds))
                buildings_in_tile = buildings.iloc[possible_matches]
                buildings_in_tile = buildings_in_tile[buildings_in_tile.intersects(tile_box)]
                
                if buildings_in_tile.empty: continue

                # Sample 50 buildings
                sample_buildings = buildings_in_tile.sample(n=min(50, len(buildings_in_tile)), random_state=RANDOM_SEED)
                
                for _, b in sample_buildings.iterrows():
                    # Check if building geometry needs reprojection to match Raster
                    geom = b.geometry
                    if buildings.crs != src_lidar.crs:
                        from rasterio.warp import transform_geom
                        geom_dict = transform_geom(buildings.crs, src_lidar.crs, geom.__geo_interface__)
                        geom = shape(geom_dict)

                    cx, cy = geom.centroid.x, geom.centroid.y
                    py, px = src_lidar.index(cx, cy)
                    
                    win_x = max(0, min(px - CHIP_SIZE // 2, src_lidar.width - CHIP_SIZE))
                    win_y = max(0, min(py - CHIP_SIZE // 2, src_lidar.height - CHIP_SIZE))
                    window = Window(win_x, win_y, CHIP_SIZE, CHIP_SIZE)
                    
                    # Read Height
                    height = src_lidar.read(1, window=window)
                    
                    # GEOMETRIC FEATURES
                    geo_tensor = calculate_geometry(height)
                    
                    # MASK GENERATION
                    win_transform = src_lidar.window_transform(window)
                    bounds = rasterio.windows.bounds(window, src_lidar.transform)
                    box_geom = box(*bounds)
                    
                    # Handle Reprojection for Precision Check
                    if buildings.crs != src_lidar.crs:
                        precise = buildings_in_tile.to_crs(src_lidar.crs)
                        precise = precise[precise.intersects(box_geom)]
                    else:
                        precise = buildings_in_tile[buildings_in_tile.intersects(box_geom)]
                    
                    if precise.empty: continue
                    
                    # Prepare shapes for rasterize
                    shapes_to_burn = []
                    for g, i in zip(precise.geometry, precise.instance_id):
                        eroded = g.buffer(-MASK_EROSION)
                        if not eroded.is_empty: shapes_to_burn.append((eroded, i))
                    
                    if not shapes_to_burn: continue

                    mask = rasterize(shapes_to_burn, out_shape=(CHIP_SIZE, CHIP_SIZE), transform=win_transform, fill=0, dtype='int32')
                    
                    local_mask = np.zeros_like(mask, dtype='int32')
                    uids = np.unique(mask[mask != 0])
                    for i, uid in enumerate(uids): local_mask[mask == uid] = i + 1

                    # SAVE
                    uid = f"t{original_id}_b{b['instance_id']}"
                    cv2.imwrite(f"{OUTPUT_DIR}/images/{uid}.png", geo_tensor)
                    cv2.imwrite(f"{OUTPUT_DIR}/masks/{uid}.png", local_mask.astype(np.uint16))
                    
                    # Debug
                    vis_img = np.hstack([geo_tensor, cv2.applyColorMap(local_mask.astype(np.uint8)*50, cv2.COLORMAP_JET)])
                    cv2.imwrite(f"{OUTPUT_DIR}/debug_vis/{uid}.jpg", vis_img)
                    
                    total_chips += 1

        except Exception as e:
            continue

    print(f"Done. Generated {total_chips} geometric chips.")

if __name__ == "__main__":
    generate_data()