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
import fiona
import random

# --- CONFIGURATION ---
# Adjust paths as needed for your project structure
VECTOR_PATH = "data/raw/referentiel_batiments.gpkg" 
ORTHO_INDEX_PATH = "data/raw/ortho_index_monteregie_2023.gpkg"
LIDAR_INDEX_PATH = "data/raw/lidar_index.geojson"

CACHE_DIR = "data/raw/cache"
OUTPUT_DIR = "data/processed/mask2former_train"

CHIP_SIZE = 512 
MAX_TILES = 6         # Process only 6 tiles for portfolio speed
MASK_EROSION = 0.5    # Shrink polygons by 0.5m to fix "jittery" edges
RANDOM_SEED = 42      # Deterministic sampling

def setup_dirs():
    """Creates necessary directories for training data and cache."""
    for d in ["images", "masks", "debug_vis"]:
        Path(f"{OUTPUT_DIR}/{d}").mkdir(parents=True, exist_ok=True)
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

def download_file(url, local_filename):
    """
    Robust file downloader with progress bar and integrity checks.
    """
    local_path = os.path.join(CACHE_DIR, local_filename)
    
    # 1. Check if file exists and is valid (> 10KB)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 10240:
        return local_path
    
    print(f"   ⬇️ Downloading: {local_filename}...")
    
    try:
        # Stream download with 60s timeout
        with requests.get(url, stream=True, timeout=60) as r:
            if r.status_code != 200:
                print(f"   ❌ HTTP Error {r.status_code} for {url}")
                return None
            
            total_size = int(r.headers.get('content-length', 0))
            temp_path = local_path + ".tmp"
            
            # Write to temp file
            with open(temp_path, 'wb') as f, tqdm(
                desc=local_filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
            
            # Validate size (small files are usually error HTML pages)
            if os.path.getsize(temp_path) < 1024:
                print(f"   ❌ File too small: {local_filename}")
                os.remove(temp_path)
                return None

            # Atomic move
            if os.path.exists(local_path): os.remove(local_path)
            os.rename(temp_path, local_path)
            return local_path

    except Exception as e:
        print(f"   ❌ Download Failed: {e}")
        return None

def load_lidar_index(path):
    """Parses custom LiDAR GeoJSON."""
    print("2. Loading LiDAR Index...")
    with open(path) as f:
        data = json.load(f)
    features = data['features']
    geoms = [shape(f['geometry']) for f in features]
    props = [f['properties'] for f in features]
    # Handle CRS key usually found in GeoJSON
    crs = data.get('crs', {}).get('properties', {}).get('name', 'EPSG:4326')
    return gpd.GeoDataFrame(props, geometry=geoms, crs=crs)

def load_ortho_index_smart(path):
    """Finds the correct data layer in a multi-layer GPKG."""
    print(f"3. Loading Ortho Index from {os.path.basename(path)}...")
    try:
        layers = fiona.listlayers(path)
    except Exception as e:
        print(f"   ❌ Failed to list layers: {e}")
        return None

    selected_layer = layers[0]
    # Simple heuristic to find the content layer
    for layer in layers:
        if "orthophoto" in layer.lower() and "index_tuilage" not in layer.lower():
            selected_layer = layer
            break
            
    print(f"   -> Using layer: '{selected_layer}'")
    return gpd.read_file(path, layer=selected_layer)

def save_debug_visualization(uid, rgb, height, vari, mask):
    """
    Saves a human-readable visual check.
    Left: RGB, Center: Height Heatmap, Right: Random Colored Mask
    """
    # 1. Height Heatmap
    h_vis = cv2.applyColorMap(height, cv2.COLORMAP_JET)
    
    # 2. Mask Visualization
    mask_vis = np.zeros_like(rgb)
    unique_ids = np.unique(mask)
    for i in unique_ids:
        if i == 0: continue
        np.random.seed(int(i)) 
        color = np.random.randint(50, 255, 3).tolist()
        mask_vis[mask == i] = color
    
    # Resize VARI to match others (it's 1 channel, need 3 for stacking)
    # Actually cv2.applyColorMap returns 3 channels, so we are good.
    vari_vis = cv2.applyColorMap(vari, cv2.COLORMAP_TWILIGHT) # Purple-Green map for vegetation

    # 3. Stack horizontally
    debug_img = np.hstack([
        rgb.astype(np.uint8), 
        h_vis.astype(np.uint8), 
        vari_vis.astype(np.uint8),
        mask_vis.astype(np.uint8)
    ])
    cv2.imwrite(f"{OUTPUT_DIR}/debug_vis/{uid}.jpg", debug_img)

def generate_data():
    setup_dirs()
    
    print("1. Loading Vectors...")
    buildings = gpd.read_file(VECTOR_PATH)
    # Assign unique IDs for Instance Segmentation
    buildings['instance_id'] = range(1, len(buildings) + 1)
    
    lidar_index = load_lidar_index(LIDAR_INDEX_PATH)
    ortho_index = load_ortho_index_smart(ORTHO_INDEX_PATH)
    
    # URL Column Logic
    possible_cols = ['lien_telechargement', 'url', 'URL', 'lien', 'path', 'URL_TUILE']
    url_col = next((c for c in possible_cols if c in ortho_index.columns), None)
    if not url_col:
        print(f"   ❌ ERROR: No URL column found in {ortho_index.columns}")
        return

    # CRS Sync (Ensure everything matches Ortho projection)
    target_crs = ortho_index.crs
    if buildings.crs != target_crs: buildings = buildings.to_crs(target_crs)
    if lidar_index.crs != target_crs: lidar_index = lidar_index.to_crs(target_crs)

    print("4. Calculating Building Density (Stratified Sampling)...")
    
    # Spatial Join to find which buildings are in which tiles
    buildings_with_tiles = gpd.sjoin(buildings, ortho_index, how="inner", predicate="intersects")
    tile_counts = buildings_with_tiles.groupby('index_right').size().reset_index(name='count')
    
    # Classify Tiles
    high = tile_counts[tile_counts['count'] > 500]['index_right'].tolist()
    med = tile_counts[(tile_counts['count'] >= 50) & (tile_counts['count'] <= 500)]['index_right'].tolist()
    low = tile_counts[tile_counts['count'] < 50]['index_right'].tolist()
    
    print(f"   High: {len(high)} | Med: {len(med)} | Low: {len(low)}")

    # Select Tiles Deterministically with Fallback
    random.seed(RANDOM_SEED)
    target_per_cat = max(1, MAX_TILES // 3) # At least 1 per cat
    
    selected_indices = []
    
    # Try to pick evenly
    selected_indices.extend(random.sample(high, min(len(high), target_per_cat)))
    selected_indices.extend(random.sample(med, min(len(med), target_per_cat)))
    selected_indices.extend(random.sample(low, min(len(low), target_per_cat)))
    
    # Fill remaining slots if any category was short (or due to rounding)
    remaining_needed = MAX_TILES - len(selected_indices)
    if remaining_needed > 0:
        print(f"   ⚠️ Need {remaining_needed} more tiles to reach {MAX_TILES}. Picking randomly from leftovers...")
        all_possible = set(tile_counts['index_right'].tolist())
        already_picked = set(selected_indices)
        leftovers = list(all_possible - already_picked)
        if leftovers:
            selected_indices.extend(random.sample(leftovers, min(len(leftovers), remaining_needed)))
    
    # Filter groups
    tile_groups = buildings_with_tiles[buildings_with_tiles['index_right'].isin(selected_indices)].groupby('index_right')
    
    print(f"   🎯 Selected {len(tile_groups)} tiles for processing.")

    processed_tiles = 0
    total_chips_generated = 0
    
    # Main Loop
    for tile_idx, building_subset in tqdm(tile_groups, total=len(tile_groups)):
        
        ortho_row = ortho_index.iloc[tile_idx]
        ortho_url = ortho_row[url_col]
        if not ortho_url: continue
        
        # A. Download Ortho
        ortho_path = download_file(ortho_url, f"ortho_{tile_idx}.tif")
        if not ortho_path: continue

        # B. Find & Download LiDAR
        lidar_candidates = lidar_index[lidar_index.intersects(ortho_row.geometry)]
        if lidar_candidates.empty: continue
        
        lidar_info = lidar_candidates.iloc[0]
        # Try different keys for the URL
        lidar_url = lidar_info.get('MHC') or lidar_info.get('MNS') or lidar_info.get('MNT')
        if not lidar_url: continue 

        lidar_path = download_file(lidar_url, f"lidar_{tile_idx}.tif")
        if not lidar_path: continue

        processed_tiles += 1

        try:
            with rasterio.open(ortho_path) as src_ortho, rasterio.open(lidar_path) as src_lidar:
                
                # Sample 50 buildings per tile to avoid bias
                sample_buildings = building_subset.sample(n=min(50, len(building_subset)), random_state=RANDOM_SEED)
                
                for _, b in sample_buildings.iterrows():
                    # Center window on building centroid
                    cx, cy = b.geometry.centroid.x, b.geometry.centroid.y
                    py, px = src_ortho.index(cx, cy)
                    
                    # Window bounds check
                    win_x = max(0, min(px - CHIP_SIZE // 2, src_ortho.width - CHIP_SIZE))
                    win_y = max(0, min(py - CHIP_SIZE // 2, src_ortho.height - CHIP_SIZE))
                    window = Window(win_x, win_y, CHIP_SIZE, CHIP_SIZE)
                    
                    # 1. Read RGB
                    rgb = src_ortho.read(window=window, out_shape=(3, CHIP_SIZE, CHIP_SIZE))
                    if np.all(rgb == 0): continue
                    
                    # 2. FEATURE ENGINEERING: VARI (Vegetation Index)
                    rgb_norm = rgb.astype(np.float32) / 255.0
                    r, g, b_band = rgb_norm[0], rgb_norm[1], rgb_norm[2]
                    
                    # VARI Formula: (G - R) / (G + R - B)
                    numerator = g - r
                    denominator = g + r - b_band + 1e-6
                    vari = numerator / denominator
                    
                    # Normalize VARI (-1 to 1) -> (0 to 255)
                    vari_norm = np.clip(vari, -1, 1)
                    vari_img = ((vari_norm + 1) / 2 * 255).astype(np.uint8)

                    # Prepare RGB for OpenCV
                    rgb_img = np.moveaxis(rgb, 0, -1)
                    rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

                    # 3. Read LiDAR (Resample to match Ortho)
                    bounds = rasterio.windows.bounds(window, src_ortho.transform)
                    lidar_window = from_bounds(*bounds, transform=src_lidar.transform)
                    
                    height = src_lidar.read(
                        1, 
                        window=lidar_window, 
                        out_shape=(CHIP_SIZE, CHIP_SIZE),
                        resampling=Resampling.bilinear
                    )
                    
                    # Normalize Height (0-20m -> 0-255)
                    h_norm = np.clip(height, 0, 20) / 20.0 * 255
                    h_img = h_norm.astype(np.uint8)
                    
                    # 4. Create 5-CHANNEL TENSOR (BGR + Height + VARI)
                    combined_img = np.dstack([rgb_bgr, h_img, vari_img])

                    # 5. Generate Instance Mask
                    box_geom = box(*bounds)
                    precise = building_subset[building_subset.intersects(box_geom)]
                    
                    if precise.empty: continue

                    win_transform = src_ortho.window_transform(window)
                    
                    # EROSION LOGIC
                    eroded_shapes = []
                    for g, i in zip(precise.geometry, precise.instance_id):
                        eroded_geom = g.buffer(-MASK_EROSION)
                        if not eroded_geom.is_empty:
                            eroded_shapes.append((eroded_geom, i))
                            
                    if not eroded_shapes: continue

                    # Burn Mask
                    mask = rasterize(
                        shapes=eroded_shapes,
                        out_shape=(CHIP_SIZE, CHIP_SIZE),
                        transform=win_transform,
                        fill=0,
                        dtype='int32'
                    )
                    
                    # Relocalize IDs (1..N)
                    local_mask = np.zeros_like(mask, dtype='int32')
                    uids = np.unique(mask[mask != 0])
                    for i, uid in enumerate(uids):
                        local_mask[mask == uid] = i + 1

                    # 6. Save Data
                    uid = f"t{tile_idx}_b{b['instance_id']}"
                    
                    # Training Data
                    cv2.imwrite(f"{OUTPUT_DIR}/images/{uid}.tif", combined_img)
                    cv2.imwrite(f"{OUTPUT_DIR}/masks/{uid}.png", local_mask.astype(np.uint16))
                    
                    # Debug Visualization
                    save_debug_visualization(uid, rgb_bgr, h_img, vari_img, local_mask)
                    
                    total_chips_generated += 1

        except Exception as e:
            # print(f"Error tile {tile_idx}: {e}")
            continue

    print(f"\n🚀 FLYWHEEL COMPLETE")
    print(f"   Processed Tiles: {processed_tiles}")
    print(f"   Total Training Chips: {total_chips_generated}")
    print(f"   Location: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_data()