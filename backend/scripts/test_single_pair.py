import os
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
import cv2
import random

# --- CONFIGURATION (Adjust filenames as needed) ---
# Assuming these are the files you already have downloaded
ORTHO_PATH = "data/raw/cache/ortho_0.tif" 
LIDAR_PATH = "data/raw/cache/lidar_0.tif"
VECTOR_PATH = "data/raw/referentiel_batiments.gpkg"

OUTPUT_DIR = "data/processed/test_single_run"
CHIP_SIZE = 512 
MASK_EROSION = 0.5 # Shrink polygons by 0.5 meters to remove "jittery edges"

def setup_dirs():
    for d in ["images", "masks", "debug_vis"]:
        Path(f"{OUTPUT_DIR}/{d}").mkdir(parents=True, exist_ok=True)

def save_debug_visualization(uid, rgb, height, mask):
    """Saves human-readable JPEGs to verify alignment."""
    # 1. Height Heatmap
    h_vis = cv2.applyColorMap(height, cv2.COLORMAP_JET)
    
    # 2. Mask overlay
    # Create distinct colors for each instance
    # RGB image needs to be float or uint8. Mask IDs are int.
    mask_vis = np.zeros_like(rgb)
    unique_ids = np.unique(mask)
    
    print(f"   Unique IDs in mask {uid}: {unique_ids}") # DEBUG PRINT

    for i in unique_ids:
        if i == 0: continue
        # Deterministic color per ID for stability
        np.random.seed(int(i)) # Ensure seed is int
        color = np.random.randint(50, 255, 3).tolist()
        # mask is (H, W), mask_vis is (H, W, 3). Broadcasting handles assignment.
        mask_vis[mask == i] = color
    
    # 3. Stack: RGB | Height | Mask
    # Ensure all are uint8 for stacking
    debug_img = np.hstack([rgb.astype(np.uint8), h_vis.astype(np.uint8), mask_vis.astype(np.uint8)])
    cv2.imwrite(f"{OUTPUT_DIR}/debug_vis/{uid}.jpg", debug_img)
    print(f"   Saved debug image: {OUTPUT_DIR}/debug_vis/{uid}.jpg")

def run_test():
    setup_dirs()
    
    if not os.path.exists(ORTHO_PATH) or not os.path.exists(LIDAR_PATH):
        print("❌ Error: Input files not found. Check paths.")
        return

    print("1. Loading Vectors...")
    buildings = gpd.read_file(VECTOR_PATH)
    # Assign Mock Instance IDs if missing
    if 'instance_id' not in buildings.columns:
        buildings['instance_id'] = range(1, len(buildings) + 1)
    
    # Build Spatial Index
    b_sindex = buildings.sindex

    print(f"2. Processing Test Pair...")
    print(f"   Ortho: {ORTHO_PATH}")
    print(f"   LiDAR: {LIDAR_PATH}")

    try:
        with rasterio.open(ORTHO_PATH) as src_ortho, rasterio.open(LIDAR_PATH) as src_lidar:
            
            # Sync CRS on the fly
            if buildings.crs != src_ortho.crs:
                print(f"   Reprojecting vectors from {buildings.crs} to {src_ortho.crs}...")
                buildings = buildings.to_crs(src_ortho.crs)
                b_sindex = buildings.sindex # Rebuild index after reprojection

            # Get bounds of the Ortho Tile
            tile_bounds = src_ortho.bounds
            tile_box = box(*tile_bounds)
            
            # Filter buildings to ONLY those in this tile (Optimization)
            print("   Filtering buildings in tile area...")
            possible_matches = list(b_sindex.intersection(tile_box.bounds))
            buildings_in_tile = buildings.iloc[possible_matches]
            buildings_in_tile = buildings_in_tile[buildings_in_tile.intersects(tile_box)]
            
            if buildings_in_tile.empty:
                print("   ❌ No buildings found in this tile area!")
                return

            print(f"   Found {len(buildings_in_tile)} buildings. generating 5 samples...")
            
            # Pick 5 random buildings to test
            sample_buildings = buildings_in_tile.sample(n=min(5, len(buildings_in_tile)))
            
            for idx, (_, b) in enumerate(sample_buildings.iterrows()):
                # --- CENTERING LOGIC ---
                cx, cy = b.geometry.centroid.x, b.geometry.centroid.y
                
                # Check if centroid is actually within bounds (it might be outside if polygon overlaps edge)
                if not (src_ortho.bounds.left < cx < src_ortho.bounds.right and src_ortho.bounds.bottom < cy < src_ortho.bounds.top):
                     print(f"   Skipping building {b['instance_id']} - centroid outside tile bounds.")
                     continue

                py, px = src_ortho.index(cx, cy)
                
                # Calculate Window
                win_x = max(0, min(px - CHIP_SIZE // 2, src_ortho.width - CHIP_SIZE))
                win_y = max(0, min(py - CHIP_SIZE // 2, src_ortho.height - CHIP_SIZE))
                window = Window(win_x, win_y, CHIP_SIZE, CHIP_SIZE)
                
                # --- READ DATA ---
                # 1. RGB
                rgb = src_ortho.read(window=window, out_shape=(3, CHIP_SIZE, CHIP_SIZE))
                rgb_img = np.moveaxis(rgb, 0, -1)
                rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

                # 2. LiDAR
                bounds = rasterio.windows.bounds(window, src_ortho.transform)
                lidar_window = from_bounds(*bounds, transform=src_lidar.transform)
                
                # Upsample: 1m -> 20cm (via bilinear) to match Ortho
                height = src_lidar.read(
                    1, 
                    window=lidar_window, 
                    out_shape=(CHIP_SIZE, CHIP_SIZE),
                    resampling=Resampling.bilinear
                )
                
                # Normalize Height (0-20m -> 0-255)
                h_norm = np.clip(height, 0, 20) / 20.0 * 255
                h_img = h_norm.astype(np.uint8)
                
                # 3. Combine (4-Channel)
                combined_img = np.dstack([rgb_bgr, h_img])

                # --- GENERATE MASKS ---
                box_geom = box(*bounds)
                # Filter specific buildings in this window
                # Note: We query the full 'buildings_in_tile' dataset, not just the one sample building
                # to catch neighbors in the same chip.
                precise = buildings_in_tile[buildings_in_tile.intersects(box_geom)]
                
                if precise.empty:
                    print(f"   ⚠️ No buildings found in window for sample {idx}. Strange.")
                    continue

                win_transform = src_ortho.window_transform(window)
                
                # Apply Erosion (Negative Buffer)
                eroded_shapes = []
                for g, i in zip(precise.geometry, precise.instance_id):
                    eroded_geom = g.buffer(-MASK_EROSION) # Shrink by 0.5m
                    if not eroded_geom.is_empty:
                        eroded_shapes.append((eroded_geom, i))

                if not eroded_shapes:
                    print("   ⚠️ Skipped mask generation: Polygons vanished after erosion (too small).")
                    continue

                # Burn Instance IDs
                mask = rasterize(
                    shapes=eroded_shapes,
                    out_shape=(CHIP_SIZE, CHIP_SIZE),
                    transform=win_transform,
                    fill=0,
                    dtype='int32'
                )
                
                # Relabel for Training (1..N)
                local_mask = np.zeros_like(mask, dtype='int32')
                uids = np.unique(mask[mask != 0])
                for i, uid in enumerate(uids):
                    local_mask[mask == uid] = i + 1

                # --- SAVE ---
                uid = f"test_{idx}"
                
                # Machine Data
                cv2.imwrite(f"{OUTPUT_DIR}/images/{uid}.tif", combined_img)
                cv2.imwrite(f"{OUTPUT_DIR}/masks/{uid}.png", local_mask.astype(np.uint16))
                
                # Human Debug
                save_debug_visualization(uid, rgb_bgr, h_img, local_mask)

    except Exception as e:
        print(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()