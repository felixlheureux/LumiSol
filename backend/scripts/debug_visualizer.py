import sys
import os
from pathlib import Path

# Add backend directory to sys.path to allow importing 'app'
sys.path.append(str(Path(__file__).resolve().parent.parent))

import rasterio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from pyproj import Transformer

from app.services.solar_engine import SolarEngine
from app.services.lot_manager import LotManager
from app.core.config import LIDAR_FILE, CADASTRE_FILE

# --- 1. REAL DATA LOADER ---
def load_real_lidar_sample(filepath="sample.tif"):
    """
    Loads a real GeoTIFF (MNS) to test the algorithm on actual Montreal data.
    """
    if not os.path.exists(filepath):
        print(f"⚠️ File '{filepath}' not found.")
        return None, None

    print(f"📂 Loading Real LiDAR from '{filepath}'...")
    try:
        with rasterio.open(filepath) as src:
            mns = src.read(1)
            print(f"   -> Transform: {src.transform}")
            print(f"   -> Resolution: {src.res}")
            
            # Handle NoData / Nan
            mns = np.nan_to_num(mns, nan=np.nanmin(mns))
            
            # For testing without a matching MNT file, we estimate ground 
            # as the lowest point in the tile (Good enough for single-building crops)
            mnt = np.full_like(mns, np.min(mns))
            
            return mns, mnt
    except Exception as e:
        print(f"❌ Error loading TIF: {e}")
        return None, None

# --- 3. THE TEST RUNNER ---
def run_visual_test():
    print("🔬 Starting Visual Debugger...")
    
    # Initialize Engine
    engine = SolarEngine()
    
    # Try loading real data first
    mns, mnt = load_real_lidar_sample(str(LIDAR_FILE))
    
    if mns is None:
        print("❌ Could not load LiDAR data. Exiting.")
        return
    
    # User provided coordinates
    click_x, click_y = 3006, 6706 # Placeholder, will be overwritten by lat/lon
    
    print(f"   -> MNS Shape: {mns.shape}")
    
    # --- USER PROVIDED COORDINATES ---
    target_lat = 45.551126
    target_lon = -73.544592
    print(f"   -> Target Location: {target_lat}, {target_lon}")

    # --- REAL LOT DATA INTEGRATION ---
    print("   -> Fetching Real Lot Polygon from GeoJSON...")
    try:
        lot_manager = LotManager(CADASTRE_FILE)
        
        # 1. Get Lot Polygon (Projected to LiDAR CRS)
        real_lot_polygon_proj = lot_manager.get_lot_at_point(target_lat, target_lon, target_crs="EPSG:2950")
        
        if real_lot_polygon_proj:
            print("      ✅ Found Lot Polygon!")
            
            with rasterio.open(str(LIDAR_FILE)) as src:
                # 2. Convert Polygon (LiDAR CRS) -> Pixels
                lot_polygon_pixels = []
                print(f"      [DEBUG] First Point (Proj): {real_lot_polygon_proj[0]}")
                for px, py in real_lot_polygon_proj:
                    col, row = ~src.transform * (px, py)
                    if len(lot_polygon_pixels) == 0:
                        print(f"      [DEBUG] First Point (Pixel): col={col}, row={row}")
                    lot_polygon_pixels.append((col, row))
                
                lot_polygon = lot_polygon_pixels
                
                # 3. Get Center Pixel of the Lot (for cropping)
                # We can just take the centroid of the polygon pixels
                poly_cols = [p[0] for p in lot_polygon]
                poly_rows = [p[1] for p in lot_polygon]
                click_x = int(sum(poly_cols) / len(poly_cols))
                click_y = int(sum(poly_rows) / len(poly_rows))
                print(f"      Center Pixel: {click_x}, {click_y}")
                
                # BOUNDS CHECK
                if not (0 <= click_x < src.width and 0 <= click_y < src.height):
                    print(f"❌ Coordinates {click_x},{click_y} are out of bounds (Image: {src.width}x{src.height}).")
                    return

                print(f"      [DEBUG] Click Projected: {click_x}, {click_y}")
                print(f"      [DEBUG] Lot Polygon Start: {lot_polygon[0]}")
                print(f"      [DEBUG] Click Projected: {click_x}, {click_y}")
                print(f"      [DEBUG] Lot Polygon Start: {lot_polygon[0]}")

        else:
            print("      ⚠️ No Lot found at this location.")
            return

    except Exception as e:
        print(f"      ❌ Error loading lot data: {e}")
        return

    # Run Segmentation (Lot Structures)
    print("   -> Running 'Lot Structure Segmentation' (Blob Detection)...")
    
    # We need to manually crop and prepare data for segment_lot_structures
    # essentially replicating what segment_roofs does but for the whole lot
    
    # 1. Define Crop Window (Large enough for the lot)
    crop_size = 300 # Bigger crop for whole lot
    rows, cols = mns.shape
    half_size = crop_size // 2
    x_min = max(0, click_x - half_size)
    x_max = min(cols, click_x + half_size)
    y_min = max(0, click_y - half_size)
    y_max = min(rows, click_y + half_size)
    
    mns_crop = mns[y_min:y_max, x_min:x_max]
    mnt_crop = mnt[y_min:y_max, x_min:x_max]
    
    # CHECK FOR VALID MNT
    # If MNT is 0 or nDSM is huge everywhere, we need to generate a synthetic MNT.
    # We use a morphological opening (minimum filter) to estimate ground.
    from scipy.ndimage import minimum_filter
    
    # Heuristic: If 90% of pixels are > 10m (assuming flat ground is ~0), MNT is likely missing.
    # Or simpler: If MNT is all zeros.
    if np.max(mnt_crop) == 0 or np.mean(mns_crop - mnt_crop) > 1000: # 1000m is impossible for nDSM
        print("      ⚠️ MNT seems invalid or missing. Generating Synthetic MNT (Ground Estimate)...")
        # Window size should be larger than the largest building (e.g. 20m)
        # Assuming 1px = 1m (approx)
        # 2. Define Crop (300x300)
    # We use a fixed size window around the center
    crop_size = 300
    half_size = crop_size // 2
    
    # Calculate window bounds
    window = rasterio.windows.Window(click_x - half_size, click_y - half_size, crop_size, crop_size)
    
    # Read Data (Re-open file)
    # Read Data (Re-open file)
    with rasterio.open(str(LIDAR_FILE)) as src:
        mns_crop = src.read(1, window=window)
    
    # Handle NoData
    mns_crop = np.nan_to_num(mns_crop, nan=np.nanmin(mns_crop))
    
    # Generate Synthetic MNT (since we know it's missing/bad)
    # We generate it at 1m resolution first (faster/smoother)
    from scipy.ndimage import minimum_filter, gaussian_filter, zoom
    mnt_crop = minimum_filter(mns_crop, size=20)
    mnt_crop = gaussian_filter(mnt_crop, sigma=2)
    
    # --- UPSCALING (Super-Resolution) ---
    SUPER_RES_FACTOR = 2 # 2x upscaling (1m -> 0.5m)
    print(f"   -> Upscaling Data (Factor: {SUPER_RES_FACTOR}x)...")
    
    # Order=1 (Bilinear) or 3 (Cubic). Cubic is better for smooth surfaces.
    mns_crop = zoom(mns_crop, SUPER_RES_FACTOR, order=3)
    mnt_crop = zoom(mnt_crop, SUPER_RES_FACTOR, order=3)
    
    print(f"      New Shape: {mns_crop.shape}")

    # Adjust Polygon to Crop Coordinates AND Scale
    adjusted_polygon = []
    # Crop origin in global pixels
    crop_origin_x = click_x - half_size
    crop_origin_y = click_y - half_size
    
    for px, py in lot_polygon:
        # 1. Shift to Crop
        cx = px - crop_origin_x
        cy = py - crop_origin_y
        
        # 2. Scale
        cx *= SUPER_RES_FACTOR
        cy *= SUPER_RES_FACTOR
        
        adjusted_polygon.append((cx, cy))
        
    lot_mask = engine.create_lot_mask(mns_crop.shape, adjusted_polygon)
    
    # 3. Run Segmentation (Strict Legal Boundary)
    # We use the official lot polygon without any shifting.
    print("   -> Running Segmentation (Strict Legal Boundary)...")
    
    # Use original lot_mask
    structures_mask, ndsm = engine.segment_lot_structures(mns_crop, mnt_crop, lot_mask)

    if np.sum(structures_mask) == 0:
        print("❌ No structures found on lot.")
        return

    # --- ALIGNMENT CHECK (Informational Only) ---
    from scipy.ndimage import center_of_mass
    # Calculate centroid of the detected structure (in crop pixels)
    structure_cy, structure_cx = center_of_mass(structures_mask)
    
    # Calculate centroid of the lot polygon (in crop pixels)
    poly_cols = [p[0] for p in adjusted_polygon]
    poly_rows = [p[1] for p in adjusted_polygon]
    lot_cx = sum(poly_cols) / len(poly_cols)
    lot_cy = sum(poly_rows) / len(poly_rows)
    
    # Calculate Offset
    offset_x = structure_cx - lot_cx
    offset_y = structure_cy - lot_cy
    offset_dist = np.sqrt(offset_x**2 + offset_y**2)
    
    print(f"      [DEBUG] Structure Centroid: {structure_cx:.2f}, {structure_cy:.2f}")
    print(f"      [DEBUG] Lot Centroid: {lot_cx:.2f}, {lot_cy:.2f}")
    print(f"      [DEBUG] Alignment Offset: {offset_dist:.2f} pixels")
    
    # Run Solar Physics
    print("   -> Calculating Solar Potential...")
    # We can just visualize the nDSM and Structure Mask for now
    
    # --- PLOTTING ---
    print("   -> Rendering Plots...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 8))
    
    # Zoom Window (Center 100m x 100m)
    # Image is 600x600 (0.5m res) -> 300m x 300m
    # Center is 300, 300
    # 100m window = 200 pixels
    zoom_half = 100 # pixels (50m)
    center_x = mns_crop.shape[1] // 2
    center_y = mns_crop.shape[0] // 2
    x_min, x_max = center_x - zoom_half, center_x + zoom_half
    y_min, y_max = center_y - zoom_half, center_y + zoom_half
    
    # 1. MNS (Surface)
    axes[0].imshow(mns_crop, cmap='terrain')
    axes[0].set_title("LiDAR Surface (MNS)")
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_max, y_min) # Invert Y for image coords
    
    # 2. Lot Mask
    axes[1].imshow(lot_mask, cmap='gray')
    axes[1].set_title("Legal Lot Mask")
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_max, y_min)
    
    # 3. nDSM (Height)
    axes[2].imshow(ndsm, cmap='jet', vmin=0, vmax=10)
    axes[2].set_title("Normalized Height (nDSM)")
    axes[2].set_xlim(x_min, x_max)
    axes[2].set_ylim(y_max, y_min)
    
    # 4. Final Structure Segmentation
    axes[3].imshow(structures_mask, cmap='prism', interpolation='nearest')
    axes[3].set_title("Detected Structures")
    axes[3].set_xlim(x_min, x_max)
    axes[3].set_ylim(y_max, y_min)

    plt.tight_layout()
    plt.savefig("debug_output.png")
    print("✅ Test Complete. Saved to debug_output.png.")

if __name__ == "__main__":
    run_visual_test()
