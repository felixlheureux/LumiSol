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
        real_lot_polygon_geom = lot_manager.get_lot_at_point(target_lat, target_lon, target_crs="EPSG:2950")
        
        if real_lot_polygon_geom:
            print("      ✅ Found Lot Polygon!")
            
            with rasterio.open(str(LIDAR_FILE)) as src:
                # 2. Get Centroid for Cropping
                centroid = real_lot_polygon_geom.centroid
                col, row = ~src.transform * (centroid.x, centroid.y)
                click_x, click_y = int(col), int(row)
                
                print(f"      Center Pixel: {click_x}, {click_y}")
                
                # BOUNDS CHECK
                if not (0 <= click_x < src.width and 0 <= click_y < src.height):
                    print(f"❌ Coordinates {click_x},{click_y} are out of bounds (Image: {src.width}x{src.height}).")
                    return

        else:
            print("      ⚠️ No Lot found at this location.")
            return

    except Exception as e:
        print(f"      ❌ Error loading lot data: {e}")
        return

    # Run Segmentation (Lot Structures)
    print("   -> Running 'Scientific Facet Segmentation'...")
    
    # 1. Define Crop Window (Large enough for the lot)
    crop_size = 300 # Bigger crop for whole lot
    rows, cols = mns.shape
    half_size = crop_size // 2
    x_min = max(0, click_x - half_size)
    x_max = min(cols, click_x + half_size)
    y_min = max(0, click_y - half_size)
    y_max = min(rows, click_y + half_size)
    
    # Read Crop
    with rasterio.open(str(LIDAR_FILE)) as src:
        window = rasterio.windows.Window(x_min, y_min, x_max - x_min, y_max - y_min)
        mns_crop = src.read(1, window=window)
        transform = src.window_transform(window)
    
    # Handle NoData
    mns_crop = np.nan_to_num(mns_crop, nan=np.nanmin(mns_crop))
    
    # Generate Synthetic MNT
    from scipy.ndimage import minimum_filter, gaussian_filter
    mnt_crop = minimum_filter(mns_crop, size=20)
    mnt_crop = gaussian_filter(mnt_crop, sigma=2)
    
    # --- UPSCALING & MASK CREATION ---
    SUPER_RES_FACTOR = 2
    print(f"   -> Upscaling Data (Factor: {SUPER_RES_FACTOR}x)...")
    
    # We need to create the lot mask matching the upscaled dimensions
    new_transform = transform * transform.scale(0.5, 0.5)
    upscaled_shape = (mns_crop.shape[0] * SUPER_RES_FACTOR, mns_crop.shape[1] * SUPER_RES_FACTOR)
    
    lot_mask = engine.create_lot_mask(real_lot_polygon_geom, upscaled_shape, new_transform)
    
    # --- RUN SEGMENTATION ---
    # 4. Run Segmentation
    print("   -> Running 'Scientific Facet Segmentation'...")
    # Pass transform (even if dummy or real)
    structures_mask, nx, ny, nz, ndsm, hillshade = engine.segment_solar_facets(mns_crop, mnt_crop, lot_mask, transform)

    if np.sum(structures_mask) == 0:
        print("❌ No structures found on lot.")
        # Continue to show what we have
    
    # --- 5. VISUALIZE ---
    print("   -> Rendering Plots...")
    
    # Create a 2x3 grid (6 plots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. MNS (Raw Surface)
    ax = axes[0, 0]
    im = ax.imshow(mns_crop, cmap='terrain')
    ax.set_title("1. Raw LiDAR Surface (MNS)")
    plt.colorbar(im, ax=ax)
    
    # 2. nDSM (Height Above Ground) - Masked to Lot
    ax = axes[0, 1]
    # Show nDSM but only inside the lot for clarity, or full?
    # Let's show full nDSM but highlight lot
    im = ax.imshow(ndsm, cmap='jet', vmin=0, vmax=10)
    ax.set_title("2. Normalized Height (nDSM)")
    plt.colorbar(im, ax=ax)

    # 3. Lot Mask
    ax = axes[0, 2]
    ax.imshow(lot_mask, cmap='gray')
    ax.set_title("3. Legal Lot Boundaries")
    
    # 4. Hillshade (AI Vision)
    ax = axes[1, 0]
    ax.imshow(hillshade, cmap='gray')
    ax.set_title("4. AI Vision (Hillshade)")
    
    # 5. Solar Irradiance (Potential)
    # We need to calculate it briefly for display
    irradiance = engine.calculate_irradiance(nx, ny, nz, structures_mask)
    ax = axes[1, 1]
    im = ax.imshow(irradiance, cmap='hot')
    ax.set_title("5. Solar Irradiance (kWh/m²)")
    plt.colorbar(im, ax=ax)
    
    # 6. Final Segmentation
    ax = axes[1, 2]
    # Overlay mask on hillshade for context
    ax.imshow(hillshade, cmap='gray')
    # Create a red overlay for the mask
    mask_overlay = np.zeros((*structures_mask.shape, 4))
    mask_overlay[structures_mask] = [1, 0, 0, 0.5] # Red with 50% alpha
    ax.imshow(mask_overlay)
    ax.set_title(f"6. Final Roof Segments ({np.sum(structures_mask)} px)")

    plt.tight_layout()
    plt.savefig("debug_output.png")
    print("✅ Test Complete. Saved to debug_output.png.")

if __name__ == "__main__":
    run_visual_test()
