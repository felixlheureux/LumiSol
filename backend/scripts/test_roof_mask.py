import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import matplotlib.pyplot as plt
from pyproj import Transformer

# --- CONFIGURATION ---
FILES = {
    "raster": "data/montreal/homa/mns/mns.tif",
    "cadastre": "data/montreal/lot_cadastre.gpkg",
    "footprints": "data/montreal/homa/building_footprints/CARTO_BAT_TOIT.gpkg"
}

TARGET_LAT = 45.551125
TARGET_LON = -73.544594

def get_clipped_roof_mask():
    print(f"1. Searching for Cadastral Lot at {TARGET_LAT}, {TARGET_LON}...")
    
    # --- STEP 1: FIND THE LOT (The "Cookie Cutter") ---
    # We use a small buffer to avoid loading the whole city
    p_geo = Point(TARGET_LON, TARGET_LAT)
    cadastre = gpd.read_file(FILES["cadastre"], bbox=p_geo.buffer(0.001))
    
    # Find the exact lot containing the point
    target_lot = cadastre[cadastre.contains(p_geo)]
    
    if target_lot.empty:
        print("No lot found underneath this point.")
        return

    lot_id = target_lot.index[0]
    print(f"   Found Lot ID: {lot_id}")
    
    # --- STEP 2: FIND THE BUILDING (The "Dough") ---
    print("2. Searching for connected buildings...")
    
    # Get the bounding box of the LOT to filter the building load
    # We must match the CRS of the footprint file to do the bbox read efficiently
    meta_fp = gpd.read_file(FILES["footprints"], rows=0)
    
    # Reproject Lot to Footprint CRS for the bbox filter
    target_lot_proj = target_lot.to_crs(meta_fp.crs)
    
    # --- FIX IS HERE ---
    # .total_bounds returns a numpy array. We must convert it to a tuple.
    lot_bounds = tuple(target_lot_proj.total_bounds)
    
    try:
        footprints = gpd.read_file(FILES["footprints"], bbox=lot_bounds)
    except Exception as e:
        print(f"Error reading footprints: {e}")
        return

    if footprints.empty:
        print("No buildings intersect this lot.")
        return

    # --- STEP 3: THE CLIP (Cutting the cookie) ---
    print("3. Clipping building to lot boundaries...")
    
    # We perform an intersection. This cuts the building shape 
    # exactly where the lot lines are.
    # Note: 'overlay' handles the cut perfectly.
    clipped_roof = gpd.overlay(
        footprints, 
        target_lot_proj, # Both must be in same CRS (Footprint CRS)
        how='intersection'
    )
    
    if clipped_roof.empty:
        print("Intersection resulted in empty geometry.")
        return
        
    print(f"   Result: {len(clipped_roof)} roof segment(s) isolated.")

    # --- STEP 4: MASK THE RASTER ---
    print("4. Masking Raster...")
    with rasterio.open(FILES["raster"]) as src:
        # Reproject our new clipped shape to the Raster's CRS
        final_shape = clipped_roof.to_crs(src.crs)
        
        # Mask
        out_image, out_transform = mask(src, final_shape.geometry, crop=True)
        
        # Update Meta
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

    # --- VISUALIZATION ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot 1: The Context (Lot vs Full Building)
    target_lot_proj.plot(ax=ax1, facecolor='none', edgecolor='blue', linewidth=2, label='Lot')
    footprints.plot(ax=ax1, facecolor='gray', alpha=0.5, label='Raw Building')
    clipped_roof.plot(ax=ax1, facecolor='red', alpha=0.5, label='Clipped Roof')
    ax1.set_title("Clipping Operation")
    ax1.legend()
    
    # Plot 2: The Result (Masked Raster)
    show(out_image, ax=ax2, transform=out_transform, cmap='viridis')
    final_shape.plot(ax=ax2, facecolor='none', edgecolor='white', linewidth=2)
    ax2.set_title("Final Masked Raster")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    get_clipped_roof_mask()