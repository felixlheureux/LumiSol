import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import os

# CONFIG
LIDAR_PATH = "MNS(terrain+batiment)_2015_1m_Mercier–Hochelaga-Maisonneuve.tif"       
CADASTRE_PATH = "quebec_lots.gpkg" # Changed to GPKG

def check_alignment():
    # 1. Check files
    if not os.path.exists(LIDAR_PATH):
        print(f"❌ Error: LiDAR file '{LIDAR_PATH}' not found.")
        return
    if not os.path.exists(CADASTRE_PATH):
        print(f"❌ Error: Cadastre database '{CADASTRE_PATH}' not found.")
        return

    print(f"🔍 Checking alignment between {LIDAR_PATH} and {CADASTRE_PATH}...")

    # 2. Load the LiDAR (Raster)
    with rasterio.open(LIDAR_PATH) as src:
        print(f"   -> LiDAR CRS: {src.crs}")
        print(f"   -> LiDAR Bounds: {src.bounds}")
        
        # Read data for plotting
        # Since this file might be huge, we should read a window if possible, 
        # but for check_alignment we usually plot the whole thing or a large chunk.
        # If it's too big (e.g. > 1GB), reading the whole thing might crash.
        # Let's assume it fits in memory (it's likely ~100MB for a district).
        try:
            lidar_data = src.read(1)
        except Exception as e:
             print(f"⚠️ Warning: Could not read entire raster ({e}). Reading a window around target...")
             # Fallback: Read window around target
             # We need to transform target to pixels first
             from pyproj import Transformer
             transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
             zoom_lon, zoom_lat = -73.544592, 45.551126
             zoom_x, zoom_y = transformer.transform(zoom_lon, zoom_lat)
             py, px = src.index(zoom_x, zoom_y)
             window = rasterio.windows.Window(px - 500, py - 500, 1000, 1000)
             lidar_data = src.read(1, window=window)
             # Note: Plotting this window requires adjusting transform, which is complex for this script.
             # Let's hope it fits in memory.
             
        # Mask out 'nodata' values so the plot looks nice
        if src.nodata is not None:
            import numpy as np
            lidar_data = np.ma.masked_equal(lidar_data, src.nodata)

        # 3. Fetch Relevant Lots (Vector)
        # We need to find lots that overlap with this specific LiDAR tile.
        
        # First, define the bounding box of the LiDAR in its OWN projection
        lidar_box = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        
        # We need to know what CRS the GeoPackage is in to query it correctly.
        try:
            temp_gdf = gpd.read_file(CADASTRE_PATH, rows=1)
            cadastre_crs = temp_gdf.crs
        except Exception as e:
            print(f"   -> Error reading CRS from GPKG: {e}")
            return
            
        print(f"   -> Cadastre CRS: {cadastre_crs}")

        # Create a GeoDataFrame for the bounding box and reproject it to match the Cadastre
        bbox_gdf = gpd.GeoDataFrame({'geometry': [lidar_box]}, crs=src.crs)
        bbox_reprojected = bbox_gdf.to_crs(cadastre_crs)
        search_bounds = bbox_reprojected.total_bounds # [minx, miny, maxx, maxy]

        print(f"   -> Querying database for lots in this area: {search_bounds}")
        
        # Load ONLY the lots inside the bounding box (Spatial Filter)
        try:
            # Specify layer for GPKG (lots)
            try:
                lots = gpd.read_file(CADASTRE_PATH, bbox=tuple(search_bounds), layer='lots')
            except ValueError:
                lots = gpd.read_file(CADASTRE_PATH, bbox=tuple(search_bounds))
        except Exception as e:
             print(f"   -> Error querying GPKG: {e}")
             return
        
        if len(lots) == 0:
            print("⚠️ No lots found in this area! Check if your files cover the same location.")
            return

        print(f"   -> Found {len(lots)} lots.")

        # 4. Reproject Lots to match LiDAR
        # This is the CRITICAL step. We must align Vector to Raster.
        lots_aligned = lots.to_crs(src.crs)

        # 5. Plot Result
        print("   -> Rendering plot...")
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # A. Plot LiDAR (Background)
        # 'extent' ensures the image is plotted in map coordinates, not pixels
        show(src, ax=ax, cmap='gray', title="LiDAR (Gray) vs Cadastre (Red)")
        
        # B. Plot Cadastre (Foreground)
        # We expect Polygons now.
        print("   -> Plotting Real Polygons...")
        
        # Check if Point or Polygon just in case
        geom_type = lots_aligned.geometry.type.iloc[0]
        if 'Point' in geom_type:
            print("   ⚠️ Warning: Still finding Points. Buffering...")
            lots_aligned['geometry'] = lots_aligned.geometry.buffer(30, cap_style=3)
        
        lots_aligned.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2, linestyle='--', label='Real Lot Boundary')
        
        # ZOOM IN
        # Target: 45.551126, -73.544592 (User Provided)
        # We need to convert this to EPSG:2950 (LiDAR CRS)
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        zoom_lon, zoom_lat = -73.544592, 45.551126
        zoom_x, zoom_y = transformer.transform(zoom_lon, zoom_lat)
        
        print(f"   -> Zoom Target (Lat/Lon): {zoom_lat}, {zoom_lon}")
        print(f"   -> Zoom Target (Projected): {zoom_x}, {zoom_y}")
        
        # Restore Zoom
        zoom_radius = 100 # meters
        ax.set_xlim(zoom_x - zoom_radius, zoom_x + zoom_radius)
        ax.set_ylim(zoom_y - zoom_radius, zoom_y + zoom_radius)
        
        plt.savefig("alignment_check_zoomed.png")
        print("✅ Saved to alignment_check_zoomed.png. Check if the red markers/lines align with the houses.")

if __name__ == "__main__":
    check_alignment()
