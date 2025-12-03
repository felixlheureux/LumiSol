import rasterio
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from pyproj import Transformer, CRS
import os
import numpy as np

# --- CONFIGURATION ---
FILES = {
    "raster_mns": "data/montreal/homa/mns/mns.tif",
    "raster_mnc": "data/montreal/homa/mnc/mnc.tif",
    "footprints": "data/montreal/homa/building_footprints/CARTO_BAT_TOIT.gpkg",
    "cadastre": "data/montreal/lot_cadastre.gpkg"
}

TARGET_LAT = 45.551125
TARGET_LON = -73.544594
BUFFER_METERS = 50 

# Montreal Standard CRS (MTM Zone 8) - Everything will be plotted in this system
PLOT_CRS = "EPSG:32188"

def get_target_window_in_meters():
    """
    Returns the target bounding box (minx, miny, maxx, maxy) in the PLOT_CRS (meters).
    """
    # Transform the Lat/Lon point to the Metric Plot CRS
    transformer = Transformer.from_crs("EPSG:4326", PLOT_CRS, always_xy=True)
    cx, cy = transformer.transform(TARGET_LON, TARGET_LAT)
    
    return (cx - BUFFER_METERS, cy - BUFFER_METERS, cx + BUFFER_METERS, cy + BUFFER_METERS)

def plot_raster(ax, file_path, target_bounds):
    """
    Reads the raster, reprojects the relevant chunk to PLOT_CRS, and plots.
    """
    try:
        with rasterio.open(file_path) as src:
            # 1. We need to figure out which pixels in the SOURCE file overlap our TARGET box
            # Transform our metric box (target_bounds) back to the SOURCE CRS to find the read window
            transformer_to_src = Transformer.from_crs(PLOT_CRS, src.crs, always_xy=True)
            minx, miny, maxx, maxy = target_bounds
            
            # Project the corners to find the source window
            xs, ys = transformer_to_src.transform([minx, maxx], [miny, maxy])
            src_bbox = (min(xs), min(ys), max(xs), max(ys))
            
            # 2. Read the data within that source window
            window = from_bounds(*src_bbox, transform=src.transform)
            data = src.read(1, window=window)
            
            # 3. Handle 'nodata' or empty reads
            if data.size == 0 or np.all(data == src.nodata):
                ax.text(0.5, 0.5, "No Data", ha='center')
                return

            # 4. Now we have the data, but it's still in the source CRS (e.g., Lat/Lon or bad projection)
            # We must reproject this specific chunk of data to our PLOT_CRS for display
            
            # Calculate transform for the reprojection
            src_window_transform = src.window_transform(window)
            dst_transform, width, height = calculate_default_transform(
                src.crs, PLOT_CRS, window.width, window.height, *src_bbox
            )
            
            # Create destination array
            dst_array = np.zeros((height, width), dtype=data.dtype)

            reproject(
                source=data,
                destination=dst_array,
                src_transform=src_window_transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=PLOT_CRS,
                resampling=Resampling.nearest
            )

            # 5. Plot using the PROJECTED transform
            show(dst_array, ax=ax, transform=dst_transform, cmap='viridis')
            ax.set_title(os.path.basename(file_path))
            
            # Force the axis limits to match our target metric box exactly
            ax.set_xlim(target_bounds[0], target_bounds[2])
            ax.set_ylim(target_bounds[1], target_bounds[3])

    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center')

def plot_vector(ax, file_path, target_bounds):
    """
    Reads vector, converts to PLOT_CRS, and crops to view.
    """
    try:
        # 1. Get Source CRS efficiently
        meta = gpd.read_file(file_path, rows=1)
        
        # 2. Convert our Metric Target Box -> Source CRS (to filter reading)
        transformer_to_src = Transformer.from_crs(PLOT_CRS, meta.crs, always_xy=True)
        minx, miny, maxx, maxy = target_bounds
        xs, ys = transformer_to_src.transform([minx, maxx], [miny, maxy])
        
        # Note: If converting Meters -> Degrees, we need to be careful with min/max order
        src_bbox_geom = box(min(xs), min(ys), max(xs), max(ys))

        # 3. Read ONLY features in that box
        gdf = gpd.read_file(file_path, bbox=src_bbox_geom)

        if not gdf.empty:
            # 4. CRITICAL STEP: Convert data to the Metric PLOT_CRS
            gdf = gdf.to_crs(PLOT_CRS)
            
            gdf.plot(ax=ax, color='orange', edgecolor='black', alpha=0.6)
        else:
            ax.text(0.5, 0.5, "No features", ha='center')

        ax.set_title(os.path.basename(file_path))
        # Force limits to match the other plots
        ax.set_xlim(target_bounds[0], target_bounds[2])
        ax.set_ylim(target_bounds[1], target_bounds[3])

    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center')

# --- MAIN ---
def generate_composite():
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axs = axes.flatten()

    # Get the "Master" bounding box in Meters (EPSG:32188)
    # This ensures all 4 plots share the EXACT same coordinate space
    target_bounds = get_target_window_in_meters()
    print(f"Target Window (MTM8): {target_bounds}")

    plot_raster(axs[0], FILES["raster_mns"], target_bounds)
    plot_raster(axs[1], FILES["raster_mnc"], target_bounds)
    plot_vector(axs[2], FILES["footprints"], target_bounds)
    plot_vector(axs[3], FILES["cadastre"], target_bounds)

    plt.tight_layout()
    plt.savefig("preview_tiles.png", dpi=150)
    print("Saved preview_tiles.png")

if __name__ == "__main__":
    generate_composite()