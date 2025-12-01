import sys
import os

# Add backend to path
sys.path.append(os.getcwd())

from scripts.pipeline.data_generator import DataGenerator

# --- CONFIGURATION ---
CONFIG = {
    # Paths
    "OUTPUT_VECTOR_PATH": "data/raw/referentiel_batiments.gpkg", # Use raw vectors for test if we want, or processed
    # Actually, for the test to work on a specific tile that might NOT be in the selected set,
    # we should probably use the raw vectors or ensure the test tile is covered.
    # The previous test script used raw vectors. Let's stick to that for flexibility.
    
    "CACHE_DIR": "data/raw/cache",
    "OUTPUT_DIR": "data/processed/mask2former_data",
    
    # Parameters
    "CHIP_SIZE": 512,
    "MASK_EROSION": 0.5,
    "RANDOM_SEED": 42
}

TEST_TILE_FILENAME = "lidar_2464.tif"

def run_test():
    print(f"🚀 Starting Local Test on {TEST_TILE_FILENAME}")
    
    # 1. Ensure Vector Chunk Exists for Test Tile
    # We need to extract the chunk from the raw vectors if it doesn't exist
    tile_id = TEST_TILE_FILENAME.replace("lidar_", "").replace(".tif", "")
    chunk_path = os.path.join(CONFIG["CACHE_DIR"], "vectors", f"vectors_{tile_id}.gpkg")
    
    if not os.path.exists(chunk_path):
        print(f"   ⚠️ Vector chunk not found for test tile. Generating from raw vectors...")
        print(f"      Loading {CONFIG['OUTPUT_VECTOR_PATH']} (this might take a moment)...")
        
        # We need the bounds of the test tile to filter efficiently
        # Let's assume we have the tile file to get bounds
        lidar_path = os.path.join(CONFIG["CACHE_DIR"], TEST_TILE_FILENAME)
        if not os.path.exists(lidar_path):
             print(f"❌ Test LiDAR file not found: {lidar_path}")
             return

        import rasterio
        from shapely.geometry import box
        import geopandas as gpd
        
        with rasterio.open(lidar_path) as src:
            tile_bounds = src.bounds
            tile_box = box(*tile_bounds)
            
            # Create a GDF for the box to project if needed
            # Assuming raw vectors are in EPSG:4326 or MTM? 
            # We should check raw vector CRS first.
            # But for speed, let's try to load with bbox if possible.
            # If raw vectors are GPKG, we can use bbox.
            
            # We need to know the CRS of the raw vectors to project our query box
            meta = gpd.read_file(CONFIG['OUTPUT_VECTOR_PATH'], rows=1)
            
            box_gdf = gpd.GeoDataFrame({'geometry': [tile_box]}, crs=src.crs)
            if box_gdf.crs != meta.crs:
                box_gdf = box_gdf.to_crs(meta.crs)
            
            query_bounds = box_gdf.geometry.iloc[0].bounds
            
            print(f"      Filtering with bbox: {query_bounds}")
            vectors = gpd.read_file(CONFIG['OUTPUT_VECTOR_PATH'], bbox=query_bounds)
            
            # Ensure output dir exists
            os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
            
            vectors.to_file(chunk_path, driver="GPKG")
            print(f"      ✅ Saved chunk to {chunk_path}")
    else:
        print(f"   ✅ Vector chunk exists: {chunk_path}")

    # Override vector path to use the generated chunk
    CONFIG["OUTPUT_VECTOR_PATH"] = chunk_path
    
    # Create a mock tile info
    test_tile_list = [{
        "filename": TEST_TILE_FILENAME,
        "id": tile_id,
        "url": "http://mock" # Not used since file should be in cache
    }]
    
    generator = DataGenerator(CONFIG)
    generator.run(tile_list=test_tile_list)

if __name__ == "__main__":
    run_test()
