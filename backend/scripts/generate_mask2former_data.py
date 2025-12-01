import sys
import os

# Add backend to path so we can import from scripts
sys.path.append(os.getcwd())

from scripts.pipeline.vector_filter import VectorFilter
from scripts.pipeline.lidar_downloader import LidarDownloader
from scripts.pipeline.data_generator import DataGenerator

# --- CONFIGURATION ---
CONFIG = {
    # Paths
    "INPUT_VECTOR_PATH": "data/raw/referentiel_batiments.gpkg",
    "LIDAR_INDEX_PATH": "data/raw/lidar_index.geojson",
    "OUTPUT_VECTOR_PATH": "data/processed/selected_vectors.gpkg",
    "OUTPUT_TILES_PATH": "data/processed/selected_tiles.json",
    "CACHE_DIR": "data/raw/cache",
    "OUTPUT_DIR": "data/processed/mask2former_data",
    
    # Parameters
    "MAX_TILES": 15,
    "CHIP_SIZE": 512,
    "MASK_EROSION": 0.5,
    "RANDOM_SEED": 42
}

def main():
    print("🚀 Starting Mask2Former Data Generation Pipeline")
    print("===============================================")
    
    # 1. Filter Vectors & Select Tiles
    print("\n[Step 1/3] Filtering Vectors & Selecting Tiles")
    vector_filter = VectorFilter(CONFIG)
    vector_filter.run()
    
    # 2. Download LiDAR
    print("\n[Step 2/3] Downloading LiDAR Tiles")
    downloader = LidarDownloader(CONFIG)
    downloader.run()
    
    # 3. Generate Data
    print("\n[Step 3/3] Generating Training Data")
    generator = DataGenerator(CONFIG)
    generator.run()
    
    print("\n✅ Pipeline Complete!")

if __name__ == "__main__":
    main()