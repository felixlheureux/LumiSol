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

import argparse

def main():
    parser = argparse.ArgumentParser(description="Mask2Former Data Generation Pipeline")
    parser.add_argument("--skip-vectors", action="store_true", help="Skip Step 1: Vector Filtering & Tile Selection")
    parser.add_argument("--skip-download", action="store_true", help="Skip Step 2: LiDAR Downloading")
    parser.add_argument("--skip-generation", action="store_true", help="Skip Step 3: Data Generation")
    args = parser.parse_args()

    print("🚀 Starting Mask2Former Data Generation Pipeline")
    print("===============================================")
    
    # 1. Filter Vectors & Select Tiles
    if not args.skip_vectors:
        print("\n[Step 1/3] Filtering Vectors & Selecting Tiles")
        vector_filter = VectorFilter(CONFIG)
        vector_filter.run()
    else:
        print("\n[Step 1/3] Skipping Vector Filtering...")
    
    # 2. Download LiDAR
    if not args.skip_download:
        print("\n[Step 2/3] Downloading LiDAR Tiles")
        downloader = LidarDownloader(CONFIG)
        downloader.run()
    else:
        print("\n[Step 2/3] Skipping LiDAR Download...")
    
    # 3. Generate Data
    if not args.skip_generation:
        print("\n[Step 3/3] Generating Training Data")
        generator = DataGenerator(CONFIG)
        generator.run()
    else:
        print("\n[Step 3/3] Skipping Data Generation...")
    
    print("\n✅ Pipeline Complete!")

if __name__ == "__main__":
    main()