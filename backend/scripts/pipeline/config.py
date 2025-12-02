import os

# Base Project Paths
# Assuming this file is in backend/scripts/pipeline/config.py
# BASE_DIR should be backend/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

CONFIG = {
    # --- Paths ---
    "CACHE_DIR": os.path.join(DATA_DIR, "raw", "cache"),
    "INPUT_VECTOR_PATH": os.path.join(DATA_DIR, "raw", "referentiel_batiments.gpkg"),
    "LIDAR_INDEX_PATH": os.path.join(DATA_DIR, "raw", "lidar_index.geojson"),
    "OUTPUT_TILES_PATH": os.path.join(DATA_DIR, "processed", "selected_tiles.json"),
    "OUTPUT_VECTOR_PATH": os.path.join(DATA_DIR, "processed", "vectors", "combined_vectors.gpkg"),
    
    # Training Data Output
    "OUTPUT_DIR": os.path.join(DATA_DIR, "processed", "mask2former_data"),
    "MODEL_OUTPUT_DIR": os.path.join(BASE_DIR, "models", "lumisol_v1"),
    
    # --- Data Generation Params ---
    "CHIP_SIZE": 512,
    "MAX_TILES": 15,
    "MASK_EROSION": 0.5,
    "RANDOM_SEED": 42,
    
    # --- Training Hyperparameters ---
    "MODEL_CHECKPOINT": "facebook/mask2former-swin-tiny-coco-instance",
    "TRAIN_BATCH_SIZE": 2,
    "TRAIN_EPOCHS": 10,
    "LEARNING_RATE": 1e-4,
}
