import os

# Data Paths
RAW_DATA_DIR = "./data/raw"
PROCESSED_DATA_DIR = "./data/processed"
BASEMAP_TILES_DIR = "./data/tiles"
PATCHES_DIR = "./data/patches"

# Tiling Parameters
PATCH_SIZE = 512
STRIDE = 410  # ~20% overlap to prevent edge-boundary data loss

# Radiometric Normalization Percentiles
PERCENTILE_LOW = 2
PERCENTILE_HIGH = 98

# Ensure directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, BASEMAP_TILES_DIR, PATCHES_DIR]:
    os.makedirs(path, exist_ok=True)