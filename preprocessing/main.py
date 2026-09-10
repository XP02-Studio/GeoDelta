import os
import config
from pipeline.ingestion import DataIngestion
from pipeline.alignment import AutoAlignment
from pipeline.basemap import BasemapGenerator
from pipeline.patch_tiler import PatchTiler

def run_pipeline(t1_filename: str, t2_filename: str):
    t1_path = os.path.join(config.RAW_DATA_DIR, t1_filename)
    t2_path = os.path.join(config.RAW_DATA_DIR, t2_filename)

    print("[Step 1] Ingesting, reprojecting, and normalizing GeoTIFFs...")
    t1_norm, t2_norm, geotransform, proj, warped_t2_path = DataIngestion.reproject_and_normalize(t1_path, t2_path)

    print("[Step 2] Executing auto-alignment feature matching (SIFT)...")
    t2_aligned = AutoAlignment.align_images(t1_norm, t2_norm)

    print("[Step 3] Generating offline XYZ tile directory for Leaflet UI...")
    BasemapGenerator.generate_xyz_tiles(t1_path, style_name="satellite")

    print("[Step 4] Creating ML patch tensors (.npz) and GPS bounding box metadata...")
    PatchTiler.create_ml_patches(t1_norm, t2_aligned, geotransform)

    print("Pipeline execution completed successfully!")

if __name__ == "__main__":
    # Place sample GeoTIFF files inside data/raw/ before running
    run_pipeline("t1_baseline.tif", "t2_current.tif")