import sys
from pathlib import Path

# Add geospatial_pipeline folder to Python path
pipeline_dir = Path(__file__).resolve().parent / "geospatial_pipeline"
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

from main import run_pipeline

if __name__ == "__main__":
    print("==================================================")
    print("   SIH262270: Air-Gapped Satellite Pipeline Run   ")
    print("==================================================\n")
    
    run_pipeline("t1_baseline.tif", "t2_current.tif")