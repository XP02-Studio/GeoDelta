import os
import sys
import subprocess
from pathlib import Path

class BasemapGenerator:
    @staticmethod
    def generate_xyz_tiles(input_path: str, output_dir: str = "data/tiles/satellite", style_name: str = "satellite"):
        """
        Generates offline XYZ tile structure (/{z}/{x}/{y}.png) for Leaflet/OpenLayers maps.
        Resolves gdal2tiles script path within the active virtual environment on Windows.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Locate gdal2tiles.py script in the active virtual environment
        env_dir = Path(sys.executable).parent
        gdal2tiles_script = env_dir / "gdal2tiles.py"
        
        if gdal2tiles_script.exists():
            cmd = [
                sys.executable,
                str(gdal2tiles_script),
                "-z", "12-16",
                "--processes=4",
                input_path,
                output_dir
            ]
        else:
            # Fallback to system PATH executable call
            cmd = [
                "gdal2tiles",
                "-z", "12-16",
                "--processes=4",
                input_path,
                output_dir
            ]
        
        try:
            print(f"Generating web tiles from '{input_path}' into '{output_dir}'...")
            subprocess.run(cmd, check=True)
            print(f"XYZ map tiles generated successfully at '{output_dir}'")
        except subprocess.CalledProcessError as e:
            print(f"Error generating map tiles via gdal2tiles: {e}")
        except Exception as e:
            print(f"Unexpected error during tile generation: {e}")