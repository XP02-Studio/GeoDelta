import os
import sys
import subprocess

def main():
    # Ensure we are operating from the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    os.chdir(project_root)

    print("==================================================")
    print("      INITIALIZING COMPLETE GEODELTA SYSTEM       ")
    print("==================================================\n")

    print(">>> [1/3] Generating Synthetic Satellite Data...")
    try:
        subprocess.run([sys.executable, "make_dummy_data.py"], check=True)
        print("Data generation complete.\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to generate dummy data: {e}")
        sys.exit(1)

    print(">>> [2/3] Executing Geospatial Processing Pipeline...")
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
        print("Pipeline execution complete.\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        sys.exit(1)

    print(">>> [3/3] Booting up Frontend UI Server...")
    try:
        # This will block and keep the server running until the user presses Ctrl+C
        subprocess.run([sys.executable, "Frontend_ui/run.py"], check=True)
    except KeyboardInterrupt:
        print("\nServer shutdown requested by user. Terminating processes.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Frontend server crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
