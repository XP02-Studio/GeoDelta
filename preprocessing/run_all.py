import os
import sys
import subprocess

def main():
    # Ensure we are operating from the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    os.chdir(project_root)

    # Ensure we use the virtual environment Python if it exists, otherwise create it
    import venv
    venv_dir = os.path.join(project_root, "geodelta_env")
    venv_python_win = os.path.join(venv_dir, "Scripts", "python.exe")
    venv_python_unix = os.path.join(venv_dir, "bin", "python")
    
    created_venv = not os.path.exists(venv_dir)
    if created_venv:
        print(f"Creating virtual environment in {venv_dir}...")
        venv.create(venv_dir, with_pip=True)
        print("Virtual environment created.")

    if os.path.exists(venv_python_win):
        python_exe = venv_python_win
    elif os.path.exists(venv_python_unix):
        python_exe = venv_python_unix
    else:
        python_exe = sys.executable

    req_file = os.path.join(current_dir, "requirements.txt")
    if os.path.exists(req_file):
        print(f"Installing requirements from {req_file}...")
        try:
            subprocess.run([python_exe, "-m", "pip", "install", "-r", req_file], check=True)
            print("Requirements installed successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Failed to install requirements: {e}")
            sys.exit(1)

    print(f"Using Python environment: {python_exe}\n")

    print(">>> [1/3] Generating Synthetic Satellite Data...")
    try:
        subprocess.run([python_exe, "make_dummy_data.py"], check=True)
        print("Data generation complete.\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to generate dummy data: {e}")
        sys.exit(1)

    print(">>> [2/3] Executing Geospatial Processing Pipeline...")
    try:
        subprocess.run([python_exe, "main.py"], check=True)
        print("Pipeline execution complete.\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        sys.exit(1)

    print(">>> [3/3] Booting up Frontend UI Server...")
    try:
        # This will block and keep the server running until the user presses Ctrl+C
        subprocess.run([python_exe, "Frontend_ui/run.py"], check=True)
    except KeyboardInterrupt:
        print("\nServer shutdown requested by user. Terminating processes.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Frontend server crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
