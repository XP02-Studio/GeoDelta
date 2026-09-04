import os
import sys
import subprocess

def main():
    # Ensure we are in the script's directory (Frontend_ui)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    # Use shell=True for npm commands on Windows to avoid executable not found errors
    is_windows = sys.platform.startswith('win')
    
    print("Checking for Node.js (npm)...")
    try:
        subprocess.run(['npm', '-v'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=is_windows)
    except FileNotFoundError:
        print("Error: Node.js (npm) is not installed or not in PATH.")
        print("Please install Node.js from https://nodejs.org/")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("Error: Failed to run npm. Please ensure Node.js is properly installed.")
        sys.exit(1)

    # Check if node_modules exists
    node_modules_path = os.path.join(current_dir, 'node_modules')
    if not os.path.exists(node_modules_path):
        print("Dependencies not found (node_modules is missing).")
        print("Running 'npm install'...")
        try:
            subprocess.run(['npm', 'install'], check=True, shell=is_windows)
            print("Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies. npm install exited with error: {e}")
            sys.exit(1)
    else:
        print("Dependencies already installed.")
    
    print("Starting the development server with 'npm run dev'...")
    try:
        # Run npm run dev and pipe the output to the console
        subprocess.run(['npm', 'run', 'dev'], check=True, shell=is_windows)
    except KeyboardInterrupt:
        print("\nDevelopment server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start the server. npm run dev exited with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
