import os
import shutil
import sys
import subprocess

def main():
    # Ensure we are in the script's directory (Frontend_ui)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    npm_command = shutil.which("npm.cmd" if sys.platform.startswith("win") else "npm")
    if not npm_command and sys.platform.startswith("win"):
        standard_npm = os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "nodejs", "npm.cmd")
        if os.path.isfile(standard_npm):
            npm_command = standard_npm

    print("Checking for Node.js (npm)...")
    if not npm_command:
        print("Error: Node.js (npm) is not installed or not in PATH.")
        print("Please install Node.js from https://nodejs.org/")
        sys.exit(1)

    npm_env = os.environ.copy()
    node_dir = os.path.dirname(npm_command)
    npm_env["PATH"] = node_dir + os.pathsep + npm_env.get("PATH", "")

    try:
        subprocess.run([npm_command, "-v"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=npm_env)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Error: Failed to run npm. Please ensure Node.js is properly installed.")
        sys.exit(1)

    print("Ensuring dependencies are installed...")
    try:
        subprocess.run([npm_command, "install"], check=True, env=npm_env)
        print("Dependencies check complete.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies. npm install exited with error: {e}")
        sys.exit(1)
    
    print("Starting the development server with 'npm run dev'...")
    try:
        # Run npm run dev and pipe the output to the console
        subprocess.run([npm_command, "run", "dev"], check=True, env=npm_env)
    except KeyboardInterrupt:
        print("\nDevelopment server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start the server. npm run dev exited with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
