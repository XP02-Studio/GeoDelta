import subprocess
import sys
import os
import time

def main():
    print("==================================================")
    print("   SIH262270: Universal Application Runner        ")
    print("==================================================\n")
    
    # Resolve absolute paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "backend")
    frontend_dir = os.path.join(root_dir, "frontend")
    
    print("[+] Initializing FastAPI Backend on Port 8000...")
    backend_cmd = ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    backend_process = subprocess.Popen(backend_cmd, cwd=backend_dir)
    
    # Allow backend a moment to spin up
    time.sleep(2)
    
    print("[+] Initializing Vite React Frontend...")
    # shell=True required for npm execution on Windows
    frontend_cmd = ["npm", "run", "dev"]
    frontend_process = subprocess.Popen(frontend_cmd, cwd=frontend_dir, shell=True)
    
    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\n[!] Shutting down pipeline gracefully...")
        backend_process.terminate()
        frontend_process.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main()
