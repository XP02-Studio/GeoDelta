import subprocess
import sys
import time
import atexit
import os

def main():
    print("🚀 Starting Geodelta Tactical Platform...")
    
    # 1. Start FastAPI Backend (Port 8000)
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", "8000"], 
        cwd="backend"
    )
    
    # 2. Start ML Engine (Port 5000)
    ml_proc = subprocess.Popen(
        [sys.executable, "-m", "src.server"], 
        cwd="ml-engineer"
    )
    
    # 3. Start React Frontend (Port 5173)
    # Using shell=False and platform-specific npm to prevent zombie processes
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"], 
        cwd="frontend"
    )

    # 4. Clean Shutdown Handler
    def cleanup():
        print("\n🛑 Shutting down all Geodelta services...")
        backend_proc.terminate()
        ml_proc.terminate()
        frontend_proc.terminate()
        
    atexit.register(cleanup)

    try:
        # Keep the main thread alive while subprocesses run
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExit command received.")
        # atexit handler will automatically trigger cleanup()

if __name__ == "__main__":
    main()