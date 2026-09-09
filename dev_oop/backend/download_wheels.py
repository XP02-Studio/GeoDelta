"""
Pre-Hackathon Wheel Downloader Utility
Execute this script on an internet-connected machine before air-gapping:
    python download_wheels.py
This populates the backend/wheels/ directory with all platform-compatible wheels.
"""
import subprocess
import sys
import os

def download_wheels():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    wheels_dir = os.path.join(current_dir, "wheels")
    requirements_file = os.path.join(current_dir, "requirements.txt")

    os.makedirs(wheels_dir, exist_ok=True)
    print(f"[*] Downloading offline wheels to: {wheels_dir}")
    
    cmd = [
        sys.executable, "-m", "pip", "wheel",
        "-r", requirements_file,
        "-w", wheels_dir,
        "--prefer-binary"
    ]
    
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("[+] Wheels successfully downloaded for air-gapped deployment!")
    else:
        print("[-] Failed to download wheels.")

if __name__ == "__main__":
    download_wheels()
