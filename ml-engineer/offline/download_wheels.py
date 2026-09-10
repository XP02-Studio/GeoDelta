"""
Air-Gapped Offline Wheel & Asset Pre-Fetcher
Run this on an internet-connected machine to pre-fetch all Python wheels and assets
into the local offline/ directory before deploying to an air-gapped environment.
"""

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"
WHEELS_DIR = BASE_DIR / "wheels"

def fetch_wheels():
    print(f"[*] Preparing offline wheels directory at: {WHEELS_DIR}")
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "pip", "download",
        "-r", str(REQUIREMENTS_FILE),
        "-d", str(WHEELS_DIR),
        "--prefer-binary"
    ]
    
    print(f"[*] Executing command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("[OK] All wheels successfully downloaded to offline/wheels/")
    except subprocess.CalledProcessError as e:
        print(f"[!] Error downloading wheels: {e}")
        return False
    return True

if __name__ == "__main__":
    fetch_wheels()
