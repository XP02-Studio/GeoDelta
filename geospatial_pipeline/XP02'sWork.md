Here is the updated, battle-tested `README.md` that incorporates all the step-by-step solutions for the PowerShell policy errors, GDAL wheel build failures on Windows, and missing frontend dependencies:

# SIH262270: Air-Gapped Satellite Pipeline

An end-to-end geospatial processing pipeline designed for **offline / air-gapped satellite imagery analysis**. It ingests dual-date GeoTIFFs, applies radiometric normalization and SIFT-based auto-alignment, generates offline XYZ map tiles for Leaflet, and outputs ML-ready `.npz` tensor patches with spatial metadata.

---

## ⚡ Quick Setup & Execution Guide

Follow these exact steps from the project root directory (`Satellite_ka_14`):

### Step 1: Set PowerShell Policy (Run Once)
To prevent Windows execution policy blocks (`Activate.ps1` or `npm.ps1` disabled errors):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

```

### Step 2: Create & Activate Virtual Environment

```powershell
python -m venv geodelta_env
.\geodelta_env\Scripts\Activate.ps1

```

### Step 3: Install Dependencies (Windows GDAL Fix)

If installing GDAL throws C++ build or wheel errors, run the binary installation commands in this order:

```powershell
# 1. Upgrade core pip tooling
python -m pip install --upgrade pip setuptools wheel

# 2. Force pre-compiled binary wheel installation for GDAL
pip install gdal==3.8.4 --only-binary=:all:

# 3. Install remaining pipeline requirements
pip install -r geospatial_pipeline/requirements.txt

```

### Step 4: Run the Pipeline

```powershell
# Generate sample GeoTIFF test data (Optional if data/raw/ is populated)
python make_dummy_data.py

# Run the main geospatial processing pipeline
python main.py

```

---

## 🌐 Running the Frontend UI

If `vite` is not recognized or server startup fails when running `Frontend_ui/run.py`:

```powershell
# 1. Navigate to the frontend directory
cd Frontend_ui

# 2. Install/re-link Node modules and Vite binaries
npm install

# 3. Return to root and run the launch script
cd ..
python -u "Frontend_ui/run.py"

```

---

## 🛰️ Working with Real Satellite Data

Place temporal satellite images inside `data/raw/`:

```text
data/raw/
├── t1_baseline.tif   # Base / Reference image
└── t2_current.tif    # Target / Current image

```

Then run `python main.py`.

---

## 📂 Output Handoff Locations

| Output | Location | Purpose |
| --- | --- | --- |
| **ML Tensor Patches (`.npz`)** | `data/patches/` | PyTorch model training |
| **Spatial Metadata (`.json`)** | `data/patches/` | Vector indexing & GPS search |
| **Offline Map Tiles (`/{z}/{x}/{y}.png`)** | `data/tiles/satellite/` | Offline Leaflet / OpenLayers web UI |

---

## ❓ Troubleshooting Common Setup Issues

* **`ModuleNotFoundError: No module named 'osgeo'` or `'numpy'**`
* Your terminal is running outside the environment. Make sure `(geodelta_env)` shows in your prompt. Re-run: `.\geodelta_env\Scripts\Activate.ps1`.


* **`ERROR: Could not build wheels for gdal`**
* Run `pip install gdal==3.8.4 --only-binary=:all:` to avoid compiling from source.


* **`'vite' is not recognized as an internal command`**
* Run `npm install` inside the `Frontend_ui` folder.


* **Script execution errors on PowerShell**
* Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force`.
