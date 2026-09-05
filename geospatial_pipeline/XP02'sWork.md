Here is a simplified, copy-paste ready `README.md` that cuts out the jargon and gives your team straightforward commands and instructions.

---

# SIH262270: Satellite Pipeline Setup Guide

A simple guide to set up and run the satellite image processing pipeline.

---

## ⚡ Quick Start (3 Steps)

Run these commands in PowerShell inside the `Satellite_ka_14` folder:

### 1. Create & Activate Virtual Environment
```powershell
python -m venv geodelta_env
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\geodelta_env\Scripts\Activate.ps1

```

### 2. Install Dependencies

```powershell
pip install -r geospatial_pipeline/requirements.txt

```

### 3. Generate Test Data & Run

```powershell
python make_dummy_data.py
python main.py

```

If you see `Pipeline execution completed successfully!`, everything works!

---

## 📂 Where Your Outputs Go

* **ML Model Data (`.npz` files):** `data/patches/`
* **Location Metadata (`.json` files):** `data/patches/`
* **Frontend Map Tiles (Leaflet):** `data/tiles/satellite/`

---

## 🛰️ Working with Real Data

To run the pipeline with real satellite images:

1. Place your base image at `data/raw/t1_baseline.tif`
2. Place your target image at `data/raw/t2_current.tif`
3. Run:
```powershell
python main.py

```



---

## ❓ Common Fixes & Troubleshooting

### Error: `ModuleNotFoundError: No module named 'osgeo'`

* **Cause:** Your terminal is using global Python instead of the virtual environment.
* **Fix:** Make sure `(geodelta_env)` appears at the start of your terminal line. Run:
```powershell
.\geodelta_env\Scripts\Activate.ps1

```



### Error: `Running scripts is disabled on this system`

* **Fix:** Allow PowerShell to run the activation script by executing:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force

```



### Error: `FileNotFoundError` for input GeoTIFFs

* **Fix:** Either run `python make_dummy_data.py` to make test images, or ensure your files are named **exactly** `t1_baseline.tif` and `t2_current.tif` inside `data/raw/`.
