# SIH262270: Air-Gapped Satellite Pipeline

An end-to-end, reproducible geospatial processing pipeline designed for **offline / air-gapped satellite imagery analysis**.

The pipeline ingests dual-date GeoTIFF imagery, performs radiometric normalization and sub-pixel image alignment using **SIFT feature matching**, generates **offline XYZ map tiles** for web-based visualization, and produces **ML-ready tensor patches with GPS/spatial metadata**.

---

## Features

* Dual-date satellite imagery processing
* GeoTIFF ingestion and validation
* CRS reprojection
* Percentile-based radiometric normalization
* Sub-pixel image alignment
* SIFT feature detection and matching
* Affine image warping
* Offline XYZ satellite tile generation
* Spatial patch extraction
* ML-ready `.npz` tensor generation
* GPS / bounding-box metadata generation
* Fully reproducible offline pipeline
* Synthetic test-data generation for development

---

## Output Handoff Locations

| Output                                 | Location                | Purpose                               |
| -------------------------------------- | ----------------------- | ------------------------------------- |
| ML Tensor Patches (`.npz`)             | `data/patches/`         | PyTorch / ML model training           |
| Spatial Metadata (`.json`)             | `data/patches/`         | Vector indexing & bounding-box search |
| Offline Map Tiles (`/{z}/{x}/{y}.png`) | `data/tiles/satellite/` | Leaflet / OpenLayers web UI           |

---

# Quick Setup & Execution

The following instructions assume **Windows PowerShell** and that the commands are executed from the project root:

```text
Satellite_ka_14/
```

## 1. Create Virtual Environment

```powershell
python -m venv geodelta_env
```

---

## 2. Enable PowerShell Script Execution

If PowerShell prevents the virtual environment activation script from running, execute:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
```

This changes the execution policy only for the current PowerShell session.

---

## 3. Activate Environment & Install Dependencies

Activate the virtual environment:

```powershell
.\geodelta_env\Scripts\Activate.ps1
```

Then install the required dependencies:

```powershell
pip install -r geospatial_pipeline/requirements.txt
```

---

## 4. Generate Synthetic Test Data

If real satellite imagery is not available, synthetic GeoTIFF test data can be generated automatically.

Run:

```powershell
python make_dummy_data.py
```

This creates sample imagery that can be used to verify that the pipeline is functioning correctly.

> This step is optional when real GeoTIFF imagery has already been placed in `data/raw/`.

---

## 5. Run the Pipeline

From the project root:

```powershell
python main.py
```

The pipeline will process the available imagery and generate the required outputs.

---

# Working with Real Satellite Data

The pipeline can process compatible satellite GeoTIFF imagery such as **Sentinel-2** or **Landsat** data.

Place the two temporal images inside:

```text
data/raw/
```

Use the following filenames:

```text
data/raw/
├── t1_baseline.tif
└── t2_current.tif
```

Where:

* `t1_baseline.tif` — earlier / reference satellite image
* `t2_current.tif` — newer / target satellite image

Then execute:

```powershell
python main.py
```

The pipeline will use the baseline image as the spatial reference and process the current image accordingly.

---

# Pipeline Workflow

The processing pipeline follows the general sequence:

```text
Dual-Date GeoTIFF Input
        │
        ▼
┌──────────────────────────┐
│      Data Ingestion      │
│  CRS + Normalization     │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│    Image Alignment       │
│ SIFT + Feature Matching  │
│ + Affine Warping         │
└────────────┬─────────────┘
             │
             ├──────────────────────┐
             │                      │
             ▼                      ▼
┌──────────────────────┐   ┌──────────────────────┐
│  Offline Basemap     │   │    Patch Tiling      │
│   XYZ Tile Output    │   │   Tensor Generation  │
└──────────┬───────────┘   └──────────┬───────────┘
           │                          │
           ▼                          ▼
   data/tiles/satellite/       data/patches/
                                      │
                                      ├── .npz
                                      └── .json
```

---

# Project Architecture

```text
Satellite_ka_14/
│
├── data/
│   ├── raw/
│   │   ├── t1_baseline.tif
│   │   └── t2_current.tif
│   │
│   ├── processed/
│   │   └── # Re-projected / warped imagery
│   │
│   ├── patches/
│   │   ├── *.npz
│   │   └── *.json
│   │
│   └── tiles/
│       └── satellite/
│           └── {z}/{x}/{y}.png
│
├── geospatial_pipeline/
│   ├── config.py
│   ├── main.py
│   ├── requirements.txt
│   │
│   └── pipeline/
│       ├── ingestion.py
│       ├── alignment.py
│       ├── basemap.py
│       └── patch_tiler.py
│
├── make_dummy_data.py
├── main.py
└── README.md
```

---

# Core Components

## `ingestion.py`

Responsible for initial satellite-image preparation.

Main responsibilities include:

* GeoTIFF ingestion
* CRS handling
* Reprojection
* Radiometric normalization
* Percentile-based normalization
* Preparation of imagery for downstream processing

---

## `alignment.py`

Handles spatial registration between the two temporal images.

The alignment stage uses:

* SIFT keypoint detection
* Feature descriptor extraction
* Feature matching
* Geometric transformation estimation
* Affine warping

The objective is to minimize spatial misalignment between the baseline and current imagery before further analysis.

---

## `basemap.py`

Generates offline XYZ map tiles from processed satellite imagery.

The generated structure follows:

```text
data/tiles/satellite/
└── {z}/
    └── {x}/
        └── {y}.png
```

This structure is compatible with common web mapping frameworks such as:

* Leaflet
* OpenLayers

Because the tiles are generated locally, the resulting map layer can be served in an offline or air-gapped environment.

---

## `patch_tiler.py`

Converts processed imagery into smaller spatial patches suitable for machine-learning workflows.

Outputs include:

```text
*.npz
```

for numerical tensor data and:

```text
*.json
```

for corresponding spatial metadata.

The metadata can contain information required for spatial indexing and bounding-box based retrieval.

---

## `config.py`

Contains configurable pipeline parameters such as:

* Input/output directories
* Patch dimensions
* Tiling parameters
* Processing configuration
* Output paths

Modify this file when adapting the pipeline to different datasets or deployment requirements.

---

# Output Structure

After a successful execution, the generated data is organized approximately as follows:

```text
data/
│
├── raw/
│   ├── t1_baseline.tif
│   └── t2_current.tif
│
├── processed/
│   └── ...
│
├── patches/
│   ├── patch_0001.npz
│   ├── patch_0001.json
│   ├── patch_0002.npz
│   ├── patch_0002.json
│   └── ...
│
└── tiles/
    └── satellite/
        ├── 0/
        ├── 1/
        ├── 2/
        └── ...
```

---

# Machine Learning Handoff

The generated `.npz` files are intended to serve as the handoff between the geospatial preprocessing pipeline and the machine-learning stage.

Example:

```text
data/patches/
├── patch_0001.npz
├── patch_0001.json
├── patch_0002.npz
├── patch_0002.json
└── ...
```

The tensor data can subsequently be loaded into a Python ML workflow, including PyTorch-based training or inference pipelines.

The accompanying JSON metadata provides the spatial context necessary to associate each tensor patch with its geographic location.

---

# Offline Web Visualization

The generated XYZ tiles can be consumed by an offline web interface.

Expected tile format:

```text
/{z}/{x}/{y}.png
```

Example local tile path:

```text
data/tiles/satellite/12/2048/1365.png
```

A frontend such as Leaflet or OpenLayers can use these locally generated tiles as a raster map layer without requiring an internet connection.

---

# Air-Gapped Deployment

The pipeline is designed with offline environments in mind.

Once the required Python dependencies, source code, satellite imagery, and processing resources are available locally, the core workflow can execute without relying on external online geospatial services.

Recommended deployment structure:

```text
AIR-GAPPED SYSTEM
│
├── Python Environment
│
├── Satellite_ka_14/
│   ├── Source Code
│   ├── Configuration
│   ├── Raw Imagery
│   ├── Processed Imagery
│   ├── ML Patches
│   └── Offline Map Tiles
│
└── Local Web Interface
        │
        ├── Leaflet / OpenLayers
        └── Local XYZ Tiles
```

---

# Requirements

Dependencies are maintained in:

```text
geospatial_pipeline/requirements.txt
```

The project primarily relies on geospatial and computer-vision tooling including:

* GDAL
* OpenCV
* NumPy
* Python geospatial libraries

Install all project dependencies with:

```powershell
pip install -r geospatial_pipeline/requirements.txt
```

---

# Troubleshooting

## PowerShell blocks activation

If you encounter an execution-policy error:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
```

Then activate the environment again:

```powershell
.\geodelta_env\Scripts\Activate.ps1
```

---

## Input GeoTIFFs are not detected

Verify that the files exist at exactly:

```text
data/raw/t1_baseline.tif
data/raw/t2_current.tif
```

Check:

* Filename spelling
* `.tif` extension
* File location
* Read permissions
* GeoTIFF validity

---

## Testing without satellite imagery

Use:

```powershell
python make_dummy_data.py
```

Then execute:

```powershell
python main.py
```

This provides a controlled test environment before processing real satellite imagery.

---

# Reproducibility

For reproducible execution:

1. Use the provided virtual environment.
2. Install dependencies from `requirements.txt`.
3. Keep input imagery in the expected directory structure.
4. Avoid modifying pipeline configuration unless required.
5. Record configuration changes when running experiments.
6. Preserve generated patch metadata alongside the corresponding `.npz` files.

---

# Typical Execution

A complete fresh setup can be performed with:

```powershell
python -m venv geodelta_env

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force

.\geodelta_env\Scripts\Activate.ps1

pip install -r geospatial_pipeline/requirements.txt

python make_dummy_data.py

python main.py
```

For production / real-data processing, replace the synthetic-data step with the required GeoTIFF inputs.

---

# Project Status

**SIH Problem Statement:** `SIH262270`

**Project:** Air-Gapped Satellite Pipeline

**Primary Objective:**
Provide a reproducible offline geospatial preprocessing pipeline capable of transforming dual-date satellite imagery into aligned, spatially indexed, ML-ready data and locally consumable map tiles.

---

## License

Add the project's applicable license here.

```text
© 2026 Satellite_ka_14
```
