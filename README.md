### 1. Go to the folder 

```powershell
cd geospatial_pipeline/
```
### 2. Run 

```powershell
python run_all.py
```
### To run frontend and geospatial pipeline together  
```
📂 GeoDelta Project Directory Blueprint
geodelta/
├── docker-compose.yml           # DevOps: Orchestrates all 5 local service containers
├── .env.example                 # Global configuration for offline ports & database keys
├── README.md                    # Project documentation & SIH setup instructions
│
├── backend/                     # BACKEND & PRESENTATION LEAD (FastAPI Server) [1, 2]
│   ├── Dockerfile               # Configured for offline Python wheel installation [3, 4]
│   ├── requirements.txt         # FastAPI, uvicorn, qdrant-client, psycopg2, etc. [2]
│   ├── main.py                  # API entry point & Uvicorn server configuration
│   └── app/
│       ├── __init__.py
│       ├── api/                 # Endpoint routers [2]
│       │   ├── search.py        # POST /api/v1/search/target (RS-CLIP query endpoint) [1]
│       │   └── analyze.py       # POST /api/v1/analyze/sector (ChangeFormer trigger) [3]
│       ├── core/                # Server settings & security configurations [2]
│       │   └── config.py
│       ├── db/                  # Database connections & session makers [2]
│       │   ├── postgres.py      # PostGIS client initialization
│       │   └── vector_store.py  # Qdrant client connector [2]
│       ├── services/            # Bridge between API and offline ML pipeline [2]
│       │   └── orchestrator.py  # Sub-process manager for async ML inference execution [2]
│       └── schemas/             # Pydantic schemas (validates UI request/response payloads) [1, 3]
│           └── spatial.py
│
├── frontend/                    # FRONTEND & WEBGL DEVELOPER (React / Vite) [5, 6]
│   ├── Dockerfile               # Multi-stage build (Vite Node environment -> local NGINX server) [4]
│   ├── package.json             # React, Three.js, react-leaflet, gsap, @turf/area [5, 7]
│   ├── tailwind.config.js       # Set up for Matte Tactical Dark theme styling [5, 8]
│   ├── public/                  # Raw UI assets (icons, temporary loading animations, logos)
│   └── src/
│       ├── main.jsx             # Entry script
│       ├── App.jsx              # Controls views (Landing Modal -> 3D Globe -> 2D Leaflet) [7]
│       ├── context/             # Global AuthState and SectorContext states [5, 6]
│       │   └── UIContext.jsx
│       ├── services/            # Axios API calls mapping frontend to backend endpoints [5, 7]
│       │   └── api.js
│       └── components/          # Reusable UI parts [5]
│           ├── Map/             # 2D Leaflet map, swipe compare slider, bounding boxes [5, 7]
│           │   ├── MapViewport.jsx
│           │   └── SwipeSlider.jsx
│           ├── UI/              # Sidebar (Wizard), Target Drawer (UVP Metrics), Telemetry HUD [5, 7]
│           │   ├── SidebarWizard.jsx
│           │   ├── TargetIntelDrawer.jsx
│           │   └── TelemetryHUD.jsx
│           └── WebGL/           # Three.js 3D rotating Earth, orbiting satellites & space [7, 9]
│               ├── SpaceCanvas.jsx
│               ├── RotatingEarth.jsx
│               └── Satellites.jsx
│
├── ml-engine/                   # DEEP LEARNING CORE (PyTorch / Inference Scripts) [10]
│   ├── Dockerfile               # Built on nvidia/cuda runtime for local GPU passthrough [4]
│   ├── requirements.txt         # PyTorch, TorchGeo, transformers, ONNX Runtime [10, 11]
│   ├── models/                  # PERSISTENT LOCAL WEIGHT STORAGE (100% Offline) [10]
│   │   ├── changeformer_v1.pt   # Downloaded ChangeFormer baseline model weights [10]
│   │   └── rs_clip/             # Local tokenizer & vocab configurations for RS-CLIP [10]
│   └── src/
│       ├── inference_engine.py  # Singleton model loader classes for local VRAM optimization [10]
│       ├── change_former.py     # Siamese Network inference & multi-class mask segmentation [12]
│       └── rs_clip_search.py    # Zero-shot embedding generator for plain-English query matching [13]
│
├── preprocessing/               # GEOSPATIAL DATA ENGINEER (GDAL Pipeline) [14, 15]
│   ├── Dockerfile               # Configured to host GDAL/Rasterio dependencies locally [15]
│   └── src/
│       ├── main_pipeline.py     # Script orchestrating tiler, aligner, and normalizer [14]
│       ├── tiler.py             # GDAL sliding-window patch tiler (512x512 slices with 20% overlap) [14, 15]
│       ├── co_registration.py   # SIFT/ORB feature auto-alignment (Zero Jitter pipeline) [14, 15]
│       ├── normalizer.py        # Radiometric normalizer (2%-98% percentile clipping) [14, 15]
│       └── tile_converter.py    # Standardizes preprocessed tensors into PyTorch binary formats (.npz) [14, 15]
│
└── local_storage/               # DEVOPS / DATA SPECS: Persistent Local Hardware Directories [4, 15]
    ├── qdrant_storage/          # Persistent folder for local Qdrant collection vectors [16, 17]
    ├── postgis_data/            # Persistent local folder holding PostgreSQL coordinate tables [16]
    └── local_map_tiles/         # XYZ tile pyramid folder (/z/x/y.png) served by backend offline [5, 15]