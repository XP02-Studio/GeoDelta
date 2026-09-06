### 1. Go to the folder 

```powershell
cd geospatial_pipeline
```
### 2. Run 

```powershell
python run_all.py
```
### To run frontend and geospatial pipeline together  
```

📂 GeoDelta Project Directory Structure
geodelta/

├── docker-compose.yml              # DevOps: Orchestrates all local service containers
├── .env.example                    # Global configuration for offline ports & database keys
├── README.md                       # Project documentation & SIH setup instructions

│
├── backend/                        # MEMBER 1 — BACKEND LEAD
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   └── app/
│       ├── __init__.py
│       ├── api/
│       │   ├── search.py
│       │   └── analyze.py
│       ├── core/
│       │   └── config.py
│       ├── db/
│       │   ├── postgres.py
│       │   └── vector_store.py
│       ├── services/
│       │   └── orchestrator.py
│       └── schemas/
│           └── spatial.py

│
├── frontend/                       # MEMBER 2 — FRONTEND & UI/UX DEVELOPER
│   ├── Dockerfile
│   ├── package.json
│   ├── tailwind.config.js
│   ├── public/
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── context/
│       │   └── UIContext.jsx
│       ├── services/
│       │   └── api.js
│       └── components/
│           ├── Map/
│           │   ├── MapViewport.jsx
│           │   └── SwipeSlider.jsx
│           ├── UI/
│           │   ├── SidebarWizard.jsx
│           │   ├── TargetIntelDrawer.jsx
│           │   └── TelemetryHUD.jsx
│           └── WebGL/
│               ├── SpaceCanvas.jsx
│               ├── RotatingEarth.jsx
│               └── Satellites.jsx

│
├── preprocessing/                  # MEMBER 3 — GEOSPATIAL DATA ENGINEER
│   ├── Dockerfile
│   └── src/
│       ├── main_pipeline.py
│       ├── tiler.py
│       ├── co_registration.py
│       ├── normalizer.py
│       └── tile_converter.py

│
├── vector-search/                  # MEMBER 4 — VECTOR DATABASE & SEARCH SPECIALIST
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── qdrant_manager.py
│       ├── collection_setup.py
│       ├── embedding_indexer.py
│       └── semantic_search.py

│
├── ml-engine/                      # MEMBER 5 — DEEP LEARNING CORE (ML ENGINEER)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── models/
│   │   ├── changeformer_v1.pt
│   │   └── rs_clip/
│   └── src/
│       ├── inference_engine.py
│       ├── change_former.py
│       └── rs_clip_search.py

│
└── local_storage/                  # MEMBER 6 — DEVOPS & DEPLOYMENT LEAD
    ├── qdrant_storage/
    ├── postgis_data/
    └── local_map_tiles/


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                         TEAM MEMBERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Member 1 (Backend)                              :- Dhurbo
Member 2 (Frontend & UI/UX Developer)           :- Rehan
Member 3 (Geospatial Data Engineer)             :- Ayush
Member 4 (Vector Database & Search Specialist)  :- Adarsh
Member 5 (Deep Learning Core / ML Engineer)     :- Binit
Member 6 (DevOps & Deployment Lead)             :- Sneha
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━