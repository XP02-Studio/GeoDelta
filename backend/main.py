import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from app.api import search, analyze

# Import the engine at the top of your file if it isn't already there
from app.db.vector_store import vector_engine

app = FastAPI(
    title="Tactical Imagery Analysis Backend",
    version="1.0.0",
    description="Offline-ready geospatial change detection and semantic search API."
)

@app.on_event("startup")
async def startup_event():
    print("Initializing Qdrant Vector Database pool...")
    await vector_engine.startup()

@app.on_event("shutdown")
async def shutdown_event():
    await vector_engine.close()

@app.get("/")
def read_root():
    return {"status": "online", "message": "Geodelta Backend is running"}

from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:5173", # Vite local dev
    "http://localhost:5174", # Vite local dev (alternative port)
    "https://geodelta-sih.vercel.app", # Vercel production
    os.getenv("VITE_API_URL", "") # Fallback from env
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Endpoint Routers
app.include_router(search.router)
app.include_router(analyze.router)

# Mount local directory for temporal pair tiles ($T_1$ and $T_2$)
TILES_DIRECTORY = os.path.join(os.path.dirname(__file__), "tiles")
os.makedirs(TILES_DIRECTORY, exist_ok=True)
app.mount("/tiles", StaticFiles(directory=TILES_DIRECTORY), name="tiles")


@app.get("/api/v1/imagery/pair")
async def get_raster_chip_pair(bbox: str):
    """
    Raster Chip Endpoint serving high-res WebP/PNG chips with strict caching.
    """
    response = JSONResponse(content={
        "status": "SUCCESS",
        "bbox": bbox,
        "t1_chip_url": f"/tiles/t1_chip.webp",
        "t2_chip_url": f"/tiles/t2_chip.webp"
    })
    # Set strict local caching headers for immediate swipe rendering
    response.headers["Cache-Control"] = "public, max-age=86400"
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
