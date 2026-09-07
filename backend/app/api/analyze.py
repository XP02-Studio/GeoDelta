import asyncio
from fastapi import APIRouter, status
from app.schemas.spatial import (
    AnalyzeSectorRequest,
    GeoJSONFeatureCollection,
    GeoJSONFeature,
    GeoJSONGeometry,
    TargetProperties
)

router = APIRouter(prefix="/api/v1/analyze", tags=["Analyze"])

def run_heavy_inference_and_polygonize(sector_id: str, bbox: list, target_query: str) -> dict:
    """
    Synchronous CPU/GPU heavy pipeline (ChangeFormer + rasterio polygonization + Threat Scoring).
    Ran inside a separate thread to prevent blocking Uvicorn's event loop.
    """
    # 1. Simulate ChangeFormer model generating a change mask & rasterio extracting polygons
    # Coordinates in EPSG:4326 format
    min_x, min_y, max_x, max_y = bbox
    polygon_coords = [
        [
            [min_x + 0.001, min_y + 0.001],
            [max_x - 0.001, min_y + 0.001],
            [max_x - 0.001, max_y - 0.001],
            [min_x + 0.001, max_y - 0.001],
            [min_x + 0.001, min_y + 0.001]
        ]
    ]
    
    # 2. Threat Categorization Logic based on RS-CLIP Similarity Score
    # Score >= 0.85 -> CRITICAL (#EF4444)
    # Score 0.60 - 0.84 -> CAUTION (#EAB308)
    # Score < 0.60 -> CLEAR (#22C55E)
    rs_clip_score = 0.96  # High-confidence example
    
    if rs_clip_score >= 0.85:
        threat_level = "CRITICAL"
        stroke_color = "#EF4444"
    elif rs_clip_score >= 0.60:
        threat_level = "CAUTION"
        stroke_color = "#EAB308"
    else:
        threat_level = "CLEAR"
        stroke_color = "#22C55E"

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": polygon_coords
                },
                "properties": {
                    "target_id": "TGT-8842",
                    "threat_level": threat_level,
                    "stroke_color": stroke_color,
                    "match_confidence": f"{int(rs_clip_score * 100)}% (High)",
                    "area_sq_meters": 1420.5,
                    "gps_display": "27°43'01.9\"N 85°19'26.4\"E"
                }
            }
        ]
    }

@router.post("/sector", response_model=GeoJSONFeatureCollection, status_code=status.HTTP_200_OK)
async def analyze_sector(payload: AnalyzeSectorRequest):
    # Non-blocking dispatch to worker thread via asyncio.to_thread
    geojson_result = await asyncio.to_thread(
        run_heavy_inference_and_polygonize,
        payload.sector_id,
        payload.bbox,
        payload.target_query
    )
    
    return geojson_result