from fastapi import APIRouter, HTTPException, status
from app.schemas.spatial import SearchTargetRequest, SearchTargetResponse, Coordinates, CameraConfig

router = APIRouter(prefix="/api/v1/search", tags=["Search"])

# Dummy local gazetteer dictionary mapping landmark keywords for air-gapped lookup
LOCAL_GAZETTEER = {
    "nepal border": {
        "target_class": "Airstrip",
        "lat": 27.7172,
        "lng": 85.3240,
        "zoom": 15,
        "pitch": 45,
        "bearing": 0
    },
    "lac sector 4": {
        "target_class": "Helipad",
        "lat": 34.1526,
        "lng": 77.5771,
        "zoom": 14,
        "pitch": 30,
        "bearing": 15
    },
    "sri lanka border": {
        "target_class": "Naval Base",
        "lat": 9.3820,
        "lng": 79.8988,
        "zoom": 15,
        "pitch": 40,
        "bearing": -20
    }
}

def mock_rs_clip_text_encoder(text: str) -> list:
    """
    Simulates generating a 512-dimensional query vector using an offline RS-CLIP model.
    """
    # Return a dummy 512-D vector
    return [0.042] * 512

@router.post("/target", response_model=SearchTargetResponse, status_code=status.HTTP_200_OK)
async def search_target(payload: SearchTargetRequest):
    query_lower = payload.query.lower()
    
    # 1. Offline Entity & Coordinate Extraction via Gazetteer Lookup
    matched_entry = None
    for location_key, data in LOCAL_GAZETTEER.items():
        if location_key in query_lower or (payload.sector_context and location_key in payload.sector_context.lower()):
            matched_entry = data
            break
            
    # Default location fallback if no exact gazetteer match is found
    if not matched_entry:
        matched_entry = {
            "target_class": "Structure",
            "lat": 27.7172,
            "lng": 85.3240,
            "zoom": 15,
            "pitch": 45,
            "bearing": 0
        }

    # 2. Extract 512-dimensional query vector using RS-CLIP
    _ = mock_rs_clip_text_encoder(payload.query)

    return SearchTargetResponse(
        status="TARGET_LOCKED",
        target_class=matched_entry["target_class"],
        coordinates=Coordinates(lat=matched_entry["lat"], lng=matched_entry["lng"]),
        camera=CameraConfig(
            zoom=matched_entry["zoom"],
            pitch=matched_entry["pitch"],
            bearing=matched_entry["bearing"]
        ),
        cache_ready=True
    )