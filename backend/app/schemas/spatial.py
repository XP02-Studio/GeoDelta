from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


# ==========================================
# 1. Search Target Schemas
# ==========================================

class SearchTargetRequest(BaseModel):
    query: str = Field(..., example="new airstrip near Nepal border")
    sector_context: Optional[str] = Field(None, example="Nepal border")


class CameraConfig(BaseModel):
    zoom: int = Field(..., example=15)
    pitch: int = Field(..., example=45)
    bearing: int = Field(..., example=0)


class Coordinates(BaseModel):
    lat: float = Field(..., example=27.7172)
    lng: float = Field(..., example=85.3240)


class SearchTargetResponse(BaseModel):
    status: str = Field("TARGET_LOCKED", example="TARGET_LOCKED")
    target_class: str = Field(..., example="Airstrip")
    coordinates: Coordinates
    camera: CameraConfig
    cache_ready: bool = Field(True, example=True)


# ==========================================
# 2. Sector Analysis & GeoJSON Schemas
# ==========================================

class AnalyzeSectorRequest(BaseModel):
    sector_id: str = Field(..., example="SEC-NEPAL-04")
    bbox: List[float] = Field(
        ..., 
        min_items=4, 
        max_items=4, 
        description="Bounding box [minX, minY, maxX, maxY]",
        example=[85.3100, 27.7000, 85.3400, 27.7300]
    )
    target_query: str = Field(..., example="Airstrip")


class TargetProperties(BaseModel):
    target_id: str = Field(..., example="TGT-8842")
    threat_level: Literal["CRITICAL", "CAUTION", "CLEAR"] = Field(..., example="CRITICAL")
    stroke_color: str = Field(..., example="#EF4444")
    match_confidence: str = Field(..., example="96% (High)")
    area_sq_meters: float = Field(..., example=1420.5)
    gps_display: str = Field(..., example="27°43'01.9\"N 85°19'26.4\"E")


class GeoJSONGeometry(BaseModel):
    type: Literal["Polygon"] = "Polygon"
    coordinates: List[List[List[float]]]


class GeoJSONFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: GeoJSONGeometry
    properties: TargetProperties


class GeoJSONFeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: List[GeoJSONFeature]