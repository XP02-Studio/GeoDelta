"""Live geocoding, STAC discovery, and local image-pair caching.

The service is deliberately endpoint-configurable: staging can use public
Nominatim/STAC services, while an air-gapped deployment can point the same
variables at locally mirrored services.
"""
from __future__ import annotations

import re
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
STAC_URL = "https://earth-search.aws.element84.com/v1"
LIVE_CACHE: dict[str, dict[str, Any]] = {}


def _environment(name: str, default: str) -> str:
    import os
    return os.getenv(name, default).rstrip("/")


def _location_terms(query: str) -> list[str]:
    """Try the complete query first, then the place phrase after a locator."""
    lowered = query.strip()
    candidates = [lowered]
    match = re.search(r"\b(?:near|in|around|at|by)\s+(.+)$", lowered, re.I)
    if match:
        candidates.append(match.group(1).strip())
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def geocode(query: str) -> dict[str, Any]:
    url = _environment("GEOCODER_URL", NOMINATIM_URL)
    headers = {"User-Agent": "GeoDelta-LiveSearch/1.0 (hackathon-demo)"}
    for candidate in _location_terms(query):
        response = requests.get(
            url,
            params={"q": candidate, "format": "jsonv2", "limit": 1},
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        matches = response.json()
        if not matches:
            continue
        result = matches[0]
        south, north, west, east = map(float, result["boundingbox"])
        return {
            "label": result.get("display_name", candidate),
            "center": {"lat": float(result["lat"]), "lng": float(result["lon"])},
            # GeoJSON / STAC ordering: west, south, east, north.
            "bbox": [west, south, east, north],
        }
    raise LookupError("No geographic location could be resolved from the query.")


def _stac_items(bbox: list[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    endpoint = f"{_environment('STAC_API_URL', STAC_URL)}/search"
    today = date.today()
    payload = {
        "collections": [_environment("STAC_COLLECTION", "sentinel-2-l2a")],
        "bbox": bbox,
        "datetime": f"{today - timedelta(days=365)}T00:00:00Z/{today}T23:59:59Z",
        "limit": 100,
        "sortby": [{"field": "properties.datetime", "direction": "asc"}],
    }
    response = requests.post(endpoint, json=payload, timeout=30)
    response.raise_for_status()
    features = response.json().get("features", [])
    if len(features) < 2:
        raise LookupError("STAC returned fewer than two usable observations for this area.")
    return features[0], features[-1]


def _visual_asset(item: dict[str, Any]) -> str:
    assets = item.get("assets", {})
    for key in ("visual", "rendered_preview", "thumbnail"):
        href = assets.get(key, {}).get("href")
        if href:
            return href
    raise LookupError("The selected STAC item has no displayable multi-band asset.")


def _write_png(asset_href: str, bbox: list[float], destination: Path) -> None:
    """Read just the requested geospatial window and normalize it for the UI/ML API."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    with rasterio.open(asset_href) as dataset:
        source_bounds = transform_bounds("EPSG:4326", dataset.crs, *bbox, densify_pts=21)
        window = from_bounds(*source_bounds, transform=dataset.transform).round_offsets().round_lengths()
        if window.width <= 0 or window.height <= 0:
            raise ValueError("The STAC asset does not overlap the geocoded bounding box.")
        scale = min(1.0, 1024 / max(window.width, window.height))
        out_height = max(1, round(window.height * scale))
        out_width = max(1, round(window.width * scale))
        indexes = list(range(1, min(dataset.count, 3) + 1))
        image = dataset.read(
            indexes=indexes,
            window=window,
            boundless=True,
            out_shape=(len(indexes), out_height, out_width),
            resampling=Resampling.bilinear,
        )

    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)
    elif image.shape[0] == 2:
        image = np.concatenate([image, image[:1]], axis=0)
    low, high = np.percentile(image, (2, 98))
    normalized = np.clip((image - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.moveaxis(normalized, 0, -1), mode="RGB").save(destination)


def create_live_search(query: str) -> dict[str, Any]:
    location = geocode(query)
    first, latest = _stac_items(location["bbox"])
    asset_id = uuid.uuid4().hex
    tiles_dir = Path(__file__).resolve().parents[2] / "tiles" / "live" / asset_id
    t1_path, t2_path = tiles_dir / "t1.png", tiles_dir / "t2.png"
    _write_png(_visual_asset(first), location["bbox"], t1_path)
    _write_png(_visual_asset(latest), location["bbox"], t2_path)

    LIVE_CACHE[asset_id] = {
        "t1_path": str(t1_path),
        "t2_path": str(t2_path),
        "bbox": location["bbox"],
        "query": query,
    }
    return {
        "status": "TARGET_READY",
        "query": query,
        "target_label": location["label"],
        "coordinates": location["center"],
        "bbox": location["bbox"],
        "asset_id": asset_id,
        "imagery": {
            "t1_url": f"/tiles/live/{asset_id}/t1.png",
            "t2_url": f"/tiles/live/{asset_id}/t2.png",
            "t1_datetime": first.get("properties", {}).get("datetime"),
            "t2_datetime": latest.get("properties", {}).get("datetime"),
        },
    }


def get_live_pair(asset_id: str) -> dict[str, Any]:
    pair = LIVE_CACHE.get(asset_id)
    if not pair:
        raise LookupError("This live-search imagery pair is unavailable. Run the search again.")
    return pair
