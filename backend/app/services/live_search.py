"""Live geocoding, STAC discovery, and local image-pair caching.

Lightweight version — no rasterio/GDAL required. Downloads pre-rendered
STAC thumbnails directly via requests + Pillow.
"""
from __future__ import annotations

import io
import os
import re
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests
from PIL import Image


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
STAC_URL = "https://earth-search.aws.element84.com/v1"
LIVE_CACHE: dict[str, dict[str, Any]] = {}


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).rstrip("/")


def _location_terms(query: str) -> list[str]:
    lowered = query.strip()
    candidates = [lowered]
    match = re.search(r"\b(?:near|in|around|at|by)\s+(.+)$", lowered, re.I)
    if match:
        candidates.append(match.group(1).strip())
    return list(dict.fromkeys(c for c in candidates if c))


def geocode(query: str) -> dict[str, Any]:
    url = _env("GEOCODER_URL", NOMINATIM_URL)
    headers = {"User-Agent": "GeoDelta-LiveSearch/1.0"}
    for candidate in _location_terms(query):
        try:
            resp = requests.get(
                url,
                params={"q": candidate, "format": "jsonv2", "limit": 1},
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            matches = resp.json()
            if not matches:
                continue
            r = matches[0]
            south, north, west, east = map(float, r["boundingbox"])
            return {
                "label": r.get("display_name", candidate),
                "center": {"lat": float(r["lat"]), "lng": float(r["lon"])},
                "bbox": [west, south, east, north],
            }
        except Exception:
            continue
    raise LookupError("No geographic location could be resolved from the query.")


def _stac_items(bbox: list[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    endpoint = f"{_env('STAC_API_URL', STAC_URL)}/search"
    today = date.today()
    payload = {
        "collections": [_env("STAC_COLLECTION", "sentinel-2-l2a")],
        "bbox": bbox,
        "datetime": f"{today - timedelta(days=365)}T00:00:00Z/{today}T23:59:59Z",
        "limit": 100,
    }
    resp = requests.post(endpoint, json=payload, timeout=30)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    if len(features) < 2:
        raise LookupError("STAC returned fewer than two usable observations for this area.")
    return features[0], features[-1]


def _visual_asset(item: dict[str, Any]) -> str:
    assets = item.get("assets", {})
    for key in ("visual", "rendered_preview", "thumbnail", "overview"):
        href = assets.get(key, {}).get("href")
        if href:
            return href
    raise LookupError("The selected STAC item has no displayable asset.")


def _download_and_save(asset_href: str, destination: Path) -> None:
    """Download an image asset and save as PNG."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(asset_href, timeout=60)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img.save(destination, format="PNG")


def create_live_search(query: str) -> dict[str, Any]:
    location = geocode(query)
    first, latest = _stac_items(location["bbox"])
    asset_id = uuid.uuid4().hex
    tiles_dir = Path(__file__).resolve().parents[2] / "tiles" / "live" / asset_id
    t1_path, t2_path = tiles_dir / "t1.png", tiles_dir / "t2.png"

    _download_and_save(_visual_asset(first), t1_path)
    _download_and_save(_visual_asset(latest), t2_path)

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
