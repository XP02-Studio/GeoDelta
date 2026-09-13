"""Live geocoding + STAC satellite imagery discovery.

Downloads small overview thumbnails from STAC (JPEG, ~200KB each)
and serves them as static files for the frontend ImageOverlay.
"""
from __future__ import annotations

import io
import os
import re
import time
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests
from PIL import Image

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
STAC_URL = "https://earth-search.aws.element84.com/v1"
LIVE_CACHE: dict[str, dict[str, Any]] = {}
_GEOCODE_CACHE: dict[str, dict[str, Any]] = {}

_GEOCODER_TIMEOUT = 12
_STAC_TIMEOUT = 15
_IMAGE_DOWNLOAD_TIMEOUT = 15


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).rstrip("/")


def _tiles_dir() -> Path:
    d = Path(__file__).resolve().parents[2] / "tiles" / "live"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _location_terms(query: str) -> list[str]:
    lowered = query.strip()
    candidates = [lowered]
    match = re.search(r"\b(?:near|in|around|at|by)\s+(.+)$", lowered, re.I)
    if match:
        candidates.append(match.group(1).strip())
    return list(dict.fromkeys(c for c in candidates if c))


def geocode(query: str) -> dict[str, Any]:
    cache_key = query.strip().lower()
    if cache_key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[cache_key]

    url = _env("GEOCODER_URL", NOMINATIM_URL)
    headers = {"User-Agent": "GeoDelta-LiveSearch/1.0 (student-project; geodelta-sih)"}
    last_error = None
    for candidate in _location_terms(query):
        try:
            print(f"[geocode] candidate={candidate!r} url={url}")
            resp = requests.get(
                url,
                params={"q": candidate, "format": "jsonv2", "limit": 1},
                headers=headers,
                timeout=_GEOCODER_TIMEOUT,
            )
            print(f"[geocode] status={resp.status_code} body_len={len(resp.text)}")
            if resp.status_code == 429:
                last_error = "Rate limited by geocoder (429)"
                print(f"[geocode] rate limited, waiting 2s")
                time.sleep(2)
                continue
            resp.raise_for_status()
            matches = resp.json()
            if not matches:
                print(f"[geocode] empty results for {candidate!r}")
                continue
            r = matches[0]
            south, north, west, east = map(float, r["boundingbox"])
            result = {
                "label": r.get("display_name", candidate),
                "center": {"lat": float(r["lat"]), "lng": float(r["lon"])},
                "bbox": [west, south, east, north],
            }
            _GEOCODE_CACHE[cache_key] = result
            return result
        except Exception as exc:
            last_error = exc
            print(f"[geocode] error: {exc}")
            continue
    raise LookupError(f"No geographic location could be resolved from the query. Last error: {last_error}")


def _stac_items(bbox: list[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    endpoint = f"{_env('STAC_API_URL', STAC_URL)}/search"
    today = date.today()
    payload = {
        "collections": [_env("STAC_COLLECTION", "sentinel-2-l2a")],
        "bbox": bbox,
        "datetime": f"{today - timedelta(days=365)}T00:00:00Z/{today}T23:59:59Z",
        "limit": 100,
    }
    resp = requests.post(endpoint, json=payload, timeout=_STAC_TIMEOUT)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    if len(features) < 2:
        raise LookupError("STAC returned fewer than two usable observations for this area.")
    return features[0], features[-1]


def _pick_asset(item: dict[str, Any]) -> str:
    assets = item.get("assets", {})
    for key in ("overview", "thumbnail", "rendered_preview", "visual"):
        asset = assets.get(key)
        if asset and asset.get("href"):
            return asset["href"]
    raise LookupError("STAC item has no usable image asset.")


def _download_as_png(href: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(href, timeout=_IMAGE_DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    img.save(destination, format="PNG")


def create_live_search(query: str) -> dict[str, Any]:
    location = geocode(query)
    first, latest = _stac_items(location["bbox"])

    asset_id = uuid.uuid4().hex
    tiles_dir = _tiles_dir() / asset_id
    t1_path = tiles_dir / "t1.png"
    t2_path = tiles_dir / "t2.png"

    t1_href = _pick_asset(first)
    t2_href = _pick_asset(latest)
    print(f"[live_search] t1 asset: {t1_href[:120]}")
    print(f"[live_search] t2 asset: {t2_href[:120]}")
    _download_as_png(t2_href, t2_path)

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
