"""Live geocoding + STAC satellite imagery discovery.

Lightweight version — no rasterio/GDAL, no image downloads.
Returns STAC asset URLs directly so the frontend can render them
without blocking the search response.
"""
from __future__ import annotations

import os
import re
from datetime import date, timedelta
from typing import Any

import time
import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
STAC_URL = "https://earth-search.aws.element84.com/v1"
LIVE_CACHE: dict[str, dict[str, Any]] = {}
_GEOCODE_CACHE: dict[str, dict[str, Any]] = {}

_GEOCODER_TIMEOUT = 12
_STAC_TIMEOUT = 15


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
                print(f"[geocode] rate limited, waiting 2s before next candidate")
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
        "limit": 2,
    }
    resp = requests.post(endpoint, json=payload, timeout=_STAC_TIMEOUT)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    if len(features) < 2:
        raise LookupError("STAC returned fewer than two usable observations for this area.")
    return features[0], features[-1]


def _visual_asset(item: dict[str, Any]) -> dict[str, Any] | None:
    assets = item.get("assets", {})
    for key in ("visual", "rendered_preview", "thumbnail", "overview"):
        asset = assets.get(key)
        if asset and asset.get("href"):
            return {"key": key, "href": asset["href"], "title": asset.get("title", key)}
    return None


def create_live_search(query: str) -> dict[str, Any]:
    location = geocode(query)
    first, latest = _stac_items(location["bbox"])

    first_asset = _visual_asset(first)
    latest_asset = _visual_asset(latest)

    result: dict[str, Any] = {
        "status": "TARGET_READY",
        "query": query,
        "target_label": location["label"],
        "coordinates": location["center"],
        "bbox": location["bbox"],
        "imagery": {
            "t1_datetime": first.get("properties", {}).get("datetime"),
            "t2_datetime": latest.get("properties", {}).get("datetime"),
            "t1_asset_key": first_asset["key"] if first_asset else None,
            "t2_asset_key": latest_asset["key"] if latest_asset else None,
        },
    }

    if first_asset:
        result["imagery"]["t1_url"] = first_asset["href"]
    if latest_asset:
        result["imagery"]["t2_url"] = latest_asset["href"]

    return result


def get_live_pair(asset_id: str) -> dict[str, Any]:
    pair = LIVE_CACHE.get(asset_id)
    if not pair:
        raise LookupError("This live-search imagery pair is unavailable. Run the search again.")
    return pair
