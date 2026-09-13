"""
Seed script: populates Qdrant + PostGIS with searchable geographic sectors.
Run once after databases are created.

Usage:
    cd backend && python -m seed_data
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncpg
from qdrant_client import AsyncQdrantClient, models

# Re-use the same settings the backend uses
from app.core.config import settings
from app.services.clip_adapter import encode, VECTOR_SIZE

# --------------------------------------------------------------------------- #
#  Sample sectors covering strategic Indian border / infrastructure regions
# --------------------------------------------------------------------------- #
SECTORS = [
    {
        "patch_id": "SECTOR-NEPAL-01",
        "label": "New airstrip near Nepal border",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[85.31, 27.70], [85.34, 27.70], [85.34, 27.73], [85.31, 27.73], [85.31, 27.70]]],
        },
        "lat": 27.715,
        "lng": 85.325,
    },
    {
        "patch_id": "SECTOR-KASHMIR-01",
        "label": "Military bunker in forested valley",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[74.80, 34.05], [74.83, 34.05], [74.83, 34.08], [74.80, 34.08], [74.80, 34.05]]],
        },
        "lat": 34.065,
        "lng": 74.815,
    },
    {
        "patch_id": "SECTOR-RJ-01",
        "label": "New highway construction near Rajasthan border",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[70.50, 24.50], [70.54, 24.50], [70.54, 24.54], [70.50, 24.54], [70.50, 24.50]]],
        },
        "lat": 24.52,
        "lng": 70.52,
    },
    {
        "patch_id": "SECTOR-ARUNACHAL-01",
        "label": "Concrete hangar and aircraft shelter",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[97.50, 27.90], [97.54, 27.90], [97.54, 27.94], [97.50, 27.94], [97.50, 27.90]]],
        },
        "lat": 27.92,
        "lng": 97.52,
    },
    {
        "patch_id": "SECTOR-LADAKH-01",
        "label": "Radar installation on hilltop",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[78.50, 34.20], [78.54, 34.20], [78.54, 34.24], [78.50, 34.24], [78.50, 34.20]]],
        },
        "lat": 34.22,
        "lng": 78.52,
    },
    {
        "patch_id": "SECTOR-PUNJAB-01",
        "label": "Logistics depot with vehicle convoy",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[74.50, 32.10], [74.54, 32.10], [74.54, 32.14], [74.50, 32.14], [74.50, 32.10]]],
        },
        "lat": 32.12,
        "lng": 74.52,
    },
    {
        "patch_id": "SECTOR-GUJARAT-01",
        "label": "Coastal fence and perimeter wall",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[68.80, 23.60], [68.84, 23.60], [68.84, 23.64], [68.80, 23.64], [68.80, 23.60]]],
        },
        "lat": 23.62,
        "lng": 68.82,
    },
    {
        "patch_id": "SECTOR-SIKKIM-01",
        "label": "Missile silo excavation site",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[88.50, 27.30], [88.54, 27.30], [88.54, 27.34], [88.50, 27.34], [88.50, 27.30]]],
        },
        "lat": 27.32,
        "lng": 88.52,
    },
    {
        "patch_id": "SECTOR-ASSAM-01",
        "label": "Forest clearing and trench excavation",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[92.70, 26.10], [92.74, 26.10], [92.74, 26.14], [92.70, 26.14], [92.70, 26.10]]],
        },
        "lat": 26.12,
        "lng": 92.72,
    },
    {
        "patch_id": "SECTOR-UTTARAKHAND-01",
        "label": "Bridge extension over river valley",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[79.50, 30.30], [79.54, 30.30], [79.54, 30.34], [79.50, 30.34], [79.50, 30.30]]],
        },
        "lat": 30.32,
        "lng": 79.52,
    },
]


async def seed():
    print("[seed] Connecting to Qdrant ...")
    qdrant = AsyncQdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

    collection = settings.QDRANT_COLLECTION
    exists = await qdrant.collection_exists(collection)
    if not exists:
        await qdrant.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        )
        print(f"[seed] Created collection '{collection}'")

    now = datetime.now(timezone.utc).isoformat()

    # --- 1. Upsert vectors into Qdrant ---
    points = []
    for sector in SECTORS:
        vector = encode(sector["label"])
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"geodelta:{sector['patch_id']}"))
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={"patch_id": sector["patch_id"], "timestamp": now},
            )
        )

    await qdrant.upsert(collection_name=collection, points=points)
    print(f"[seed] Upserted {len(points)} vectors into Qdrant")

    # --- 2. Upsert geometries into PostGIS ---
    print("[seed] Connecting to PostGIS ...")
    pool = await asyncpg.create_pool(settings.POSTGRES_URL, min_size=1, max_size=5)

    for sector in SECTORS:
        await pool.execute(
            """
            INSERT INTO detected_changes (patch_id, timestamp, geometry)
            VALUES ($1, $2, ST_SetSRID(ST_GeomFromGeoJSON($3), 4326))
            ON CONFLICT (patch_id) DO UPDATE SET
                timestamp = EXCLUDED.timestamp,
                geometry   = EXCLUDED.geometry
            """,
            sector["patch_id"],
            now,
            json.dumps(sector["geometry"]),
        )
    print(f"[seed] Upserted {len(SECTORS)} geometries into PostGIS")

    await pool.close()
    await qdrant.close()

    print(f"\n[seed] Done!  {len(SECTORS)} sectors are now searchable.")
    print("[seed] Try searching for: 'airstrip', 'bunker', 'missile silo', 'bridge', 'convoy'")


if __name__ == "__main__":
    asyncio.run(seed())
