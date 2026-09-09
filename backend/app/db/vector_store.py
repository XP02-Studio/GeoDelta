from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Sequence

import asyncpg
from qdrant_client import AsyncQdrantClient, models

from app.core.config import settings
from app.services.semantic_encoder import RSCLIPTextEncoder, VECTOR_SIZE


class VectorSearchEngine:
    def __init__(self, encoder: RSCLIPTextEncoder | None = None) -> None:
        self.qdrant = AsyncQdrantClient(url=settings.QDRANT_URL)
        self.postgres_url = settings.POSTGRES_URL
        self.collection_name = settings.QDRANT_COLLECTION
        self.score_threshold = settings.SEARCH_SCORE_THRESHOLD
        self.encoder = encoder or RSCLIPTextEncoder()
        self._pool: asyncpg.Pool | None = None

    async def startup(self) -> None:
        """Initializes PostgreSQL connection pool and Qdrant 512-D collection."""
        self._pool = await asyncpg.create_pool(self.postgres_url, min_size=1, max_size=10)
        
        exists = await self.qdrant.collection_exists(self.collection_name)
        if not exists:
            await self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
            )

    async def close(self) -> None:
        """Gracefully closes DB connection pools."""
        if self._pool is not None:
            await self._pool.close()
        await self.qdrant.close()

    async def upsert_embeddings(self, patch_id: str, timestamp: datetime, vector: list[float]) -> None:
        """Upserts a single 512-D embedding into Qdrant."""
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"geodelta:{patch_id}"))
        point = models.PointStruct(
            id=point_id,
            vector=vector,
            payload={"patch_id": patch_id, "timestamp": timestamp.isoformat()},
        )
        await self.qdrant.upsert(collection_name=self.collection_name, points=[point])

    async def upsert_geometry(self, patch_id: str, timestamp: datetime, geometry: dict[str, Any]) -> None:
        """Stores or updates spatial polygon geometries in PostGIS."""
        if self._pool is None:
            raise RuntimeError("Database pool not initialized.")

        await self._pool.execute(
            """
            INSERT INTO detected_changes (patch_id, timestamp, geometry)
            VALUES ($1, $2, ST_SetSRID(ST_GeomFromGeoJSON($3), 4326))
            ON CONFLICT (patch_id) DO UPDATE SET
                timestamp = EXCLUDED.timestamp,
                geometry = EXCLUDED.geometry
            """,
            patch_id,
            timestamp,
            json.dumps(geometry),
        )

    async def search_similar_targets(self, query_text: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Generates 512-D text embedding via RS-CLIP, searches Qdrant, 
        and joins matching target records with PostGIS geometry[cite: 1].
        """
        vector = self.encoder.encode(query_text)
        hits = await self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=limit,
            score_threshold=self.score_threshold,
            with_payload=True,
        )

        patch_ids = [str(hit.payload["patch_id"]) for hit in hits if hit.payload and "patch_id" in hit.payload]
        if not patch_ids:
            return []

        if self._pool is None:
            raise RuntimeError("Database pool not initialized.")

        rows = await self._pool.fetch(
            """
            SELECT patch_id, timestamp,
                   ST_AsGeoJSON(geometry)::json AS geometry,
                   ST_Y(ST_Centroid(geometry)) AS lat,
                   ST_X(ST_Centroid(geometry)) AS lng
            FROM detected_changes
            WHERE patch_id = ANY($1::text[])
            """,
            patch_ids,
        )

        by_id = {row["patch_id"]: dict(row) for row in rows}
        
        results = []
        for hit in hits:
            if hit.payload and (patch_id := str(hit.payload.get("patch_id"))) in by_id:
                record = by_id[patch_id]
                results.append({
                    **record,
                    "score": hit.score,
                })

        return results


# Global Instance for application dependency injection
vector_engine = VectorSearchEngine()