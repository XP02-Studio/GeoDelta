from __future__ import annotations

import os
import json
import uuid
from datetime import datetime
from typing import Any, Sequence

import asyncpg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models

from .semantic_encoder import RSCLIPTextEncoder, VECTOR_SIZE


COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "geodelta_semantic")
SCORE_THRESHOLD = float(os.getenv("SEARCH_SCORE_THRESHOLD", "0.85"))


class EmbeddingUpsert(BaseModel):
    patch_id: str = Field(min_length=1)
    timestamp: datetime
    vector: list[float] = Field(min_length=VECTOR_SIZE, max_length=VECTOR_SIZE)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)


class SpatialUpsert(BaseModel):
    patch_id: str = Field(min_length=1)
    timestamp: datetime
    geometry: dict[str, Any]


class SearchEngine:
    def __init__(self, encoder: RSCLIPTextEncoder | None = None) -> None:
        self.qdrant = AsyncQdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333")
        )
        self.postgres_url = os.getenv(
            "POSTGRES_URL",
            "postgresql://geodelta:geodelta@localhost:5432/geodelta",
        )
        self.encoder = encoder
        self._pool: asyncpg.Pool | None = None

    async def startup(self) -> None:
        self._pool = await asyncpg.create_pool(self.postgres_url, min_size=1, max_size=10)
        exists = await self.qdrant.collection_exists(COLLECTION_NAME)
        if not exists:
            await self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
        await self.qdrant.close()

    async def upsert(self, items: Sequence[EmbeddingUpsert]) -> int:
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"geodelta:{item.patch_id}")),
                vector=item.vector,
                payload={"patch_id": item.patch_id, "timestamp": item.timestamp.isoformat()},
            )
            for item in items
        ]
        await self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        return len(points)

    async def upsert_geometry(self, item: SpatialUpsert) -> None:
        assert self._pool is not None
        await self._pool.execute(
            """
            INSERT INTO detected_changes (patch_id, timestamp, geometry)
            VALUES ($1, $2, ST_SetSRID(ST_GeomFromGeoJSON($3), 4326))
            ON CONFLICT (patch_id) DO UPDATE SET
                timestamp = EXCLUDED.timestamp,
                geometry = EXCLUDED.geometry
            """,
            item.patch_id,
            item.timestamp,
            json.dumps(item.geometry),
        )

    async def search(self, request: SearchRequest) -> list[dict[str, Any]]:
        if self.encoder is None:
            self.encoder = RSCLIPTextEncoder()
        vector = self.encoder.encode(request.query)
        hits = await self.qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=request.limit,
            score_threshold=SCORE_THRESHOLD,
            with_payload=True,
        )
        patch_ids = [str(hit.payload["patch_id"]) for hit in hits if hit.payload]
        if not patch_ids:
            return []
        assert self._pool is not None
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
        return [
            {
                **by_id[patch_id],
                "score": hit.score,
            }
            for hit in hits
            if (patch_id := str(hit.payload["patch_id"])) in by_id and hit.payload
        ]


app = FastAPI(title="GeoDelta Semantic Search", version="1.0.0")
engine = SearchEngine()


@app.on_event("startup")
async def startup() -> None:
    await engine.startup()


@app.on_event("shutdown")
async def shutdown() -> None:
    await engine.close()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/embeddings/upsert")
async def upsert_embeddings(items: list[EmbeddingUpsert]) -> dict[str, int]:
    try:
        return {"upserted": await engine.upsert(items)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/search")
async def search(request: SearchRequest) -> dict[str, Any]:
    try:
        return {"query": request.query, "results": await engine.search(request)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


    @app.post("/geometry/upsert")
    async def upsert_geometry(item: SpatialUpsert) -> dict[str, str]:
        try:
            await engine.upsert_geometry(item)
            return {"patch_id": item.patch_id, "status": "stored"}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc