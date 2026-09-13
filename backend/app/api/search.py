from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio

from app.services.live_search import create_live_search

# Initialize FastAPI Router
router = APIRouter()


# ---------------------------------------------------------
# 2. Pydantic Models for Data Validation
# ---------------------------------------------------------
class SearchRequest(BaseModel):
    query_text: str


# ---------------------------------------------------------
# 3. Real-Time Semantic Search Endpoint
# ---------------------------------------------------------
@router.post("/api/v1/search/target")
@router.post("/search")
async def search_endpoint(request: SearchRequest):
    """
    Takes a natural language query, converts it to a vector, 
    and returns the top_k most similar satellite image records from Qdrant.
    """
    try:
        query = request.query_text.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query_text must not be empty")

        return await asyncio.to_thread(create_live_search, query)

    except HTTPException:
        raise
    except (RuntimeError, LookupError) as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search pipeline failed: {str(e)}")
