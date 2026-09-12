from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.db.vector_store import vector_engine

# Initialize FastAPI Router
router = APIRouter()


# ---------------------------------------------------------
# 2. Pydantic Models for Data Validation
# ---------------------------------------------------------
class SearchRequest(BaseModel):
    query_text: str
    top_k: int = 5


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

        # The shared engine returns Qdrant hits joined to their PostGIS geometry,
        # including the latitude and longitude the frontend needs to navigate.
        results = await vector_engine.search_similar_targets(query, limit=request.top_k)

        return {
            "status": "success",
            "query": query,
            "results": results
        }

    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search pipeline failed: {str(e)}")
