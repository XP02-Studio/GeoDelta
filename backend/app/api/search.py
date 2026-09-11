import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from dotenv import load_dotenv

from app.services.semantic_encoder import RSCLIPTextEncoder

# Load environment variables (.env)
load_dotenv()

# Initialize FastAPI Router
router = APIRouter()

# ---------------------------------------------------------
# 1. Configuration & Client Initialization
# ---------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "geospatial_metadata") # Change if your collection name differs

try:
    # Connect to Qdrant Cloud
    qdrant = QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY
    )
except Exception as e:
    print(f"[-] Failed to connect to Qdrant: {e}")
    qdrant = None

# Initialize the local encoder (512-D RS-CLIP)
encoder = RSCLIPTextEncoder()


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
    if not qdrant:
        raise HTTPException(status_code=500, detail="Vector search engine is offline.")

    try:
        # 1. Generate the 512-D vector locally! No internet required.
        query_vector = encoder.encode(request.query_text)
        
        # 2. Query Qdrant for nearest neighbors in real-time
        search_results = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=request.top_k,
        )

        # 3. Process hits and extract the payload (metadata)
        results = []
        for hit in search_results:
            results.append({
                "id": hit.id,
                "similarity_score": round(hit.score, 4),
                "metadata": hit.payload
            })

        return {
            "status": "success",
            "query": request.query_text,
            "vector_length": len(query_vector),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search pipeline failed: {str(e)}")