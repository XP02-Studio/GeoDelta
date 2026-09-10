import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

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

try:
    # Load a fast embedding model for real-time text-to-vector conversion
    # "all-MiniLM-L6-v2" is standard for fast, high-quality semantic search
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"[-] Failed to load SentenceTransformer: {e}")
    embedder = None

# ---------------------------------------------------------
# 2. Pydantic Models for Data Validation
# ---------------------------------------------------------
class SearchRequest(BaseModel):
    query_text: str
    top_k: int = 5
    # You can add optional fields here (e.g., location, date_range) if you want to use Qdrant filters

# ---------------------------------------------------------
# 3. Real-Time Semantic Search Endpoint
# ---------------------------------------------------------
@router.post("/search")
async def perform_semantic_search(request: SearchRequest):
    """
    Takes a natural language query, converts it to a vector, 
    and returns the top_k most similar satellite image records from Qdrant.
    """
    if not qdrant or not embedder:
        raise HTTPException(status_code=500, detail="Vector search engine is offline.")

    try:
        # A. Encode the user's text (e.g., "new building construction") into a vector
        query_vector = embedder.encode(request.query_text).tolist()

        # B. Query Qdrant for nearest neighbors in real-time
        search_results = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=request.top_k,
            # query_filter=qdrant_models.Filter(...) # Add payload filters here if you want hybrid search
        )

        # C. Process hits and extract the payload (metadata)
        results = []
        for hit in search_results:
            results.append({
                "id": hit.id,
                "similarity_score": round(hit.score, 4),
                "metadata": hit.payload # Contains coordinates, image paths, timestamps, etc.
            })

        return {
            "status": "success",
            "query": request.query_text,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search pipeline failed: {str(e)}")