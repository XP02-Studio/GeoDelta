import sys
import os

# Add the backend directory to the Python path so that internal imports 
# like `from app.api import ...` work correctly when uvicorn is run from the root.
from backend.app.api.search import router as search_router
app.include_router(search_router, prefix="/api/v1")

backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Import the FastAPI app instance so `uvicorn main:app` works on Render
from backend.main import app
