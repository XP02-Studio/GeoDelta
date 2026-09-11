import os
import sys

# Add the backend directory to sys.path so 'from app.api import ...' works
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Import the FastAPI app instance from backend/main.py
from backend.main import app