import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Tactical Imagery Analysis"
    API_V1_STR: str = "/api/v1"

    # Fetch Database & Vector Store URLs directly from Environment variables (.env)
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY", None)
    QDRANT_COLLECTION: str = "geodelta_semantic"
    POSTGRES_URL: str = os.getenv("DATABASE_URL", "postgresql://geodelta:geodelta@localhost:5432/geodelta")

    # ML Inference & Threshold Settings
    SEARCH_SCORE_THRESHOLD: float = 0.85
    RSCLIP_ENCODER: str = ""

    # Pipeline Data Paths
    RAW_DATA_DIR: str = "./data/raw"
    PROCESSED_DATA_DIR: str = "./data/processed"
    BASEMAP_TILES_DIR: str = "./data/tiles"
    PATCHES_DIR: str = "./data/patches"

    # Tiling & Normalization Parameters
    PATCH_SIZE: int = 512
    STRIDE: int = 410  # ~20% overlap to prevent edge-boundary data loss
    PERCENTILE_LOW: int = 2
    PERCENTILE_HIGH: int = 98

    def initialize_directories(self) -> None:
        """Ensures all required pipeline data directories exist on system startup."""
        for path_str in [
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.BASEMAP_TILES_DIR,
            self.PATCHES_DIR,
        ]:
            Path(path_str).mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Instantiate settings instance
settings = Settings()

# Automatically trigger directory creation
settings.initialize_directories()