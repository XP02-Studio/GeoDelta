import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from app.core.config import settings

# Initialize connection pool using POSTGRES_URL from config
try:
    db_pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=20,
        dsn=settings.POSTGRES_URL
    )
except Exception as e:
    print(f"[!] PostgreSQL Pool Initialization Error: {e}")
    db_pool = None


@contextmanager
def get_db_cursor():
    """
    Context manager to safely acquire and release database connections from the pool.
    Usage:
        with get_db_cursor() as cursor:
            cursor.execute("SELECT * FROM detected_changes")
            results = cursor.fetchall()
    """
    if not db_pool:
        raise RuntimeError("PostgreSQL connection pool is not initialized.")

    conn = db_pool.getconn()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        db_pool.putconn(conn)


def init_db():
    """
    Ensures PostGIS extension and detected_changes table/indexes exist.
    """
    init_sql = """
    CREATE EXTENSION IF NOT EXISTS postgis;

    CREATE TABLE IF NOT EXISTS detected_changes (
        patch_id TEXT PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL,
        geometry GEOMETRY(POLYGON, 4326) NOT NULL
    );

    CREATE INDEX IF NOT EXISTS detected_changes_geometry_gist
        ON detected_changes USING GIST (geometry);

    CREATE INDEX IF NOT EXISTS detected_changes_timestamp_idx
        ON detected_changes (timestamp);
    """
    with get_db_cursor() as cursor:
        cursor.execute(init_sql)
        print("[OK] PostGIS extension and detected_changes schema verified.")


def insert_detected_change(patch_id: str, timestamp_str: str, wkt_polygon: str):
    """
    Inserts a detected spatial change into the database.
    wkt_polygon: Well-Known Text geometry (e.g., 'POLYGON((85.31 27.70, 85.34 27.70, ...))')
    """
    query = """
    INSERT INTO detected_changes (patch_id, timestamp, geometry)
    VALUES (%s, %s, ST_GeomFromText(%s, 4326))
    ON CONFLICT (patch_id) DO UPDATE 
    SET timestamp = EXCLUDED.timestamp,
        geometry = EXCLUDED.geometry;
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (patch_id, timestamp_str, wkt_polygon))


def get_changes_within_bbox(min_x: float, min_y: float, max_x: float, max_y: float):
    """
    Queries detected changes overlapping with a given bounding box [min_x, min_y, max_x, max_y].
    Returns geometry as GeoJSON.
    """
    query = """
    SELECT 
        patch_id, 
        timestamp, 
        ST_AsGeoJSON(geometry)::json AS geometry 
    FROM detected_changes
    WHERE ST_Intersects(
        geometry, 
        ST_MakeEnvelope(%s, %s, %s, %s, 4326)
    );
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (min_x, min_y, max_x, max_y))
        return cursor.fetchall()