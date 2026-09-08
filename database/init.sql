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