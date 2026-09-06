import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

os.makedirs("data/raw", exist_ok=True)

def create_dummy_geotiff(filename, offset=0):
    width, height = 1024, 1024
    bands = 3
    
    # Generate random raster data
    data = np.random.randint(50, 200, (bands, height, width), dtype=np.uint8) + offset

    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=bands,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=from_origin(77.1025, 28.7041, 0.0001, 0.0001),
    ) as dataset:
        dataset.write(data)

    print(f"Created dummy GeoTIFF: {filename}")

create_dummy_geotiff("data/raw/t1_baseline.tif", offset=0)
create_dummy_geotiff("data/raw/t2_current.tif", offset=15)