import os
import numpy as np
from osgeo import gdal, osr

os.makedirs("data/raw", exist_ok=True)

def create_dummy_geotiff(filename, offset=0):
    width, height = 1024, 1024
    bands = 3
    
    # Generate random raster data
    data = np.random.randint(50, 200, (bands, height, width), dtype=np.uint8) + offset

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(filename, width, height, bands, gdal.GDT_Byte)

    # Set spatial reference (WGS84 / EPSG:4326)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())

    # Set geotransform: [top_left_x, w_e_pixel_resolution, rotation, top_left_y, rotation, n_s_pixel_resolution]
    ds.SetGeoTransform([77.1025, 0.0001, 0, 28.7041, 0, -0.0001])

    for i in range(bands):
        ds.GetRasterBand(i + 1).WriteArray(data[i])

    ds.FlushCache()
    ds = None
    print(f"Created dummy GeoTIFF: {filename}")

create_dummy_geotiff("data/raw/t1_baseline.tif", offset=0)
create_dummy_geotiff("data/raw/t2_current.tif", offset=15)