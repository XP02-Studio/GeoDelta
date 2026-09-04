import os
import numpy as np
from osgeo import gdal
import config

# Enable GDAL exception throwing explicitly to eliminate GDAL 4.0 warnings
gdal.UseExceptions()

class DataIngestion:
    @staticmethod
    def reproject_and_normalize(t1_path: str, t2_path: str):
        """
        Reads T1 and T2, reprojects T2 to T1's grid via gdalwarp, 
        and applies percentile min-max normalization.
        """
        # Ensure target processing directory exists
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

        t1_ds = gdal.Open(t1_path)
        if not t1_ds:
            raise FileNotFoundError(f"Could not open T1 image at {t1_path}")

        t1_proj = t1_ds.GetProjection()
        t1_gt = t1_ds.GetGeoTransform()
        
        # Warp T2 to match T1 CRS & Resolution
        warped_t2_path = os.path.join(config.PROCESSED_DATA_DIR, "t2_warped.tif")
        gdal.Warp(
            warped_t2_path, 
            t2_path, 
            dstSRS=t1_proj, 
            targetAlignedPixels=True,
            xRes=t1_gt[1], 
            yRes=abs(t1_gt[5])
        )

        t2_ds = gdal.Open(warped_t2_path)

        # Read as numpy arrays
        t1_arr = t1_ds.ReadAsArray().astype(np.float32)
        t2_arr = t2_ds.ReadAsArray().astype(np.float32)

        # Radiometric Percentile Normalization
        def normalize_array(arr):
            p_low, p_high = np.percentile(arr, (config.PERCENTILE_LOW, config.PERCENTILE_HIGH))
            arr_clipped = np.clip(arr, p_low, p_high)
            norm = (arr_clipped - p_low) / (p_high - p_low + 1e-8) * 255.0
            return norm.astype(np.uint8)

        t1_norm = normalize_array(t1_arr)
        t2_norm = normalize_array(t2_arr)

        return t1_norm, t2_norm, t1_gt, t1_proj, warped_t2_path