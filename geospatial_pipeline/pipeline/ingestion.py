import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import config

class DataIngestion:
    @staticmethod
    def reproject_and_normalize(t1_path: str, t2_path: str):
        """
        Reads T1 and T2, reprojects T2 to T1's grid via gdalwarp, 
        and applies percentile min-max normalization.
        """
        # Ensure target processing directory exists
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

        with rasterio.open(t1_path) as t1_ds, rasterio.open(t2_path) as t2_ds:
            t1_arr = t1_ds.read().astype(np.float32)
            t1_crs = t1_ds.crs
            t1_transform = t1_ds.transform
            t1_gt = t1_transform.to_gdal()
            t1_proj = t1_crs.to_wkt() if t1_crs else ""

            # Warp T2 to match T1 CRS, resolution, and dimensions.
            warped_t2_path = os.path.join(config.PROCESSED_DATA_DIR, "t2_warped.tif")
            t2_arr = np.zeros_like(t1_arr, dtype=np.float32)
            reproject(
                source=t2_ds.read().astype(np.float32),
                destination=t2_arr,
                src_transform=t2_ds.transform,
                src_crs=t2_ds.crs,
                dst_transform=t1_transform,
                dst_crs=t1_crs,
                resampling=Resampling.bilinear,
            )

            with rasterio.open(
                warped_t2_path,
                "w",
                driver="GTiff",
                width=t1_ds.width,
                height=t1_ds.height,
                count=t2_arr.shape[0],
                dtype=t2_arr.dtype,
                crs=t1_crs,
                transform=t1_transform,
            ) as warped_t2:
                warped_t2.write(t2_arr)

        # Radiometric Percentile Normalization
        def normalize_array(arr):
            p_low, p_high = np.percentile(arr, (config.PERCENTILE_LOW, config.PERCENTILE_HIGH))
            arr_clipped = np.clip(arr, p_low, p_high)
            norm = (arr_clipped - p_low) / (p_high - p_low + 1e-8) * 255.0
            return norm.astype(np.uint8)

        t1_norm = normalize_array(t1_arr)
        t2_norm = normalize_array(t2_arr)

        return t1_norm, t2_norm, t1_gt, t1_proj, warped_t2_path