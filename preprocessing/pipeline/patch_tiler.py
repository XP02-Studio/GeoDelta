import os
import json
import numpy as np
import config

class PatchTiler:
    @staticmethod
    def create_ml_patches(t1_norm: np.ndarray, t2_aligned: np.ndarray, geotransform: tuple):
        """
        Slices T1 and T2 arrays into 512x512 patches with a stride overlap,
        saving binary .npz tensors and bounding box JSON metadata.
        """
        _, h, w = t1_norm.shape if t1_norm.ndim == 3 else (1, *t1_norm.shape)
        patch_size = config.PATCH_SIZE
        stride = config.STRIDE
        gt = geotransform
        patch_id = 0

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                t1_patch = t1_norm[:, y:y+patch_size, x:x+patch_size]
                t2_patch = t2_aligned[:, y:y+patch_size, x:x+patch_size]

                # Save binary .npz tensor for Binit (PyTorch)
                npz_filename = os.path.join(config.PATCHES_DIR, f"patch_{patch_id:04d}.npz")
                np.savez_compressed(npz_filename, t1=t1_patch, t2=t2_patch)

                # Compute GPS Bounding Box (minX, minY, maxX, maxY) for Adarsh (Vector DB)
                minX = gt[0] + x * gt[1] + y * gt[2]
                maxY = gt[3] + x * gt[4] + y * gt[5]
                maxX = gt[0] + (x + patch_size) * gt[1] + (y + patch_size) * gt[2]
                minY = gt[3] + (x + patch_size) * gt[4] + (y + patch_size) * gt[5]

                meta = {
                    "patch_id": patch_id,
                    "bbox_gps": [minX, minY, maxX, maxY],
                    "pixel_coords": [x, y, patch_size, patch_size],
                    "npz_file": f"patch_{patch_id:04d}.npz"
                }

                # Save JSON metadata
                json_filename = os.path.join(config.PATCHES_DIR, f"patch_{patch_id:04d}.json")
                with open(json_filename, "w") as f:
                    json.dump(meta, f, indent=4)

                patch_id += 1

        print(f"Generated {patch_id} ML patches and metadata files in {config.PATCHES_DIR}")