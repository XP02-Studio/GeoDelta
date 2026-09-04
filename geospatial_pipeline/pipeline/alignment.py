import cv2
import numpy as np

class AutoAlignment:
    @staticmethod
    def align_images(t1_norm: np.ndarray, t2_norm: np.ndarray) -> np.ndarray:
        """
        Calculates SIFT tie-points and warps T2 onto T1's exact geometry
        to guarantee sub-pixel alignment for smooth UI swipe transitions.
        """
        # Convert 3-channel or single channel for feature extraction
        img1 = np.moveaxis(t1_norm[:3], 0, -1) if t1_norm.ndim == 3 else t1_norm
        img2 = np.moveaxis(t2_norm[:3], 0, -1) if t2_norm.ndim == 3 else t2_norm

        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.ndim == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.ndim == 3 else img2

        # Extract SIFT keypoints
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            raise ValueError("SIFT feature extraction failed. Ensure clear image contrast.")

        # Match features
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

        # Extract top 100 feature points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)

        # Compute Affine matrix to prevent non-rigid warping distortions
        affine_matrix, _ = cv2.estimateAffine2D(dst_pts, src_pts)

        h, w = gray1.shape
        if t2_norm.ndim == 3:
            t2_aligned = np.zeros_like(t2_norm)
            for b in range(t2_norm.shape[0]):
                t2_aligned[b] = cv2.warpAffine(t2_norm[b], affine_matrix, (w, h))
        else:
            t2_aligned = cv2.warpAffine(t2_norm, affine_matrix, (w, h))

        return t2_aligned