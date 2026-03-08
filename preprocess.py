"""
preprocess.py — shared image preprocessing for retinal registration scripts.

Usage:
    from preprocess import preprocess_image

    prep = preprocess_image(img_rgb)          # defaults
    prep = preprocess_image(img_rgb,
                            clip_limit=3.0,
                            tile_grid=(8, 8),
                            blur_ksize=5,
                            blur_sigma=1.2)
"""

import cv2
import numpy as np


def preprocess_image(img_rgb: np.ndarray,
                     clip_limit: float = 3.0,
                     tile_grid: tuple  = (8, 8),
                     blur_ksize: int   = 5,
                     blur_sigma: float = 1.2) -> np.ndarray:
    """
    Preprocess a retinal RGB image:
      1. CLAHE on the L channel (LAB colour space) — enhances local contrast
         without blowing out bright regions.
      2. Gaussian blur — reduces noise before keypoint / feature extraction.

    Parameters
    ----------
    img_rgb    : H×W×3 uint8 RGB image
    clip_limit : CLAHE clip limit (higher = stronger contrast boost)
    tile_grid  : CLAHE tile grid size (rows, cols)
    blur_ksize : Gaussian kernel size (must be odd)
    blur_sigma : Gaussian sigma

    Returns
    -------
    Preprocessed H×W×3 uint8 RGB image
    """
    # ── CLAHE on L channel ────────────────────────────────────────────────────
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq  = clahe.apply(l)

    lab_eq   = cv2.merge([l_eq, a, b])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # ── Gaussian blur ─────────────────────────────────────────────────────────
    ksize   = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1  # must be odd
    blurred = cv2.GaussianBlur(enhanced, (ksize, ksize), blur_sigma)

    return blurred