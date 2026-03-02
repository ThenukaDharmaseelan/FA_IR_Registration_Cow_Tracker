#!/usr/bin/env python3
"""
Retinal image registration using CoWTracker point tracking.

Pipeline:
  1. Track points on enriched vessel maps (cross-modal safe)
  2. Extract confident correspondences from the flow field
  3. Fit a HOMOGRAPHY from those correspondences (RANSAC robust fitting)
  4. Warp moving image and vessel mask with the homography
  5. Generate a horizontal 7-panel grid per pair with labels and Dice scores

Usage:
    python register_grid.py --csv pairs.csv --output_dir results
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from cowtracker import CoWTracker

# ── Config ───────────────────────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
INF_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
TARGET_H  = 224   # must be multiple of 14
TARGET_W  = 224   # must be multiple of 14

# ── I/O ─────────────────────────────────────────────────────────────────
def load_image(path, h, w):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

def load_vessel(path, h, w):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.float32)

# ── Vessel enrichment ─────────────────────────────────────────────────
def enrich_vessel(vessel):
    binary  = (vessel * 255).astype(np.uint8)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    dist = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)
    dist = (dist / dist.max() * 255).astype(np.uint8) if dist.max() > 0 else np.zeros_like(binary)
    blurred = cv2.GaussianBlur(dilated, (11, 11), 3)
    return np.stack([dilated, dist, blurred], axis=-1)

# ── CoWTracker ───────────────────────────────────────────────────────
def run_cowtracker(model, fixed_vessel, moving_vessel):
    video = np.stack([enrich_vessel(fixed_vessel),
                      enrich_vessel(moving_vessel)], axis=0)
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float().to(DEVICE)
    torch.cuda.empty_cache()
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=INF_DTYPE):
            predictions = model.forward(video=video_tensor, queries=None)
    tracks = predictions["track"][0].cpu()
    vis    = predictions["vis"][0].cpu()
    conf   = predictions["conf"][0].cpu()
    return tracks, vis, conf

# ── Homography estimation ──────────────────────────────────────────────
def estimate_homography(tracks, vis, conf, conf_thresh=0.3, min_points=20):
    confidence = vis[1] * conf[1]
    mask       = confidence > conf_thresh
    if mask.sum() < min_points:
        conf_thresh = float(np.quantile(confidence[confidence>0], 0.5) if confidence.max()>0 else 0)
        mask = confidence > conf_thresh
        if mask.sum() < 8:
            print(f"  ERROR: only {mask.sum()} points — cannot fit homography")
            return None
    src = tracks[0][mask].numpy()
    dst = tracks[1][mask].numpy()
    H, inlier_mask = cv2.findHomography(
        dst.astype(np.float32),
        src.astype(np.float32),
        cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.999,
    )
    return H

# ── Warping ───────────────────────────────────────────────────────────
def warp_with_homography(img, H, h, w, flags=cv2.INTER_LINEAR):
    return cv2.warpPerspective(img, H, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

# ── Metrics ───────────────────────────────────────────────────────────
def dice_score(a, b):
    inter = (a * b).sum()
    return float((2 * inter) / (a.sum() + b.sum() + 1e-8))

# ── FOV mask ──────────────────────────────────────────────────────────
def get_fov_mask(img):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return (mask > 0).astype(np.float32)

# ── Overlap composites ─────────────────────────────────────────────────
def make_overlap_image(fixed_img, warped_img):
    fov = get_fov_mask(fixed_img)
    def norm(img):
        g = img.astype(np.float32).mean(axis=-1)
        mn, mx = g.min(), g.max()
        return (g - mn) / (mx - mn + 1e-8)
    f = norm(fixed_img)
    w = norm(warped_img)
    out = np.zeros((*f.shape, 3), dtype=np.float32)
    out[..., 0] = f
    out[..., 1] = w
    out[..., 2] = f
    out *= fov[..., np.newaxis]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)

def make_overlap_vessels(fixed_v, warped_v, fixed_img):
    fov = get_fov_mask(fixed_img)
    H, W = fixed_v.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[..., 0] = (warped_v * 255).astype(np.uint8)
    out[..., 1] = (fixed_v * 255).astype(np.uint8)
    out *= fov[..., np.newaxis]
    return out

# ── Visualisation helpers ─────────────────────────────────────────────
def mask_to_rgb(mask):
    g = (mask * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)

def add_label(img, text):
    out = img.copy()
    for color, thick in [((255, 255, 255), 2), ((0, 0, 0), 1)]:
        cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thick, cv2.LINE_AA)
    return out

# ── Build horizontal grid ─────────────────────────────────────────────
def make_row_grid(fixed_img, fixed_vessel, moving_img, moving_vessel,
                  warped_img, warped_vessel, dice_before=None, dice_after=None):
    GAP = 6
    H = fixed_img.shape[0]
    gap_v = np.ones((H, GAP, 3), dtype=np.uint8) * 60

    ov_img = make_overlap_image(fixed_img, warped_img)
    ov_vessel = make_overlap_vessels(fixed_vessel, warped_vessel, fixed_img)

    panels = [
        add_label(fixed_img, "Fixed Image"),
        add_label(mask_to_rgb(fixed_vessel), "Fixed Vessel"),
        add_label(moving_img, "Moving Image"),
        add_label(mask_to_rgb(moving_vessel), "Moving Vessel"),
        add_label(warped_img, "Registered"),
        add_label(ov_img, "Fixed+Reg Overlap"),
        add_label(ov_vessel, "Vessel Overlap")
    ]

    row_parts = []
    for idx, panel in enumerate(panels):
        row_parts.append(panel)
        if idx < len(panels) - 1:
            row_parts.append(gap_v)

    row = np.concatenate(row_parts, axis=1)

    # Add Dice score sidebar
    if dice_before is not None and dice_after is not None:
        SIDEBAR_W = 120
        sidebar = np.zeros((H, SIDEBAR_W, 3), dtype=np.uint8)
        sidebar[:] = (30, 30, 30)
        cv2.putText(sidebar, f"Before: {dice_before:.4f}", (6, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(sidebar, f"After : {dice_after:.4f}", (6, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        row = np.concatenate([sidebar, row], axis=1)

    return row

# ── Save outputs ──────────────────────────────────────────────────────
def save_outputs(out_dir, name, fixed_img, fixed_vessel, moving_img, moving_vessel,
                 warped_img, warped_vessel, dice_before=None, dice_after=None):
    grid = make_row_grid(fixed_img, fixed_vessel, moving_img, moving_vessel,
                         warped_img, warped_vessel, dice_before, dice_after)
    cv2.imwrite(str(out_dir / f"{name}_grid.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CoWTracker Retinal Registration Grid")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output_dir", default="dino_newresults")
    parser.add_argument("--moving_col", default="moving")
    parser.add_argument("--fixed_col", default="fixed")
    parser.add_argument("--moving_vessel_col", default="moving_vessel_mask")
    parser.add_argument("--fixed_vessel_col", default="fixed_vessel_mask")
    parser.add_argument("--height", type=int, default=TARGET_H)
    parser.add_argument("--width", type=int, default=TARGET_W)
    parser.add_argument("--conf_thresh", type=float, default=0.3)
    parser.add_argument("--min_points", type=int, default=20)
    args = parser.parse_args()

    assert args.height % 14 == 0 and args.width % 14 == 0, "Height/Width must be multiple of 14"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CoWTracker model...")
    model = CoWTracker.from_checkpoint(device=DEVICE, dtype=INF_DTYPE)
    print("Model ready.")

    df = pd.read_csv(args.csv)
    print(f"Found {len(df)} image pairs.")

    for i, row in df.iterrows():
        moving_path = row[args.moving_col]
        fixed_path  = row[args.fixed_col]
        mv_vessel_path = row[args.moving_vessel_col]
        fx_vessel_path = row[args.fixed_vessel_col]

        name = f"{i:04d}_{Path(moving_path).stem}_to_{Path(fixed_path).stem}"
        print(f"[{i+1}/{len(df)}] {name}")

        try:
            fixed_img     = load_image(fixed_path, args.height, args.width)
            moving_img    = load_image(moving_path, args.height, args.width)
            fixed_vessel  = load_vessel(fx_vessel_path, args.height, args.width)
            moving_vessel = load_vessel(mv_vessel_path, args.height, args.width)

            tracks, vis, conf = run_cowtracker(model, fixed_vessel, moving_vessel)
            H = estimate_homography(tracks, vis, conf, args.conf_thresh, args.min_points)

            if H is None:
                print(f"  Skipping {name}, homography failed")
                continue

            warped_img = warp_with_homography(fixed_img, H, args.height, args.width)
            warped_vessel = warp_with_homography((moving_vessel*255).astype(np.uint8), H, args.height, args.width, flags=cv2.INTER_NEAREST)
            warped_vessel = (warped_vessel > 127).astype(np.float32)

            fov = get_fov_mask(fixed_img)
            dice_before = dice_score(moving_vessel*fov, fixed_vessel*fov)
            dice_after  = dice_score(warped_vessel*fov, fixed_vessel*fov)
            print(f"  Dice before: {dice_before:.4f}, after: {dice_after:.4f}")

            save_outputs(out_dir, name, fixed_img, fixed_vessel, moving_img, moving_vessel,
                         warped_img, warped_vessel, dice_before, dice_after)

        except Exception as e:
            print(f"  ERROR processing {name}: {e}")

if __name__ == "__main__":
    main()