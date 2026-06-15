import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from cowtracker import CoWTracker

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
INF_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
TARGET_H  = 224
TARGET_W  = 224

# ── Hard-reject thresholds (absolute — never auto-computed) ──────────────────
REJECT_ABS_ROT        = 90.0
REJECT_SCALE_MAX      = 3.0
REJECT_SCALE_MIN      = 0.2
REJECT_NEGATIVE_SCALE = True


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(img_rgb, clip_limit=3.0, tile_grid=(8, 8),
                     blur_ksize=5, blur_sigma=1.2):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq  = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)
    ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    return cv2.GaussianBlur(enhanced, (ksize, ksize), blur_sigma)


# ── I/O ───────────────────────────────────────────────────────────────────────
def load_image(path, h, w):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                      (w, h), interpolation=cv2.INTER_LINEAR)


def load_vessel(path, h, w):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.float32)


# ── Vessel enrichment ─────────────────────────────────────────────────────────
def enrich_vessel(vessel):
    binary  = (vessel * 255).astype(np.uint8)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    dist    = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)
    dist    = (dist / dist.max() * 255).astype(np.uint8) if dist.max() > 0 \
              else np.zeros_like(binary)
    blurred = cv2.GaussianBlur(dilated, (11, 11), 3)
    return np.stack([dilated, dist, blurred], axis=-1)


# ── CoWTracker ────────────────────────────────────────────────────────────────
def run_cowtracker(model, fixed_vessel, moving_vessel):
    video = np.stack([enrich_vessel(fixed_vessel),
                      enrich_vessel(moving_vessel)], axis=0)
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float().to(DEVICE)
    torch.cuda.empty_cache()
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=INF_DTYPE):
            predictions = model.forward(video=video_tensor, queries=None)
    return (predictions["track"][0].cpu(),
            predictions["vis"][0].cpu(),
            predictions["conf"][0].cpu())


# ── Affine estimation ─────────────────────────────────────────────────────────
def estimate_affine(tracks, vis, conf, conf_thresh=0.3,
                    min_points=6, min_inlier_ratio=0.05):
    confidence        = vis[1] * conf[1]
    mask              = confidence > conf_thresh
    low_conf_fallback = False

    if mask.sum() < min_points:
        median_conf = float(confidence[confidence > 0].quantile(0.5)
                            if confidence.max() > 0 else 0)
        mask = confidence > median_conf
        low_conf_fallback = True
        if mask.sum() < 4:
            return None, None, None, "too_few_points"

    src, dst     = (tracks[0][mask].numpy().astype(np.float32),
                    tracks[1][mask].numpy().astype(np.float32))
    n_candidates = len(src)

    M, inlier_mask = cv2.estimateAffine2D(
        dst, src, method=cv2.RANSAC,
        ransacReprojThreshold=3.0, maxIters=2000,
        confidence=0.999, refineIters=10)

    if M is None:
        return None, None, None, "cv2_failed"

    n_inliers    = int(inlier_mask.sum()) if inlier_mask is not None else 0
    inlier_ratio = n_inliers / max(n_candidates, 1)

    if n_candidates > 1000 and inlier_ratio < min_inlier_ratio:
        return None, None, None, \
            f"low_inlier_ratio {inlier_ratio:.3f}"

    decomp = decompose_affine(M)
    reject = check_degenerate(decomp)
    if reject:
        return None, None, None, reject

    meta = dict(n_candidates=n_candidates, n_inliers=n_inliers,
                inlier_ratio=round(inlier_ratio, 4),
                low_conf_fallback=low_conf_fallback)
    return M, decomp, meta, None


# ── Degenerate fit detector ───────────────────────────────────────────────────
def check_degenerate(decomp):
    ang = abs(decomp["angle_deg"])
    if ang > REJECT_ABS_ROT:
        return f"extreme rotation {ang:.1f}°"
    sx, sy = decomp["sx"], decomp["sy"]
    if REJECT_NEGATIVE_SCALE and sy < 0:
        return f"negative scale sy={sy:.4f}"
    for axis, s in [("sx", sx), ("sy", sy)]:
        if abs(s) > REJECT_SCALE_MAX:
            return f"extreme scale {axis}={s:.4f}"
        if abs(s) < REJECT_SCALE_MIN:
            return f"near-zero scale {axis}={s:.4f}"
    return None


# ── Affine decomposition ──────────────────────────────────────────────────────
def decompose_affine(M):
    tx, ty = float(M[0, 2]), float(M[1, 2])
    A      = M[:2, :2].astype(np.float64)
    U, S, Vt = np.linalg.svd(A)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        U[:, -1] *= -1
        S[-1]    *= -1
    R         = U @ Vt
    angle_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    sx, sy    = float(S[0]), float(S[1])
    shear     = float((R.T @ A)[0, 1] / (sx + 1e-8))
    return dict(tx=tx, ty=ty, angle_deg=angle_deg, sx=sx, sy=sy, shear=shear)


# ═════════════════════════════════════════════════════════════════════════════
# NEW — Auto-threshold computation from the empirical distribution
# ═════════════════════════════════════════════════════════════════════════════

def compute_auto_thresholds(decomps, z=2.0, percentile=95.0):
    """
    Compute soft-flagging thresholds automatically from the distribution of
    affine parameters across all successfully estimated pairs.

    Two methods are computed and the MORE CONSERVATIVE (lower) threshold is
    chosen for each parameter so the flags are meaningful but not over-sensitive.

    Method A — Mean ± z * std  (parametric, assumes roughly Gaussian)
    Method B — percentile-based  (non-parametric, robust to outliers)

    Parameters
    ----------
    decomps    : list of decomp dicts from decompose_affine()
    z          : number of standard deviations for method A  (default 2.0 → ~95%)
    percentile : upper percentile for method B               (default 95.0)

    Returns
    -------
    thresholds : dict with keys:
        max_translation, max_rotation, scale_lo, scale_hi, max_shear
    stats      : dict — raw statistics for logging / CSV
    """
    if len(decomps) < 5:
        print("  ⚠ Too few pairs to compute auto-thresholds reliably "
              f"(n={len(decomps)}, need ≥5). "
              "Falling back to conservative defaults.")
        return _default_thresholds(), {}

    # ── Collect per-parameter arrays ─────────────────────────────────────────
    # Translation — use max(|tx|, |ty|) per pair as a single magnitude
    trans = np.array([max(abs(d["tx"]), abs(d["ty"])) for d in decomps])
    rots  = np.array([abs(d["angle_deg"])              for d in decomps])
    sx    = np.array([d["sx"]                           for d in decomps])
    sy    = np.array([d["sy"]                           for d in decomps])
    shear = np.array([abs(d["shear"])                   for d in decomps])

    # Scale: combine sx and sy into one array for a unified bound
    scales = np.concatenate([sx, sy])

    def _threshold(arr, z, pct):
        """
        Return the more conservative of:
          - mean + z * std   (parametric upper bound)
          - percentile value (non-parametric upper bound)
        i.e. whichever is LOWER — flags more aggressively.
        """
        mu, sigma = arr.mean(), arr.std()
        t_param   = mu + z * sigma
        t_pct     = np.percentile(arr, pct)
        chosen    = min(t_param, t_pct)  # conservative = lower threshold
        return round(float(chosen), 4), round(float(t_param), 4), round(float(t_pct), 4)

    # ── Per-parameter thresholds ──────────────────────────────────────────────
    max_trans,  tp_trans,  tq_trans  = _threshold(trans,  z, percentile)
    max_rot,    tp_rot,    tq_rot    = _threshold(rots,   z, percentile)
    max_shear_, tp_shear,  tq_shear  = _threshold(shear,  z, percentile)

    # Scale bounds — lower bound from left tail, upper bound from right tail
    scale_lo_param = scales.mean() - z * scales.std()
    scale_lo_pct   = np.percentile(scales, 100 - percentile)   # e.g. 5th pct
    scale_lo       = round(float(max(scale_lo_param, scale_lo_pct)), 4)

    scale_hi_param = scales.mean() + z * scales.std()
    scale_hi_pct   = np.percentile(scales, percentile)          # e.g. 95th pct
    scale_hi       = round(float(min(scale_hi_param, scale_hi_pct)), 4)

    thresholds = dict(
        max_translation = max_trans,
        max_rotation    = max_rot,
        scale_lo        = max(scale_lo, REJECT_SCALE_MIN + 0.01),  # never below hard-reject
        scale_hi        = min(scale_hi, REJECT_SCALE_MAX - 0.01),  # never above hard-reject
        max_shear       = max_shear_,
    )

    stats = dict(
        n_pairs           = len(decomps),
        z_used            = z,
        percentile_used   = percentile,

        trans_mean        = round(float(trans.mean()),  4),
        trans_std         = round(float(trans.std()),   4),
        trans_p95         = round(float(tq_trans),      4),
        trans_z_thresh    = round(float(tp_trans),      4),
        trans_chosen      = max_trans,

        rot_mean          = round(float(rots.mean()),   4),
        rot_std           = round(float(rots.std()),    4),
        rot_p95           = round(float(tq_rot),        4),
        rot_z_thresh      = round(float(tp_rot),        4),
        rot_chosen        = max_rot,

        scale_mean        = round(float(scales.mean()), 4),
        scale_std         = round(float(scales.std()),  4),
        scale_lo_chosen   = thresholds["scale_lo"],
        scale_hi_chosen   = thresholds["scale_hi"],

        shear_mean        = round(float(shear.mean()),  4),
        shear_std         = round(float(shear.std()),   4),
        shear_p95         = round(float(tq_shear),      4),
        shear_z_thresh    = round(float(tp_shear),      4),
        shear_chosen      = max_shear_,
    )

    return thresholds, stats


def _default_thresholds():
    """Fallback thresholds if auto-computation is not possible."""
    return dict(max_translation=80.0, max_rotation=30.0,
                scale_lo=0.4, scale_hi=1.6, max_shear=0.25)


def print_auto_thresholds(thresholds, stats):
    print("\n" + "─" * 50)
    print("  Auto-computed soft-flag thresholds")
    print("─" * 50)
    if not stats:
        print("  (defaults used — too few pairs)")
    else:
        print(f"  Based on {stats['n_pairs']} pairs  "
              f"(z={stats['z_used']}, p{stats['percentile_used']:.0f})")
        print(f"  Translation : mean={stats['trans_mean']:.2f}  "
              f"std={stats['trans_std']:.2f}  "
              f"→ threshold={thresholds['max_translation']:.2f}px")
        print(f"  Rotation    : mean={stats['rot_mean']:.2f}°  "
              f"std={stats['rot_std']:.2f}°  "
              f"→ threshold={thresholds['max_rotation']:.2f}°")
        print(f"  Scale       : mean={stats['scale_mean']:.4f}  "
              f"std={stats['scale_std']:.4f}  "
              f"→ [{thresholds['scale_lo']:.4f}, {thresholds['scale_hi']:.4f}]")
        print(f"  Shear       : mean={stats['shear_mean']:.4f}  "
              f"std={stats['shear_std']:.4f}  "
              f"→ threshold={thresholds['max_shear']:.4f}")
    print("─" * 50 + "\n")


# ── Soft flagging (uses auto thresholds) ─────────────────────────────────────
def flag_affine(decomp, thresholds):
    """
    Soft flagging using automatically computed thresholds.
    thresholds dict comes from compute_auto_thresholds().
    """
    flags = []
    tx, ty = abs(decomp["tx"]), abs(decomp["ty"])
    if tx > thresholds["max_translation"] or ty > thresholds["max_translation"]:
        flags.append(
            f"large translation ({tx:.1f},{ty:.1f})px "
            f"> {thresholds['max_translation']:.1f}px")

    ang = abs(decomp["angle_deg"])
    if ang > thresholds["max_rotation"]:
        flags.append(
            f"large rotation {ang:.2f}° "
            f"> {thresholds['max_rotation']:.2f}°")

    for axis, s in [("sx", decomp["sx"]), ("sy", decomp["sy"])]:
        lo, hi = thresholds["scale_lo"], thresholds["scale_hi"]
        if not (lo <= s <= hi):
            flags.append(f"unusual scale {axis}={s:.4f} outside [{lo:.4f},{hi:.4f}]")

    if abs(decomp["shear"]) > thresholds["max_shear"]:
        flags.append(
            f"high shear {decomp['shear']:.4f} "
            f"> {thresholds['max_shear']:.4f}")
    return flags


# ── Warping / metrics / FOV / overlaps ───────────────────────────────────────
def warp_with_affine(img, M, h, w, flags=cv2.INTER_LINEAR):
    return cv2.warpAffine(img, M, (w, h), flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def dice_score(a, b):
    return float((2 * (a * b).sum()) / (a.sum() + b.sum() + 1e-8))

def get_fov_mask(img):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return (mask > 0).astype(np.float32)

def make_overlap_image(fixed_img, warped_img):
    fov = get_fov_mask(fixed_img)
    def norm(img):
        g = img.astype(np.float32).mean(axis=-1)
        return (g - g.min()) / (g.max() - g.min() + 1e-8)
    f, w = norm(fixed_img), norm(warped_img)
    out  = np.zeros((*f.shape, 3), dtype=np.float32)
    out[..., 0] = f; out[..., 1] = w; out[..., 2] = f
    return (np.clip(out * fov[..., np.newaxis], 0, 1) * 255).astype(np.uint8)

def make_overlap_vessels(fixed_v, warped_v, fixed_img):
    fov = get_fov_mask(fixed_img)
    H, W = fixed_v.shape
    out  = np.zeros((H, W, 3), dtype=np.uint8)
    out[..., 0] = (warped_v * 255).astype(np.uint8)
    out[..., 1] = (fixed_v  * 255).astype(np.uint8)
    return (out * fov[..., np.newaxis]).astype(np.uint8)

def mask_to_rgb(mask):
    g = (mask * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)

def add_label(img, text):
    out = img.copy()
    for color, thick in [((255,255,255), 2), ((0,0,0), 1)]:
        cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, thick, cv2.LINE_AA)
    return out


# ── Grid / sidebar ────────────────────────────────────────────────────────────
def make_row_grid(fixed_img, fixed_vessel,
                  moving_img, moving_vessel,
                  warped_img, warped_vessel,
                  row_label=None, dice_before=None, dice_after=None,
                  decomp=None, flags=None, meta=None, thresholds=None):

    GAP   = 6
    H     = fixed_img.shape[0]
    gap_v = np.ones((H, GAP, 3), dtype=np.uint8) * 60

    panels = [
        add_label(fixed_img,                  "Fixed Image"),
        add_label(mask_to_rgb(fixed_vessel),  "Fixed Vessel"),
        add_label(moving_img,                 "Moving Image"),
        add_label(mask_to_rgb(moving_vessel), "Moving Vessel"),
        add_label(warped_img,                 "Registered"),
        add_label(make_overlap_image(fixed_img, warped_img),              "Fixed+Reg Overlap"),
        add_label(make_overlap_vessels(fixed_vessel, warped_vessel, fixed_img), "Vessel Overlap"),
    ]
    row_parts = []
    for idx, p in enumerate(panels):
        row_parts.append(p)
        if idx < len(panels) - 1:
            row_parts.append(gap_v)
    row = np.concatenate(row_parts, axis=1)

    SIDEBAR_W = 200
    sidebar   = np.full((H, SIDEBAR_W, 3), (30, 30, 30), dtype=np.uint8)
    lines     = []

    if row_label   is not None: lines.append(row_label)
    if dice_before is not None: lines.append(f"Before:{dice_before:.4f}")
    if dice_after  is not None:
        d = dice_after - (dice_before or 0)
        lines.append(f"After: {dice_after:.4f}")
        lines.append(f"Delta:{'+' if d>=0 else ''}{d:.4f}")
    if decomp is not None:
        lines += ["---",
                  f"tx:{decomp['tx']:+.1f} ty:{decomp['ty']:+.1f}",
                  f"rot:{decomp['angle_deg']:+.2f}deg",
                  f"sx:{decomp['sx']:.3f} sy:{decomp['sy']:.3f}",
                  f"shear:{decomp['shear']:.3f}"]
    if thresholds is not None:
        lines += ["AutoThresh:",
                  f" tr<{thresholds['max_translation']:.1f}px",
                  f" rot<{thresholds['max_rotation']:.1f}deg",
                  f" s:[{thresholds['scale_lo']:.2f},{thresholds['scale_hi']:.2f}]",
                  f" sh<{thresholds['max_shear']:.3f}"]
    if meta is not None:
        lines.append(f"inlr:{meta['n_inliers']}/{meta['n_candidates']}")
        if meta.get("low_conf_fallback"): lines.append("!low_conf_fallback")
    if flags:
        lines.append("WARN:")
        for f in flags: lines.append(f"  {f[:26]}")

    flag_color = (80, 80, 255) if flags else (200, 200, 200)
    for li, line in enumerate(lines):
        y = 18 + li * 16
        if y > H - 4: break
        color = flag_color if (line.startswith("WARN") or line.startswith("  ")) \
                else ((60, 200, 255) if line.startswith("!") else (200, 200, 200))
        cv2.putText(sidebar, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, color, 1, cv2.LINE_AA)

    row = np.concatenate(
        [sidebar, np.ones((H, GAP, 3), dtype=np.uint8) * 60, row], axis=1)
    return row


# ── Save per-pair ─────────────────────────────────────────────────────────────
def save_outputs(out_dir, name,
                 fixed_img, fixed_vessel, moving_img, moving_vessel,
                 warped_img, warped_vessel,
                 dice_before=None, dice_after=None, pair_index=None,
                 decomp=None, flags=None, meta=None, thresholds=None):

    def write_rgb(fname, img):
        cv2.imwrite(str(out_dir / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    def write_mask(fname, m):
        cv2.imwrite(str(out_dir / fname), (m * 255).astype(np.uint8))

    write_rgb( f"{name}_registered.png",       warped_img)
    write_mask(f"{name}_registered_vessel.png", warped_vessel)
    write_rgb( f"{name}_overlap_image.png",    make_overlap_image(fixed_img, warped_img))
    write_rgb( f"{name}_overlap_vessels.png",
               make_overlap_vessels(fixed_vessel, warped_vessel, fixed_img))

    row = make_row_grid(fixed_img, fixed_vessel, moving_img, moving_vessel,
                        warped_img, warped_vessel,
                        row_label=f"#{pair_index}" if pair_index else None,
                        dice_before=dice_before, dice_after=dice_after,
                        decomp=decomp, flags=flags, meta=meta,
                        thresholds=thresholds)
    write_rgb(f"{name}_grid.png", row)
    return row


# ── Summary pages ─────────────────────────────────────────────────────────────
def save_summary_page(out_dir, rows, mean_before=None, mean_after=None,
                      rows_per_page=20):
    if not rows: return
    pages_dir = out_dir / "summary_pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    max_w = max(r.shape[1] for r in rows)

    def pad(r):
        if r.shape[1] < max_w:
            r = np.concatenate(
                [r, np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)],
                axis=1)
        return r

    padded  = [pad(r) for r in rows]
    sep     = np.full((4, max_w, 3), (100, 100, 100), dtype=np.uint8)
    n_pages = max(1, (len(padded) + rows_per_page - 1) // rows_per_page)

    for p in range(n_pages):
        chunk  = padded[p * rows_per_page : (p + 1) * rows_per_page]
        label  = f"{p+1:02d}of{n_pages:02d}"
        banner = np.zeros((50, max_w, 3), dtype=np.uint8)
        txt    = (f"CoWTracker Registration  |  Page {label}  |  "
                  f"pairs {p*rows_per_page+1}-"
                  f"{min((p+1)*rows_per_page, len(rows))}/{len(rows)}")
        if mean_before and mean_after:
            txt += (f"  |  Dice before:{mean_before:.4f}  "
                    f"after:{mean_after:.4f}  "
                    f"delta:{mean_after-mean_before:+.4f}")
        cv2.putText(banner, txt, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 60), 1, cv2.LINE_AA)
        page = np.concatenate(
            [banner] + [x for r in chunk for x in [sep, r]], axis=0)
        cv2.imwrite(str(pages_dir / f"summary_page_{label}.png"),
                    cv2.cvtColor(page, cv2.COLOR_RGB2BGR))

    print(f"  {n_pages} summary page(s) → {pages_dir.resolve()}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="CoWTracker Registration — Auto Soft-Flag Thresholds")
    parser.add_argument("--csv",               required=True)
    parser.add_argument("--output_dir",        default="Results_affine")
    parser.add_argument("--moving_col",        default="moving")
    parser.add_argument("--fixed_col",         default="fixed")
    parser.add_argument("--moving_vessel_col", default="moving_vessel_mask")
    parser.add_argument("--fixed_vessel_col",  default="fixed_vessel_mask")
    parser.add_argument("--height",      type=int,   default=TARGET_H)
    parser.add_argument("--width",       type=int,   default=TARGET_W)
    parser.add_argument("--conf_thresh", type=float, default=0.3)
    parser.add_argument("--min_points",  type=int,   default=6)
    parser.add_argument("--min_inlier_ratio", type=float, default=0.05)
    parser.add_argument("--preprocess",       action="store_true", default=True)
    parser.add_argument("--no_preprocess",    dest="preprocess", action="store_false")
    parser.add_argument("--clahe_clip",  type=float, default=8.0)
    parser.add_argument("--clahe_tile",  type=int,   default=4)
    parser.add_argument("--blur_ksize",  type=int,   default=3)
    parser.add_argument("--blur_sigma",  type=float, default=0.5)

    # ── Auto-threshold parameters ─────────────────────────────────────────────
    parser.add_argument(
        "--thresh_z", type=float, default=2.0,
        help="Std dev multiplier for parametric threshold  "
             "(mean + z*std). Default=2.0 → ~95%% of normal distribution.")
    parser.add_argument(
        "--thresh_percentile", type=float, default=95.0,
        help="Percentile for non-parametric threshold. Default=95.0")

    args = parser.parse_args()

    assert args.height % 14 == 0
    assert args.width  % 14 == 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CoWTracker Retinal Registration  [Auto Soft-Flag Thresholds]")
    print("=" * 60)
    print(f"Device          : {DEVICE}")
    print(f"Threshold method: mean ± {args.thresh_z}σ  vs  "
          f"p{args.thresh_percentile:.0f}  → conservative (lower) chosen")
    print(f"Output          : {out_dir.resolve()}\n")

    print("Loading CoWTracker model...")
    model = CoWTracker.from_checkpoint(device=DEVICE, dtype=INF_DTYPE)
    print("Model ready.\n")

    df = pd.read_csv(args.csv)
    print(f"Found {len(df)} image pairs.\n")

    # ════════════════════════════════════════════════════════════════════════
    # PASS 1 — Estimate affines for ALL pairs, collect decomp values
    # No flagging yet — just gather the raw parameter distribution
    # ════════════════════════════════════════════════════════════════════════
    print("─" * 60)
    print("PASS 1 — Estimating affines (no flagging yet)...")
    print("─" * 60)

    pass1_cache   = {}   # index → (fixed_img, moving_img, fixed_vessel,
                         #           moving_vessel, M, decomp, meta, reject)
    decomps_valid = []   # only successfully estimated decomps

    for i, row in df.iterrows():
        moving_path    = row[args.moving_col]
        fixed_path     = row[args.fixed_col]
        mv_vessel_path = row[args.moving_vessel_col]
        fx_vessel_path = row[args.fixed_vessel_col]

        print(f"  [{i+1}/{len(df)}] {Path(moving_path).name} → "
              f"{Path(fixed_path).name}", end="  ")

        try:
            fixed_img     = load_image(fixed_path,      args.height, args.width)
            moving_img    = load_image(moving_path,      args.height, args.width)
            fixed_vessel  = load_vessel(fx_vessel_path,  args.height, args.width)
            moving_vessel = load_vessel(mv_vessel_path,  args.height, args.width)

            if args.preprocess:
                kw = dict(clip_limit=args.clahe_clip,
                          tile_grid=(args.clahe_tile, args.clahe_tile),
                          blur_ksize=args.blur_ksize, blur_sigma=args.blur_sigma)
                fixed_img  = preprocess_image(fixed_img,  **kw)
                moving_img = preprocess_image(moving_img, **kw)

            tracks, vis, conf = run_cowtracker(model, fixed_vessel, moving_vessel)

            M, decomp, meta, reject = estimate_affine(
                tracks, vis, conf,
                conf_thresh=args.conf_thresh,
                min_points=args.min_points,
                min_inlier_ratio=args.min_inlier_ratio,
            )

            pass1_cache[i] = dict(
                fixed_img=fixed_img, moving_img=moving_img,
                fixed_vessel=fixed_vessel, moving_vessel=moving_vessel,
                moving_path=moving_path, fixed_path=fixed_path,
                M=M, decomp=decomp, meta=meta, reject=reject,
            )

            if decomp is not None:
                decomps_valid.append(decomp)
                print(f"OK  rot={decomp['angle_deg']:+.1f}°  "
                      f"tx={decomp['tx']:+.1f}  ty={decomp['ty']:+.1f}")
            else:
                print(f"REJECTED ({reject})")

        except Exception as e:
            print(f"ERROR: {e}")
            pass1_cache[i] = dict(
                moving_path=moving_path, fixed_path=fixed_path,
                M=None, decomp=None, meta=None, reject=str(e),
                fixed_img=None, moving_img=None,
                fixed_vessel=None, moving_vessel=None,
            )

    # ════════════════════════════════════════════════════════════════════════
    # Compute thresholds from the collected distribution
    # ════════════════════════════════════════════════════════════════════════
    print(f"\nComputing thresholds from {len(decomps_valid)} valid pairs...")
    thresholds, thresh_stats = compute_auto_thresholds(
        decomps_valid,
        z=args.thresh_z,
        percentile=args.thresh_percentile,
    )
    print_auto_thresholds(thresholds, thresh_stats)

    # Save threshold stats to CSV for reproducibility
    if thresh_stats:
        pd.DataFrame([{**thresholds, **thresh_stats}]).to_csv(
            out_dir / "auto_thresholds.csv", index=False)
        print(f"  Thresholds saved → {out_dir}/auto_thresholds.csv")

    # ════════════════════════════════════════════════════════════════════════
    # PASS 2 — Apply auto thresholds, warp, evaluate, visualise
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("PASS 2 — Applying auto thresholds, warping, evaluating...")
    print("─" * 60)

    results  = []
    all_rows = []

    for i, row in df.iterrows():
        cache = pass1_cache.get(i, {})
        moving_path = cache.get("moving_path", "")
        fixed_path  = cache.get("fixed_path",  "")
        name        = f"{i:04d}_{Path(str(moving_path)).stem}_to_" \
                      f"{Path(str(fixed_path)).stem}"

        print(f"[{i+1}/{len(df)}] {Path(str(moving_path)).name} → "
              f"{Path(str(fixed_path)).name}")

        M      = cache.get("M")
        decomp = cache.get("decomp")
        meta   = cache.get("meta")
        reject = cache.get("reject")

        if M is None:
            print(f"  Skipping — {reject}")
            results.append(dict(name=name, moving=moving_path,
                                fixed=fixed_path,
                                status="degenerate", error=reject))
            continue

        fixed_img     = cache["fixed_img"]
        moving_img    = cache["moving_img"]
        fixed_vessel  = cache["fixed_vessel"]
        moving_vessel = cache["moving_vessel"]

        # ── Auto-threshold soft flagging ──────────────────────────────────
        flags = flag_affine(decomp, thresholds)
        if flags:
            print(f"  ⚠ Flags: {', '.join(flags)}")
        else:
            print(f"  ✓ No flags")

        h, w = args.height, args.width
        warped_img = cv2.cvtColor(
            warp_with_affine(cv2.cvtColor(moving_img, cv2.COLOR_RGB2BGR),
                             M, h, w),
            cv2.COLOR_BGR2RGB)
        warped_vessel = (warp_with_affine(
            (moving_vessel * 255).astype(np.uint8), M, h, w,
            flags=cv2.INTER_NEAREST) > 127).astype(np.float32)

        fov         = get_fov_mask(fixed_img)
        dice_before = dice_score(moving_vessel * fov, fixed_vessel * fov)
        dice_after  = dice_score(warped_vessel  * fov, fixed_vessel * fov)
        delta       = dice_after - dice_before
        print(f"  Dice before:{dice_before:.4f}  after:{dice_after:.4f}  "
              f"{'▲' if delta>=0 else '▼'}{abs(delta):.4f}"
              f"{'  ⚠ REGRESSED' if delta < 0 else ''}")

        row_grid = save_outputs(
            out_dir, name,
            fixed_img, fixed_vessel, moving_img, moving_vessel,
            warped_img, warped_vessel,
            dice_before=dice_before, dice_after=dice_after,
            pair_index=i + 1, decomp=decomp, flags=flags, meta=meta,
            thresholds=thresholds,
        )
        all_rows.append(row_grid)

        results.append(dict(
            name=name, moving=moving_path, fixed=fixed_path,
            dice_before=round(dice_before, 4),
            dice_after=round(dice_after,   4),
            dice_delta=round(delta,         4),
            dice_regressed=delta < 0,
            tx=round(decomp["tx"],        2),
            ty=round(decomp["ty"],        2),
            angle_deg=round(decomp["angle_deg"], 4),
            sx=round(decomp["sx"],        4),
            sy=round(decomp["sy"],        4),
            shear=round(decomp["shear"],     4),
            n_candidates=meta["n_candidates"],
            n_inliers=meta["n_inliers"],
            inlier_ratio=meta["inlier_ratio"],
            low_conf_fallback=meta["low_conf_fallback"],
            flagged=bool(flags),
            flag_reasons="; ".join(flags) if flags else "",
            # also log the thresholds used so results are reproducible
            thresh_max_translation=thresholds["max_translation"],
            thresh_max_rotation=thresholds["max_rotation"],
            thresh_scale_lo=thresholds["scale_lo"],
            thresh_scale_hi=thresholds["scale_hi"],
            thresh_max_shear=thresholds["max_shear"],
            status="ok",
        ))

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "results.csv", index=False)

    ok = results_df[results_df["status"] == "ok"]
    if all_rows:
        save_summary_page(
            out_dir, all_rows,
            mean_before=ok["dice_before"].mean() if len(ok) else None,
            mean_after=ok["dice_after"].mean()   if len(ok) else None)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total pairs    : {len(df)}")
    print(f"  OK             : {len(ok)}")
    print(f"  Degenerate     : "
          f"{len(results_df[results_df['status']=='degenerate'])}")
    if len(ok):
        flagged   = ok[ok["flagged"] == True]
        regressed = ok[ok["dice_regressed"] == True]
        print(f"  Flagged (soft) : {len(flagged)}")
        print(f"  Regressed      : {len(regressed)}")
        print(f"  Mean Dice before : {ok['dice_before'].mean():.4f}")
        print(f"  Mean Dice after  : {ok['dice_after'].mean():.4f}")
        print(f"  Mean Dice delta  : {ok['dice_delta'].mean():+.4f}")
    print(f"\n  Auto thresholds used:")
    print(f"    translation  < {thresholds['max_translation']:.2f}px")
    print(f"    rotation     < {thresholds['max_rotation']:.2f}°")
    print(f"    scale in       [{thresholds['scale_lo']:.4f}, "
          f"{thresholds['scale_hi']:.4f}]")
    print(f"    shear        < {thresholds['max_shear']:.4f}")
    print(f"\n  Results  → {out_dir.resolve()}/results.csv")
    print(f"  Thresholds → {out_dir.resolve()}/auto_thresholds.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
