# # import argparse
# # from pathlib import Path

# # import cv2
# # import numpy as np
# # import pandas as pd
# # import torch

# # from cowtracker import CoWTracker

# # # ── Config ────────────────────────────────────────────────────────────────────
# # DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
# # INF_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
# # TARGET_H  = 224   # must be multiple of 14
# # TARGET_W  = 224   # must be multiple of 14

# # # ── Hard-reject thresholds ────────────────────────────────────────────────────
# # REJECT_ABS_ROT        = 90.0
# # REJECT_SCALE_MAX      = 3.0
# # REJECT_SCALE_MIN      = 0.2
# # REJECT_NEGATIVE_SCALE = True


# # # ── I/O ───────────────────────────────────────────────────────────────────────
# # def load_image(path, h, w):
# #     img = cv2.imread(str(path))
# #     if img is None:
# #         raise FileNotFoundError(f"Cannot read: {path}")
# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #     return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


# # def load_vessel(path, h, w):
# #     mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
# #     if mask is None:
# #         raise FileNotFoundError(f"Cannot read: {path}")
# #     mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
# #     return (mask > 127).astype(np.float32)


# # # ── Vessel enrichment ─────────────────────────────────────────────────────────
# # def enrich_vessel(vessel):
# #     binary  = (vessel * 255).astype(np.uint8)
# #     kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# #     dilated = cv2.dilate(binary, kernel, iterations=2)
# #     dist    = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)
# #     dist    = (dist / dist.max() * 255).astype(np.uint8) if dist.max() > 0 \
# #               else np.zeros_like(binary)
# #     blurred = cv2.GaussianBlur(dilated, (11, 11), 3)
# #     return np.stack([dilated, dist, blurred], axis=-1)


# # # ── CoWTracker (shared for both stages) ───────────────────────────────────────
# # def run_cowtracker(model, vessel_a, vessel_b):
# #     video = np.stack([enrich_vessel(vessel_a), enrich_vessel(vessel_b)], axis=0)
# #     video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float().to(DEVICE)
# #     torch.cuda.empty_cache()
# #     with torch.no_grad():
# #         with torch.amp.autocast(device_type="cuda", dtype=INF_DTYPE):
# #             predictions = model.forward(video=video_tensor, queries=None)
# #     tracks = predictions["track"][0].cpu()
# #     vis    = predictions["vis"][0].cpu()
# #     conf   = predictions["conf"][0].cpu()
# #     return tracks, vis, conf


# # # ── Stage 1: Affine estimation ────────────────────────────────────────────────
# # def estimate_affine(tracks, vis, conf, conf_thresh=0.3, min_points=6,
# #                     min_inlier_ratio=0.05):
# #     confidence        = vis[1] * conf[1]
# #     mask              = confidence > conf_thresh
# #     low_conf_fallback = False

# #     if mask.sum() < min_points:
# #         print(f"  Warning: only {int(mask.sum())} confident points "
# #               f"(need {min_points}), lowering threshold to median...")
# #         median_conf = float(confidence[confidence > 0].quantile(0.5)
# #                             if confidence.max() > 0 else 0)
# #         mask = confidence > median_conf
# #         low_conf_fallback = True
# #         if mask.sum() < 4:
# #             print(f"  ERROR: still only {int(mask.sum())} points")
# #             return None, None, None, "too_few_points"

# #     src          = tracks[0][mask].numpy().astype(np.float32)
# #     dst          = tracks[1][mask].numpy().astype(np.float32)
# #     n_candidates = len(src)

# #     print(f"  [S1] Fitting affine from {n_candidates} correspondences "
# #           f"(conf>{conf_thresh:.2f})"
# #           f"{' [fallback]' if low_conf_fallback else ''}")

# #     M, inlier_mask = cv2.estimateAffine2D(
# #         dst, src,
# #         method=cv2.RANSAC,
# #         ransacReprojThreshold=3.0,
# #         maxIters=2000,
# #         confidence=0.999,
# #         refineIters=10,
# #     )
# #     if M is None:
# #         print("  ERROR: affine estimation failed")
# #         return None, None, None, "cv2_failed"

# #     n_inliers    = int(inlier_mask.sum()) if inlier_mask is not None else 0
# #     inlier_ratio = n_inliers / max(n_candidates, 1)
# #     print(f"  [S1] RANSAC inliers: {n_inliers} / {n_candidates}  "
# #           f"(ratio={inlier_ratio:.3f})")

# #     if n_candidates > 1000 and inlier_ratio < min_inlier_ratio:
# #         reason = (f"low inlier ratio {inlier_ratio:.3f} < {min_inlier_ratio} "
# #                   f"({n_inliers}/{n_candidates})")
# #         print(f"  ✗ Hard-reject: {reason}")
# #         return None, None, None, reason

# #     decomp = decompose_affine(M)
# #     print(f"  [S1] tx={decomp['tx']:.2f}px  ty={decomp['ty']:.2f}px  "
# #           f"rot={decomp['angle_deg']:.2f}°  "
# #           f"sx={decomp['sx']:.4f}  sy={decomp['sy']:.4f}  "
# #           f"shear={decomp['shear']:.4f}")

# #     reject_reason = check_degenerate(decomp)
# #     if reject_reason:
# #         print(f"  ✗ Hard-reject (degenerate): {reject_reason}")
# #         return None, None, None, reject_reason

# #     meta = dict(
# #         n_candidates=n_candidates,
# #         n_inliers=n_inliers,
# #         inlier_ratio=round(inlier_ratio, 4),
# #         low_conf_fallback=low_conf_fallback,
# #     )
# #     return M, decomp, meta, None


# # # ── Affine decomposition (SVD-based) ─────────────────────────────────────────
# # def decompose_affine(M):
# #     tx, ty = float(M[0, 2]), float(M[1, 2])
# #     A      = M[:2, :2].astype(np.float64)
# #     U, S, Vt = np.linalg.svd(A)
# #     if np.linalg.det(U) * np.linalg.det(Vt) < 0:
# #         U[:, -1] *= -1
# #         S[-1]    *= -1
# #     R         = U @ Vt
# #     angle_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
# #     sx, sy    = float(S[0]), float(S[1])
# #     A_norm    = R.T @ A
# #     shear     = float(A_norm[0, 1] / (sx + 1e-8))
# #     return dict(tx=tx, ty=ty, angle_deg=angle_deg, sx=sx, sy=sy, shear=shear)


# # def check_degenerate(decomp):
# #     ang = abs(decomp["angle_deg"])
# #     if ang > REJECT_ABS_ROT:
# #         return f"extreme rotation {ang:.1f}° > {REJECT_ABS_ROT}°"
# #     sx, sy = decomp["sx"], decomp["sy"]
# #     if REJECT_NEGATIVE_SCALE and sy < 0:
# #         return f"negative scale sy={sy:.4f} (reflection)"
# #     for axis, s in [("sx", sx), ("sy", sy)]:
# #         if abs(s) > REJECT_SCALE_MAX:
# #             return f"extreme scale {axis}={s:.4f} > {REJECT_SCALE_MAX}"
# #         if abs(s) < REJECT_SCALE_MIN:
# #             return f"near-zero scale {axis}={s:.4f} < {REJECT_SCALE_MIN}"
# #     return None


# # def flag_affine(decomp,
# #                 max_translation=80.0,
# #                 max_rotation=30.0,
# #                 scale_range=(0.4, 1.6),
# #                 max_shear=0.25):
# #     flags = []
# #     tx, ty = abs(decomp["tx"]), abs(decomp["ty"])
# #     if tx > max_translation or ty > max_translation:
# #         flags.append(f"large translation ({tx:.1f}, {ty:.1f})px > {max_translation}px")
# #     ang = abs(decomp["angle_deg"])
# #     if ang > max_rotation:
# #         flags.append(f"large rotation {ang:.1f}° > {max_rotation}°")
# #     for axis, s in [("sx", decomp["sx"]), ("sy", decomp["sy"])]:
# #         lo, hi = scale_range
# #         if not (lo <= s <= hi):
# #             flags.append(f"unusual scale {axis}={s:.4f} outside [{lo},{hi}]")
# #     if abs(decomp["shear"]) > max_shear:
# #         flags.append(f"high shear {decomp['shear']:.4f} > {max_shear}")
# #     return flags


# # # ── Stage 1: Warping ──────────────────────────────────────────────────────────
# # def warp_with_affine(img, M, h, w, flags=cv2.INTER_LINEAR):
# #     return cv2.warpAffine(img, M, (w, h), flags=flags,
# #                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# # # ── Stage 2: MLS — Moving Least Squares (Schaefer et al. 2006) ───────────────
# # def select_mls_control_points(tracks, vis, conf, conf_thresh=0.3,
# #                                min_points=10, max_ctrl=500):
# #     """
# #     Select control points from Stage 2 CoWTracker correspondences.

# #     tracks[0] = points in fixed frame        (where to map to)
# #     tracks[1] = points in affine-warped frame (where to sample from)

# #     Returns fixed_pts, warped_pts, meta (or None, None, None if too few points).
# #     """
# #     confidence = vis[1] * conf[1]
# #     mask       = confidence > conf_thresh
# #     n_conf     = int(mask.sum())

# #     if n_conf < min_points:
# #         print(f"  [S2] Only {n_conf} confident points — "
# #               f"MLS skipped, keeping affine result")
# #         return None, None, None

# #     fixed_pts  = tracks[0][mask].numpy().astype(np.float32)
# #     warped_pts = tracks[1][mask].numpy().astype(np.float32)

# #     if len(fixed_pts) > max_ctrl:
# #         idx        = np.random.choice(len(fixed_pts), max_ctrl, replace=False)
# #         fixed_pts  = fixed_pts[idx]
# #         warped_pts = warped_pts[idx]

# #     n_ctrl = len(fixed_pts)
# #     meta   = dict(mls_n_control_pts=n_ctrl, mls_n_confident=n_conf)
# #     return fixed_pts, warped_pts, meta


# # def compute_mls_map(src_pts, dst_pts, h, w, alpha=1.0, chunk_size=5000):
# #     """
# #     Compute MLS similarity deformation map (Schaefer et al. 2006).

# #     Uses the complex-number formulation of the similarity warp.
# #     Processed in pixel chunks to keep memory bounded.

# #     src_pts : (N, 2) float32  control points in fixed frame
# #     dst_pts : (N, 2) float32  corresponding points in affine-warped frame
# #     alpha   : weight fall-off exponent (higher = more localised)

# #     Returns map_x, map_y  (H, W) float32 for cv2.remap.
# #     """
# #     yy, xx = np.mgrid[0:h, 0:w]
# #     grid   = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
# #     P      = len(grid)

# #     map_x = np.empty(P, dtype=np.float32)
# #     map_y = np.empty(P, dtype=np.float32)

# #     for start in range(0, P, chunk_size):
# #         end = min(start + chunk_size, P)
# #         v   = grid[start:end]                                    # (C, 2)

# #         diff  = v[:, np.newaxis, :] - src_pts[np.newaxis, :, :] # (C, N, 2)
# #         dist2 = np.maximum((diff ** 2).sum(-1), 1e-10)           # (C, N)
# #         wt    = 1.0 / (dist2 ** alpha)                           # (C, N)
# #         wt_s  = wt.sum(1, keepdims=True)                         # (C, 1)

# #         p_star = (wt @ src_pts) / wt_s   # (C, 2)
# #         q_star = (wt @ dst_pts) / wt_s   # (C, 2)

# #         ph = src_pts[np.newaxis] - p_star[:, np.newaxis]  # (C, N, 2)
# #         qh = dst_pts[np.newaxis] - q_star[:, np.newaxis]  # (C, N, 2)
# #         vc = v - p_star                                    # (C, 2)

# #         mu = (wt * (ph ** 2).sum(-1)).sum(1)               # (C,)

# #         # Complex similarity warp: f(v) = conj_sum * vc_c / mu + q*_c
# #         ph_c     = ph[:, :, 0] + 1j * ph[:, :, 1]          # (C, N)
# #         qh_c     = qh[:, :, 0] + 1j * qh[:, :, 1]          # (C, N)
# #         vc_c     = vc[:, 0]    + 1j * vc[:, 1]              # (C,)
# #         M_c      = (wt * np.conj(ph_c) * qh_c).sum(1)       # (C,)
# #         q_star_c = q_star[:, 0] + 1j * q_star[:, 1]         # (C,)
# #         f_c      = M_c * vc_c / np.maximum(mu, 1e-10) + q_star_c

# #         map_x[start:end] = f_c.real.astype(np.float32)
# #         map_y[start:end] = f_c.imag.astype(np.float32)

# #     return map_x.reshape(h, w), map_y.reshape(h, w)


# # def warp_with_mls(img, map_x, map_y, is_mask=False):
# #     interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
# #     return cv2.remap(img, map_x, map_y, interp,
# #                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# # # ── Metrics ───────────────────────────────────────────────────────────────────
# # def dice_score(a, b):
# #     inter = (a * b).sum()
# #     return float((2 * inter) / (a.sum() + b.sum() + 1e-8))


# # # ── FOV mask ──────────────────────────────────────────────────────────────────
# # def get_fov_mask(img):
# #     grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# #     _, mask = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)
# #     kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
# #     mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# #     mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
# #     return (mask > 0).astype(np.float32)


# # # ── Overlap composites ────────────────────────────────────────────────────────
# # def make_overlap_image(fixed_img, warped_img):
# #     fov = get_fov_mask(fixed_img)
# #     def norm(img):
# #         g = img.astype(np.float32).mean(axis=-1)
# #         mn, mx = g.min(), g.max()
# #         return (g - mn) / (mx - mn + 1e-8)
# #     f   = norm(fixed_img)
# #     w   = norm(warped_img)
# #     out = np.zeros((*f.shape, 3), dtype=np.float32)
# #     out[..., 0] = f
# #     out[..., 1] = w
# #     out[..., 2] = f
# #     out *= fov[..., np.newaxis]
# #     return (np.clip(out, 0, 1) * 255).astype(np.uint8)


# # def make_overlap_vessels(fixed_v, warped_v, fixed_img):
# #     fov = get_fov_mask(fixed_img)
# #     H, W = fixed_v.shape
# #     out  = np.zeros((H, W, 3), dtype=np.uint8)
# #     out[..., 0] = (warped_v * 255).astype(np.uint8)
# #     out[..., 1] = (fixed_v  * 255).astype(np.uint8)
# #     return (out * fov[..., np.newaxis]).astype(np.uint8)


# # # ── Visualisation helpers ─────────────────────────────────────────────────────
# # def mask_to_rgb(mask):
# #     g = (mask * 255).astype(np.uint8)
# #     return np.stack([g, g, g], axis=-1)


# # def add_label(img, text):
# #     out = img.copy()
# #     for color, thick in [((255, 255, 255), 2), ((0, 0, 0), 1)]:
# #         cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
# #                     0.55, color, thick, cv2.LINE_AA)
# #     return out


# # def make_row_grid(fixed_img, fixed_vessel,
# #                   moving_img, moving_vessel,
# #                   warped_img_affine, warped_vessel_affine,
# #                   warped_img_mls=None, warped_vessel_mls=None,
# #                   row_label=None,
# #                   dice_before=None, dice_affine=None, dice_mls=None,
# #                   mls_fallback=False,
# #                   decomp=None, flags=None, meta=None, meta_mls=None):
# #     GAP   = 6
# #     H     = fixed_img.shape[0]
# #     gap_v = np.ones((H, GAP, 3), dtype=np.uint8) * 60

# #     ov_affine = make_overlap_image(fixed_img, warped_img_affine)
# #     vv_affine = make_overlap_vessels(fixed_vessel, warped_vessel_affine, fixed_img)

# #     if warped_img_mls is not None:
# #         ov_mls    = make_overlap_image(fixed_img, warped_img_mls)
# #         vv_mls    = make_overlap_vessels(fixed_vessel, warped_vessel_mls, fixed_img)
# #         mls_label = "MLS(fallback)" if mls_fallback else "After MLS"
# #         panels = [
# #             add_label(fixed_img,                  "Fixed Image"),
# #             add_label(mask_to_rgb(fixed_vessel),  "Fixed Vessel"),
# #             add_label(moving_img,                 "Moving Image"),
# #             add_label(mask_to_rgb(moving_vessel), "Moving Vessel"),
# #             add_label(warped_img_affine,           "After Affine"),
# #             add_label(warped_img_mls,              mls_label),
# #             add_label(ov_affine,                  "Overlap Affine"),
# #             add_label(ov_mls,                     "Overlap MLS"),
# #             add_label(vv_mls,                     "Vessel Overlap"),
# #         ]
# #     else:
# #         panels = [
# #             add_label(fixed_img,                  "Fixed Image"),
# #             add_label(mask_to_rgb(fixed_vessel),  "Fixed Vessel"),
# #             add_label(moving_img,                 "Moving Image"),
# #             add_label(mask_to_rgb(moving_vessel), "Moving Vessel"),
# #             add_label(warped_img_affine,           "Registered"),
# #             add_label(ov_affine,                  "Fixed+Reg Overlap"),
# #             add_label(vv_affine,                  "Vessel Overlap"),
# #         ]

# #     row_parts = []
# #     for idx, panel in enumerate(panels):
# #         row_parts.append(panel)
# #         if idx < len(panels) - 1:
# #             row_parts.append(gap_v)
# #     row = np.concatenate(row_parts, axis=1)

# #     SIDEBAR_W = 170
# #     sidebar   = np.zeros((H, SIDEBAR_W, 3), dtype=np.uint8)
# #     sidebar[:] = (30, 30, 30)

# #     lines = []
# #     if row_label:
# #         lines.append(row_label)
# #     if dice_before is not None:
# #         lines.append(f"Before: {dice_before:.4f}")
# #     if dice_affine is not None:
# #         d1 = dice_affine - (dice_before or 0)
# #         lines.append(f"Affine: {dice_affine:.4f} ({d1:+.4f})")
# #     if dice_mls is not None:
# #         if mls_fallback and meta_mls and "mls_raw_dice" in meta_mls:
# #             raw   = meta_mls["mls_raw_dice"]
# #             d_raw = raw - (dice_affine or 0)
# #             lines.append(f"MLS raw:{raw:.4f} ({d_raw:+.4f})")
# #             lines.append(f"Final  :affine(fb)")
# #         else:
# #             d2 = dice_mls - (dice_affine or 0)
# #             lines.append(f"MLS    : {dice_mls:.4f} ({d2:+.4f})")
# #     if decomp:
# #         lines.append("---")
# #         lines.append(f"tx:{decomp['tx']:+.1f} ty:{decomp['ty']:+.1f}")
# #         lines.append(f"rot:{decomp['angle_deg']:+.2f}deg")
# #         lines.append(f"sx:{decomp['sx']:.3f} sy:{decomp['sy']:.3f}")
# #         lines.append(f"shear:{decomp['shear']:.3f}")
# #     if meta:
# #         lines.append(f"inlr:{meta['n_inliers']}/{meta['n_candidates']}")
# #         lines.append(f"ratio:{meta['inlier_ratio']:.3f}")
# #         if meta.get("low_conf_fallback"):
# #             lines.append("!low_conf_fallback")
# #     if meta_mls:
# #         lines.append(f"MLS ctrl:{meta_mls['mls_n_control_pts']}")
# #         if mls_fallback:
# #             lines.append("!MLS fallback→affine")
# #     if flags:
# #         lines.append("WARN:")
# #         for f in flags:
# #             lines.append(f"  {f[:24]}")

# #     flag_color = (80, 80, 255) if flags else (200, 200, 200)
# #     for li, line in enumerate(lines):
# #         y = 18 + li * 18
# #         if y > H - 4:
# #             break
# #         color = flag_color if (line.startswith("WARN") or line.startswith("  ")) \
# #                 else (200, 200, 200)
# #         if line.startswith("!"):
# #             color = (60, 200, 255)
# #         cv2.putText(sidebar, line, (4, y),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

# #     divider = np.ones((H, GAP, 3), dtype=np.uint8) * 60
# #     return np.concatenate([sidebar, divider, row], axis=1)


# # # ── Save per-pair outputs ─────────────────────────────────────────────────────
# # def save_outputs(out_dir, name,
# #                  fixed_img, fixed_vessel,
# #                  moving_img, moving_vessel,
# #                  warped_img_affine, warped_vessel_affine,
# #                  warped_img_mls=None, warped_vessel_mls=None,
# #                  dice_before=None, dice_affine=None, dice_mls=None,
# #                  mls_fallback=False,
# #                  pair_index=None, decomp=None, flags=None,
# #                  meta=None, meta_mls=None):

# #     def write_rgb(fname, img):
# #         cv2.imwrite(str(out_dir / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
# #     def write_mask(fname, m):
# #         cv2.imwrite(str(out_dir / fname), (m * 255).astype(np.uint8))

# #     write_rgb( f"{name}_affine.png",        warped_img_affine)
# #     write_mask(f"{name}_affine_vessel.png", warped_vessel_affine)
# #     write_rgb( f"{name}_overlap_affine.png",
# #                make_overlap_image(fixed_img, warped_img_affine))

# #     if warped_img_mls is not None:
# #         suffix = "mls_raw" if mls_fallback else "mls"
# #         write_rgb( f"{name}_{suffix}.png",        warped_img_mls)
# #         write_mask(f"{name}_{suffix}_vessel.png", warped_vessel_mls)
# #         write_rgb( f"{name}_overlap_{suffix}.png",
# #                    make_overlap_image(fixed_img, warped_img_mls))
# #         write_rgb( f"{name}_vessels_{suffix}.png",
# #                    make_overlap_vessels(fixed_vessel, warped_vessel_mls, fixed_img))
# #     if warped_img_mls is None:
# #         write_rgb( f"{name}_vessels_affine.png",
# #                    make_overlap_vessels(fixed_vessel, warped_vessel_affine, fixed_img))

# #     row = make_row_grid(
# #         fixed_img, fixed_vessel, moving_img, moving_vessel,
# #         warped_img_affine, warped_vessel_affine,
# #         warped_img_mls=warped_img_mls, warped_vessel_mls=warped_vessel_mls,
# #         row_label=f"#{pair_index}" if pair_index is not None else None,
# #         dice_before=dice_before, dice_affine=dice_affine, dice_mls=dice_mls,
# #         mls_fallback=mls_fallback,
# #         decomp=decomp, flags=flags, meta=meta, meta_mls=meta_mls,
# #     )
# #     write_rgb(f"{name}_grid.png", row)
# #     extra = 6 if warped_img_mls is not None else 4
# #     print(f"  Saved: {name}_grid.png  (+{extra} individual files)")
# #     return row


# # # ── Summary pages ─────────────────────────────────────────────────────────────
# # def save_summary_page(out_dir, rows, mean_before=None,
# #                       mean_affine=None, mean_mls=None, rows_per_page=20):
# #     if not rows:
# #         return
# #     pages_dir = out_dir / "summary_pages"
# #     pages_dir.mkdir(parents=True, exist_ok=True)

# #     ROW_SEP_H = 4
# #     BANNER_H  = 50
# #     max_w     = max(r.shape[1] for r in rows)

# #     def pad_row(r):
# #         if r.shape[1] < max_w:
# #             pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
# #             r = np.concatenate([r, pad], axis=1)
# #         return r

# #     padded  = [pad_row(r) for r in rows]
# #     sep     = np.full((ROW_SEP_H, max_w, 3), 100, dtype=np.uint8)
# #     n_pages = max(1, (len(padded) + rows_per_page - 1) // rows_per_page)

# #     for p in range(n_pages):
# #         chunk      = padded[p * rows_per_page : (p + 1) * rows_per_page]
# #         page_label = f"{p + 1:02d}of{n_pages:02d}"
# #         pairs_r    = (f"pairs {p * rows_per_page + 1}-"
# #                       f"{min((p + 1) * rows_per_page, len(rows))} / {len(rows)}")

# #         txt = f"CoWTracker FA/IR Registration  |  Page {page_label}  |  {pairs_r}"
# #         if mean_before is not None and mean_affine is not None:
# #             txt += (f"  |  Dice before:{mean_before:.4f}  "
# #                     f"affine:{mean_affine:.4f} ({mean_affine-mean_before:+.4f})")
# #         if mean_mls is not None:
# #             txt += f"  mls:{mean_mls:.4f} ({mean_mls-mean_affine:+.4f})"

# #         banner = np.zeros((BANNER_H, max_w, 3), dtype=np.uint8)
# #         cv2.putText(banner, txt, (10, 34),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 60), 1, cv2.LINE_AA)

# #         parts = [banner]
# #         for row in chunk:
# #             parts.append(sep)
# #             parts.append(row)

# #         page     = np.concatenate(parts, axis=0)
# #         out_path = pages_dir / f"summary_page_{page_label}.png"
# #         cv2.imwrite(str(out_path), cv2.cvtColor(page, cv2.COLOR_RGB2BGR))
# #         print(f"  Summary page {page_label} saved: {out_path.name}")

# #     print(f"\n  {n_pages} summary page(s) saved -> {pages_dir.resolve()}")


# # # ── Main ──────────────────────────────────────────────────────────────────────
# # def main():
# #     parser = argparse.ArgumentParser(
# #         description="CoWTracker Retinal Registration — Affine + MLS")
# #     parser.add_argument("--csv",               required=True)
# #     parser.add_argument("--output_dir",        default=None)
# #     parser.add_argument("--stage",             default="affine",
# #                         choices=["affine", "deformable"])
# #     parser.add_argument("--moving_col",        default="moving")
# #     parser.add_argument("--fixed_col",         default="fixed")
# #     parser.add_argument("--moving_vessel_col", default="moving_vessel_mask")
# #     parser.add_argument("--fixed_vessel_col",  default="fixed_vessel_mask")
# #     parser.add_argument("--height",            type=int,   default=TARGET_H)
# #     parser.add_argument("--width",             type=int,   default=TARGET_W)
# #     parser.add_argument("--conf_thresh",       type=float, default=0.3)
# #     parser.add_argument("--min_points",        type=int,   default=6)
# #     parser.add_argument("--max_translation",   type=float, default=80.0)
# #     parser.add_argument("--max_rotation",      type=float, default=30.0)
# #     parser.add_argument("--scale_lo",          type=float, default=0.4)
# #     parser.add_argument("--scale_hi",          type=float, default=1.6)
# #     parser.add_argument("--max_shear",         type=float, default=0.25)
# #     parser.add_argument("--min_inlier_ratio",  type=float, default=0.05)
# #     # MLS
# #     parser.add_argument("--mls_conf_thresh",   type=float, default=0.3)
# #     parser.add_argument("--mls_min_points",    type=int,   default=10)
# #     parser.add_argument("--mls_max_ctrl",      type=int,   default=500)
# #     parser.add_argument("--mls_alpha",         type=float, default=1.0,
# #                         help="Weight fall-off exponent. Higher = more localised.")
# #     parser.add_argument("--mls_skip_thresh",  type=float, default=0.25,
# #                         help="Skip MLS if affine Dice already >= this value "
# #                              "(avoids overfitting on well-registered pairs). "
# #                              "Set to 1.0 to disable.")
# #     parser.add_argument("--mls_chunk",         type=int,   default=5000,
# #                         help="Pixels per chunk (memory/speed tradeoff).")
# #     args = parser.parse_args()

# #     assert args.height % 14 == 0 and args.width % 14 == 0

# #     if args.output_dir is None:
# #         args.output_dir = ("Results_deformable" if args.stage == "deformable"
# #                            else "Results_affine")
# #     out_dir = Path(args.output_dir)
# #     out_dir.mkdir(parents=True, exist_ok=True)

# #     print("=" * 65)
# #     print(f"CoWTracker Retinal Registration  [Stage: {args.stage.upper()}]")
# #     print("=" * 65)
# #     print(f"Device      : {DEVICE}")
# #     print(f"Size        : {args.height} x {args.width}")
# #     print(f"Output      : {out_dir.resolve()}")
# #     print(f"Soft flags  : translation>{args.max_translation}px  "
# #           f"rotation>{args.max_rotation}°  "
# #           f"scale [{args.scale_lo},{args.scale_hi}]  shear>{args.max_shear}")
# #     print(f"Hard reject : |rot|>{REJECT_ABS_ROT}°  "
# #           f"scale >{REJECT_SCALE_MAX}/<{REJECT_SCALE_MIN}  "
# #           f"inlier_ratio<{args.min_inlier_ratio}")
# #     if args.stage == "deformable":
# #         print(f"MLS         : conf>{args.mls_conf_thresh}  "
# #               f"max_ctrl={args.mls_max_ctrl}  alpha={args.mls_alpha}  "
# #               f"skip_thresh={args.mls_skip_thresh}  "
# #               f"[auto-fallback to affine if MLS < affine]")
# #     print()

# #     print("Loading CoWTracker model...")
# #     model = CoWTracker.from_checkpoint(device=DEVICE, dtype=INF_DTYPE)
# #     print("Model ready.\n")

# #     df = pd.read_csv(args.csv)
# #     print(f"Found {len(df)} image pairs.\n")

# #     results  = []
# #     all_rows = []

# #     for i, row in df.iterrows():
# #         moving_path    = row[args.moving_col]
# #         fixed_path     = row[args.fixed_col]
# #         mv_vessel_path = row[args.moving_vessel_col]
# #         fx_vessel_path = row[args.fixed_vessel_col]
# #         name = (f"{i:04d}_{Path(moving_path).stem}"
# #                 f"_to_{Path(fixed_path).stem}")

# #         print(f"[{i+1}/{len(df)}] {Path(moving_path).name} -> "
# #               f"{Path(fixed_path).name}")

# #         try:
# #             fixed_img     = load_image(fixed_path,      args.height, args.width)
# #             moving_img    = load_image(moving_path,     args.height, args.width)
# #             fixed_vessel  = load_vessel(fx_vessel_path, args.height, args.width)
# #             moving_vessel = load_vessel(mv_vessel_path, args.height, args.width)

# #             # ── Stage 1: Affine ───────────────────────────────────────────────
# #             tracks, vis, conf = run_cowtracker(model, fixed_vessel, moving_vessel)

# #             M, decomp, meta, reject_reason = estimate_affine(
# #                 tracks, vis, conf,
# #                 conf_thresh=args.conf_thresh,
# #                 min_points=args.min_points,
# #                 min_inlier_ratio=args.min_inlier_ratio,
# #             )

# #             if M is None:
# #                 print(f"  Skipping — {reject_reason}")
# #                 results.append(dict(name=name, moving=moving_path,
# #                                     fixed=fixed_path, status="degenerate",
# #                                     error=reject_reason))
# #                 continue

# #             flags = flag_affine(decomp,
# #                                 max_translation=args.max_translation,
# #                                 max_rotation=args.max_rotation,
# #                                 scale_range=(args.scale_lo, args.scale_hi),
# #                                 max_shear=args.max_shear)
# #             if flags:
# #                 print(f"  ⚠ Flags: {', '.join(flags)}")

# #             h, w = args.height, args.width
# #             warped_img_affine = cv2.cvtColor(
# #                 warp_with_affine(
# #                     cv2.cvtColor(moving_img, cv2.COLOR_RGB2BGR), M, h, w),
# #                 cv2.COLOR_BGR2RGB)
# #             warped_vessel_affine = (
# #                 warp_with_affine((moving_vessel * 255).astype(np.uint8),
# #                                  M, h, w, flags=cv2.INTER_NEAREST) > 127
# #             ).astype(np.float32)

# #             fov         = get_fov_mask(fixed_img)
# #             dice_before = dice_score(moving_vessel        * fov, fixed_vessel * fov)
# #             dice_affine = dice_score(warped_vessel_affine  * fov, fixed_vessel * fov)

# #             print(f"  Dice before:{dice_before:.4f}  "
# #                   f"after affine:{dice_affine:.4f}  "
# #                   f"({dice_affine - dice_before:+.4f})"
# #                   f"{'  ⚠ REGRESSED' if dice_affine < dice_before else ''}")

# #             # ── Stage 2: MLS ──────────────────────────────────────────────────
# #             warped_img_mls    = None
# #             warped_vessel_mls = None
# #             dice_mls          = None
# #             meta_mls          = None
# #             mls_fallback      = False

# #             if args.stage == "deformable":
# #                 # ── Skip MLS if affine already good enough ────────────────
# #                 if dice_affine >= args.mls_skip_thresh:
# #                     print(f"  [S2] Skipping MLS — affine Dice {dice_affine:.4f} "
# #                           f">= skip_thresh {args.mls_skip_thresh}")
# #                     warped_img_mls    = warped_img_affine.copy()
# #                     warped_vessel_mls = warped_vessel_affine.copy()
# #                     dice_mls          = dice_affine
# #                     mls_fallback      = True
# #                     meta_mls          = {"mls_n_control_pts": 0,
# #                                          "mls_n_confident": 0,
# #                                          "mls_raw_dice": dice_affine,
# #                                          "mls_skipped": True}
# #                 else:
# #                     tracks2, vis2, conf2 = run_cowtracker(
# #                         model, fixed_vessel, warped_vessel_affine)

# #                     fixed_pts, warped_pts, meta_mls = select_mls_control_points(
# #                         tracks2, vis2, conf2,
# #                         conf_thresh=args.mls_conf_thresh,
# #                         min_points=args.mls_min_points,
# #                         max_ctrl=args.mls_max_ctrl,
# #                     )

# #                     if fixed_pts is not None:
# #                         print(f"  [S2] Computing MLS map "
# #                               f"({meta_mls['mls_n_control_pts']} ctrl pts, "
# #                               f"alpha={args.mls_alpha})...")

# #                         map_x, map_y = compute_mls_map(
# #                             fixed_pts, warped_pts, h, w,
# #                             alpha=args.mls_alpha,
# #                             chunk_size=args.mls_chunk,
# #                         )

# #                         # Always compute raw MLS warp for grid display
# #                         _warp_img = cv2.cvtColor(
# #                             warp_with_mls(
# #                                 cv2.cvtColor(warped_img_affine, cv2.COLOR_RGB2BGR),
# #                                 map_x, map_y),
# #                             cv2.COLOR_BGR2RGB)
# #                         _warp_v = (
# #                             warp_with_mls(
# #                                 (warped_vessel_affine * 255).astype(np.uint8),
# #                                 map_x, map_y, is_mask=True) > 127
# #                         ).astype(np.float32)
# #                         _dice_mls = dice_score(_warp_v * fov, fixed_vessel * fov)

# #                         # Grid always shows raw MLS result
# #                         warped_img_mls    = _warp_img
# #                         warped_vessel_mls = _warp_v
# #                         meta_mls["mls_raw_dice"] = round(_dice_mls, 4)

# #                         if _dice_mls >= dice_affine:
# #                             # MLS improved — use MLS as final result
# #                             dice_mls     = _dice_mls
# #                             mls_fallback = False
# #                             print(f"  Dice after MLS: {dice_mls:.4f}  "
# #                                   f"({dice_mls - dice_affine:+.4f} vs affine)")
# #                         else:
# #                             # MLS regressed — final result reverts to affine,
# #                             # but grid still shows raw MLS so you can see why
# #                             dice_mls     = dice_affine
# #                             mls_fallback = True
# #                             print(f"  MLS {_dice_mls:.4f} < affine {dice_affine:.4f} "
# #                                   f"— fallback to affine (grid shows raw MLS)")
# #                     else:
# #                         # Not enough control points — duplicate affine for grid
# #                         warped_img_mls    = warped_img_affine.copy()
# #                         warped_vessel_mls = warped_vessel_affine.copy()
# #                         dice_mls          = dice_affine
# #                         mls_fallback      = True
# #                         if meta_mls is None:
# #                             meta_mls = {}
# #                         meta_mls["mls_raw_dice"] = dice_affine

# #             row_grid = save_outputs(
# #                 out_dir, name,
# #                 fixed_img, fixed_vessel, moving_img, moving_vessel,
# #                 warped_img_affine, warped_vessel_affine,
# #                 warped_img_mls=warped_img_mls,
# #                 warped_vessel_mls=warped_vessel_mls,
# #                 dice_before=dice_before, dice_affine=dice_affine,
# #                 dice_mls=dice_mls, mls_fallback=mls_fallback,
# #                 pair_index=i + 1,
# #                 decomp=decomp, flags=flags, meta=meta, meta_mls=meta_mls,
# #             )
# #             all_rows.append(row_grid)

# #             rec = dict(
# #                 name=name,
# #                 moving=moving_path,
# #                 fixed=fixed_path,
# #                 dice_before=round(dice_before,  4),
# #                 dice_affine=round(dice_affine,  4),
# #                 delta_affine=round(dice_affine - dice_before, 4),
# #                 dice_regressed_affine=(dice_affine < dice_before),
# #                 tx=round(decomp["tx"],          2),
# #                 ty=round(decomp["ty"],          2),
# #                 angle_deg=round(decomp["angle_deg"], 4),
# #                 sx=round(decomp["sx"],          4),
# #                 sy=round(decomp["sy"],          4),
# #                 shear=round(decomp["shear"],    4),
# #                 n_candidates=meta["n_candidates"],
# #                 n_inliers=meta["n_inliers"],
# #                 inlier_ratio=meta["inlier_ratio"],
# #                 low_conf_fallback=meta["low_conf_fallback"],
# #                 flagged=bool(flags),
# #                 flag_reasons="; ".join(flags) if flags else "",
# #                 status="ok",
# #             )
# #             if args.stage == "deformable" and dice_mls is not None:
# #                 dice_mls_raw = (meta_mls.get("mls_raw_dice", dice_mls)
# #                                 if meta_mls else dice_mls)
# #                 rec["dice_mls"]          = round(dice_mls, 4)
# #                 rec["dice_mls_raw"]      = round(dice_mls_raw, 4)
# #                 rec["delta_mls"]         = round(dice_mls - dice_affine, 4)
# #                 rec["delta_total"]       = round(dice_mls - dice_before, 4)
# #                 rec["mls_fallback"]      = mls_fallback
# #                 rec["mls_n_ctrl"]        = (meta_mls.get("mls_n_control_pts", 0)
# #                                             if meta_mls else 0)
# #                 rec["mls_skipped"]       = meta_mls.get("mls_skipped", False) \
# #                                            if meta_mls else False
# #             results.append(rec)

# #         except Exception as e:
# #             print(f"  ERROR: {e}")
# #             results.append(dict(name=name, moving=moving_path,
# #                                 fixed=fixed_path, status="error", error=str(e)))

# #     results_df = pd.DataFrame(results)
# #     results_df.to_csv(out_dir / "results.csv", index=False)

# #     ok         = results_df[results_df["status"] == "ok"]
# #     degenerate = results_df[results_df["status"] == "degenerate"]
# #     errors     = results_df[results_df["status"] == "error"]

# #     mean_before = ok["dice_before"].mean() if len(ok) else None
# #     mean_affine = ok["dice_affine"].mean()  if len(ok) else None
# #     mean_mls    = ok["dice_mls"].mean()     if (len(ok) and "dice_mls" in ok.columns) else None

# #     if all_rows:
# #         save_summary_page(out_dir, all_rows,
# #                           mean_before=mean_before,
# #                           mean_affine=mean_affine,
# #                           mean_mls=mean_mls)

# #     flagged  = ok[ok["flagged"] == True] if "flagged" in ok.columns else pd.DataFrame()
# #     reg_aff  = ok[ok["dice_regressed_affine"] == True] if "dice_regressed_affine" in ok.columns else pd.DataFrame()
# #     fallback = ok[ok["low_conf_fallback"] == True] if "low_conf_fallback" in ok.columns else pd.DataFrame()

# #     print("\n" + "=" * 65)
# #     print("Summary")
# #     print("=" * 65)
# #     print(f"  Total pairs        : {len(df)}")
# #     print(f"  OK                 : {len(ok)}")
# #     print(f"  Degenerate/skip    : {len(degenerate)}")
# #     print(f"  File errors        : {len(errors)}")
# #     if len(ok):
# #         print(f"  Flagged (soft)     : {len(flagged)}")
# #         print(f"  Regressed (affine) : {len(reg_aff)}")
# #         print(f"  Low-conf fallback  : {len(fallback)}")
# #         print(f"  Mean Dice before   : {mean_before:.4f}")
# #         print(f"  Mean Dice affine   : {mean_affine:.4f}  "
# #               f"({mean_affine - mean_before:+.4f})")
# #         if mean_mls is not None:
# #             mls_fb   = ok[ok["mls_fallback"] == True] if "mls_fallback" in ok.columns else pd.DataFrame()
# #             mls_skip = ok[ok["mls_skipped"]  == True] if "mls_skipped"  in ok.columns else pd.DataFrame()
# #             mls_imp  = ok[(ok["mls_fallback"] == False) & (ok["mls_n_ctrl"] > 0)] if "mls_fallback" in ok.columns else pd.DataFrame()
# #             print(f"  Mean Dice MLS      : {mean_mls:.4f}  "
# #                   f"({mean_mls - mean_affine:+.4f} vs affine)")
# #             print(f"  MLS improved       : {len(mls_imp)}")
# #             print(f"  MLS→affine fallback: {len(mls_fb) - len(mls_skip)}")
# #             print(f"  MLS skipped (Dice>={args.mls_skip_thresh}): {len(mls_skip)}")
# #         print(f"  Mean rotation      : {ok['angle_deg'].mean():.2f}°")
# #         print(f"  Mean translation   : "
# #               f"tx={ok['tx'].mean():.1f}px  ty={ok['ty'].mean():.1f}px")
# #         print(f"  Mean inlier ratio  : {ok['inlier_ratio'].mean():.3f}")
# #     print(f"  Results saved      : {out_dir.resolve()}/results.csv")
# #     print(f"  Summary pages      : {out_dir.resolve()}/summary_pages/")
# #     print("=" * 65)


# # if __name__ == "__main__":
# #     main()

# import argparse
# from pathlib import Path

# import cv2
# import numpy as np
# import pandas as pd
# import torch

# from cowtracker import CoWTracker

# # ── Config ────────────────────────────────────────────────────────────────────
# DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
# INF_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
# TARGET_H  = 224   # must be multiple of 14
# TARGET_W  = 224   # must be multiple of 14

# # ── Hard-reject thresholds ────────────────────────────────────────────────────
# REJECT_ABS_ROT        = 90.0
# REJECT_SCALE_MAX      = 3.0
# REJECT_SCALE_MIN      = 0.2
# REJECT_NEGATIVE_SCALE = True


# # ── I/O ───────────────────────────────────────────────────────────────────────
# def load_image(path, h, w):
#     img = cv2.imread(str(path))
#     if img is None:
#         raise FileNotFoundError(f"Cannot read: {path}")
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


# def load_vessel(path, h, w):
#     mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
#     if mask is None:
#         raise FileNotFoundError(f"Cannot read: {path}")
#     mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
#     return (mask > 127).astype(np.float32)


# # ── Vessel enrichment ─────────────────────────────────────────────────────────
# def enrich_vessel(vessel):
#     binary  = (vessel * 255).astype(np.uint8)
#     kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     dilated = cv2.dilate(binary, kernel, iterations=2)
#     dist    = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)
#     dist    = (dist / dist.max() * 255).astype(np.uint8) if dist.max() > 0 \
#               else np.zeros_like(binary)
#     blurred = cv2.GaussianBlur(dilated, (11, 11), 3)
#     return np.stack([dilated, dist, blurred], axis=-1)


# # ── CoWTracker (shared for both stages) ───────────────────────────────────────
# def run_cowtracker(model, vessel_a, vessel_b):
#     video = np.stack([enrich_vessel(vessel_a), enrich_vessel(vessel_b)], axis=0)
#     video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float().to(DEVICE)
#     torch.cuda.empty_cache()
#     with torch.no_grad():
#         with torch.amp.autocast(device_type="cuda", dtype=INF_DTYPE):
#             predictions = model.forward(video=video_tensor, queries=None)
#     tracks = predictions["track"][0].cpu()
#     vis    = predictions["vis"][0].cpu()
#     conf   = predictions["conf"][0].cpu()
#     return tracks, vis, conf


# # ── Stage 1: Affine estimation ────────────────────────────────────────────────
# def estimate_affine(tracks, vis, conf, conf_thresh=0.3, min_points=6,
#                     min_inlier_ratio=0.05):
#     confidence        = vis[1] * conf[1]
#     mask              = confidence > conf_thresh
#     low_conf_fallback = False

#     if mask.sum() < min_points:
#         print(f"  Warning: only {int(mask.sum())} confident points "
#               f"(need {min_points}), lowering threshold to median...")
#         median_conf = float(confidence[confidence > 0].quantile(0.5)
#                             if confidence.max() > 0 else 0)
#         mask = confidence > median_conf
#         low_conf_fallback = True
#         if mask.sum() < 4:
#             print(f"  ERROR: still only {int(mask.sum())} points")
#             return None, None, None, "too_few_points"

#     src          = tracks[0][mask].numpy().astype(np.float32)
#     dst          = tracks[1][mask].numpy().astype(np.float32)
#     n_candidates = len(src)

#     print(f"  [S1] Fitting affine from {n_candidates} correspondences "
#           f"(conf>{conf_thresh:.2f})"
#           f"{' [fallback]' if low_conf_fallback else ''}")

#     M, inlier_mask = cv2.estimateAffine2D(
#         dst, src,
#         method=cv2.RANSAC,
#         ransacReprojThreshold=3.0,
#         maxIters=2000,
#         confidence=0.999,
#         refineIters=10,
#     )
#     if M is None:
#         print("  ERROR: affine estimation failed")
#         return None, None, None, "cv2_failed"

#     n_inliers    = int(inlier_mask.sum()) if inlier_mask is not None else 0
#     inlier_ratio = n_inliers / max(n_candidates, 1)
#     print(f"  [S1] RANSAC inliers: {n_inliers} / {n_candidates}  "
#           f"(ratio={inlier_ratio:.3f})")

#     if n_candidates > 1000 and inlier_ratio < min_inlier_ratio:
#         reason = (f"low inlier ratio {inlier_ratio:.3f} < {min_inlier_ratio} "
#                   f"({n_inliers}/{n_candidates})")
#         print(f"  ✗ Hard-reject: {reason}")
#         return None, None, None, reason

#     decomp = decompose_affine(M)
#     print(f"  [S1] tx={decomp['tx']:.2f}px  ty={decomp['ty']:.2f}px  "
#           f"rot={decomp['angle_deg']:.2f}°  "
#           f"sx={decomp['sx']:.4f}  sy={decomp['sy']:.4f}  "
#           f"shear={decomp['shear']:.4f}")

#     reject_reason = check_degenerate(decomp)
#     if reject_reason:
#         print(f"  ✗ Hard-reject (degenerate): {reject_reason}")
#         return None, None, None, reject_reason

#     meta = dict(
#         n_candidates=n_candidates,
#         n_inliers=n_inliers,
#         inlier_ratio=round(inlier_ratio, 4),
#         low_conf_fallback=low_conf_fallback,
#     )
#     return M, decomp, meta, None


# # ── Affine decomposition (SVD-based) ─────────────────────────────────────────
# def decompose_affine(M):
#     tx, ty = float(M[0, 2]), float(M[1, 2])
#     A      = M[:2, :2].astype(np.float64)
#     U, S, Vt = np.linalg.svd(A)
#     if np.linalg.det(U) * np.linalg.det(Vt) < 0:
#         U[:, -1] *= -1
#         S[-1]    *= -1
#     R         = U @ Vt
#     angle_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
#     sx, sy    = float(S[0]), float(S[1])
#     A_norm    = R.T @ A
#     shear     = float(A_norm[0, 1] / (sx + 1e-8))
#     return dict(tx=tx, ty=ty, angle_deg=angle_deg, sx=sx, sy=sy, shear=shear)


# def check_degenerate(decomp):
#     ang = abs(decomp["angle_deg"])
#     if ang > REJECT_ABS_ROT:
#         return f"extreme rotation {ang:.1f}° > {REJECT_ABS_ROT}°"
#     sx, sy = decomp["sx"], decomp["sy"]
#     if REJECT_NEGATIVE_SCALE and sy < 0:
#         return f"negative scale sy={sy:.4f} (reflection)"
#     for axis, s in [("sx", sx), ("sy", sy)]:
#         if abs(s) > REJECT_SCALE_MAX:
#             return f"extreme scale {axis}={s:.4f} > {REJECT_SCALE_MAX}"
#         if abs(s) < REJECT_SCALE_MIN:
#             return f"near-zero scale {axis}={s:.4f} < {REJECT_SCALE_MIN}"
#     return None


# def flag_affine(decomp,
#                 max_translation=80.0,
#                 max_rotation=30.0,
#                 scale_range=(0.4, 1.6),
#                 max_shear=0.25):
#     flags = []
#     tx, ty = abs(decomp["tx"]), abs(decomp["ty"])
#     if tx > max_translation or ty > max_translation:
#         flags.append(f"large translation ({tx:.1f}, {ty:.1f})px > {max_translation}px")
#     ang = abs(decomp["angle_deg"])
#     if ang > max_rotation:
#         flags.append(f"large rotation {ang:.1f}° > {max_rotation}°")
#     for axis, s in [("sx", decomp["sx"]), ("sy", decomp["sy"])]:
#         lo, hi = scale_range
#         if not (lo <= s <= hi):
#             flags.append(f"unusual scale {axis}={s:.4f} outside [{lo},{hi}]")
#     if abs(decomp["shear"]) > max_shear:
#         flags.append(f"high shear {decomp['shear']:.4f} > {max_shear}")
#     return flags


# # ── Stage 1: Warping ──────────────────────────────────────────────────────────
# def warp_with_affine(img, M, h, w, flags=cv2.INTER_LINEAR):
#     return cv2.warpAffine(img, M, (w, h), flags=flags,
#                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# # ── Stage 2: MLS — Moving Least Squares (Schaefer et al. 2006) ───────────────
# def select_tps_control_points(tracks, vis, conf, conf_thresh=0.3,
#                                min_points=10, max_ctrl=500):
#     """
#     Select control points from Stage 2 CoWTracker correspondences for TPS.

#     tracks[0] = points in fixed frame        (target positions)
#     tracks[1] = points in affine-warped frame (source positions to sample from)

#     Returns fixed_pts, warped_pts, meta (or None, None, None if too few points).
#     """
#     from scipy.interpolate import RBFInterpolator  # lazy import — check available
#     confidence = vis[1] * conf[1]
#     mask       = confidence > conf_thresh
#     n_conf     = int(mask.sum())

#     if n_conf < min_points:
#         print(f"  [S2] Only {n_conf} confident points — "
#               f"TPS skipped, keeping affine result")
#         return None, None, None

#     fixed_pts  = tracks[0][mask].numpy().astype(np.float64)
#     warped_pts = tracks[1][mask].numpy().astype(np.float64)

#     if len(fixed_pts) > max_ctrl:
#         idx        = np.random.choice(len(fixed_pts), max_ctrl, replace=False)
#         fixed_pts  = fixed_pts[idx]
#         warped_pts = warped_pts[idx]

#     n_ctrl = len(fixed_pts)
#     meta   = dict(tps_n_control_pts=n_ctrl, tps_n_confident=n_conf)
#     return fixed_pts, warped_pts, meta


# def compute_tps_map(fixed_pts, warped_pts, h, w, smoothing=1.0):
#     """
#     Compute Thin Plate Spline deformation map using RBFInterpolator.

#     fixed_pts   : (N, 2) float64  control points in fixed frame
#     warped_pts  : (N, 2) float64  corresponding points in affine-warped frame
#     smoothing   : regularisation (higher = smoother, less overfitting)

#     Returns map_x, map_y  (H, W) float32 for cv2.remap.
#     """
#     from scipy.interpolate import RBFInterpolator

#     # Fit two TPS interpolators: one for x-displacements, one for y-displacements
#     # We interpolate the mapping: fixed_pts -> warped_pts
#     rbf_x = RBFInterpolator(fixed_pts, warped_pts[:, 0],
#                              kernel="thin_plate_spline", smoothing=smoothing)
#     rbf_y = RBFInterpolator(fixed_pts, warped_pts[:, 1],
#                              kernel="thin_plate_spline", smoothing=smoothing)

#     # Build dense query grid over fixed frame
#     yy, xx  = np.mgrid[0:h, 0:w]
#     grid    = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)

#     map_x = rbf_x(grid).reshape(h, w).astype(np.float32)
#     map_y = rbf_y(grid).reshape(h, w).astype(np.float32)

#     return map_x, map_y


# def warp_with_tps(img, map_x, map_y, is_mask=False):
#     interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
#     return cv2.remap(img, map_x, map_y, interp,
#                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# # ── Metrics ───────────────────────────────────────────────────────────────────
# def dice_score(a, b):
#     inter = (a * b).sum()
#     return float((2 * inter) / (a.sum() + b.sum() + 1e-8))


# # ── FOV mask ──────────────────────────────────────────────────────────────────
# def get_fov_mask(img):
#     grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     _, mask = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)
#     kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
#     mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
#     return (mask > 0).astype(np.float32)


# # ── Overlap composites ────────────────────────────────────────────────────────
# def make_overlap_image(fixed_img, warped_img):
#     fov = get_fov_mask(fixed_img)
#     def norm(img):
#         g = img.astype(np.float32).mean(axis=-1)
#         mn, mx = g.min(), g.max()
#         return (g - mn) / (mx - mn + 1e-8)
#     f   = norm(fixed_img)
#     w   = norm(warped_img)
#     out = np.zeros((*f.shape, 3), dtype=np.float32)
#     out[..., 0] = f
#     out[..., 1] = w
#     out[..., 2] = f
#     out *= fov[..., np.newaxis]
#     return (np.clip(out, 0, 1) * 255).astype(np.uint8)


# def make_overlap_vessels(fixed_v, warped_v, fixed_img):
#     fov = get_fov_mask(fixed_img)
#     H, W = fixed_v.shape
#     out  = np.zeros((H, W, 3), dtype=np.uint8)
#     out[..., 0] = (warped_v * 255).astype(np.uint8)
#     out[..., 1] = (fixed_v  * 255).astype(np.uint8)
#     return (out * fov[..., np.newaxis]).astype(np.uint8)


# # ── Visualisation helpers ─────────────────────────────────────────────────────
# def mask_to_rgb(mask):
#     g = (mask * 255).astype(np.uint8)
#     return np.stack([g, g, g], axis=-1)


# def add_label(img, text):
#     out = img.copy()
#     for color, thick in [((255, 255, 255), 2), ((0, 0, 0), 1)]:
#         cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.55, color, thick, cv2.LINE_AA)
#     return out


# def make_row_grid(fixed_img, fixed_vessel,
#                   moving_img, moving_vessel,
#                   warped_img_affine, warped_vessel_affine,
#                   warped_img_mls=None, warped_vessel_mls=None,
#                   row_label=None,
#                   dice_before=None, dice_affine=None, dice_mls=None,
#                   mls_fallback=False,
#                   decomp=None, flags=None, meta=None, meta_mls=None):
#     GAP   = 6
#     H     = fixed_img.shape[0]
#     gap_v = np.ones((H, GAP, 3), dtype=np.uint8) * 60

#     ov_affine = make_overlap_image(fixed_img, warped_img_affine)
#     vv_affine = make_overlap_vessels(fixed_vessel, warped_vessel_affine, fixed_img)

#     if warped_img_mls is not None:
#         ov_mls    = make_overlap_image(fixed_img, warped_img_mls)
#         vv_mls    = make_overlap_vessels(fixed_vessel, warped_vessel_mls, fixed_img)
#         tps_label = "TPS(fallback)" if mls_fallback else "After TPS"
#         panels = [
#             add_label(fixed_img,                  "Fixed Image"),
#             add_label(mask_to_rgb(fixed_vessel),  "Fixed Vessel"),
#             add_label(moving_img,                 "Moving Image"),
#             add_label(mask_to_rgb(moving_vessel), "Moving Vessel"),
#             add_label(warped_img_affine,           "After Affine"),
#             add_label(warped_img_mls,              tps_label),
#             add_label(ov_affine,                  "Overlap Affine"),
#             add_label(ov_mls,                     "Overlap MLS"),
#             add_label(vv_mls,                     "Vessel Overlap"),
#         ]
#     else:
#         panels = [
#             add_label(fixed_img,                  "Fixed Image"),
#             add_label(mask_to_rgb(fixed_vessel),  "Fixed Vessel"),
#             add_label(moving_img,                 "Moving Image"),
#             add_label(mask_to_rgb(moving_vessel), "Moving Vessel"),
#             add_label(warped_img_affine,           "Registered"),
#             add_label(ov_affine,                  "Fixed+Reg Overlap"),
#             add_label(vv_affine,                  "Vessel Overlap"),
#         ]

#     row_parts = []
#     for idx, panel in enumerate(panels):
#         row_parts.append(panel)
#         if idx < len(panels) - 1:
#             row_parts.append(gap_v)
#     row = np.concatenate(row_parts, axis=1)

#     SIDEBAR_W = 170
#     sidebar   = np.zeros((H, SIDEBAR_W, 3), dtype=np.uint8)
#     sidebar[:] = (30, 30, 30)

#     lines = []
#     if row_label:
#         lines.append(row_label)
#     if dice_before is not None:
#         lines.append(f"Before: {dice_before:.4f}")
#     if dice_affine is not None:
#         d1 = dice_affine - (dice_before or 0)
#         lines.append(f"Affine: {dice_affine:.4f} ({d1:+.4f})")
#     if dice_mls is not None:
#         if mls_fallback and meta_mls and "tps_raw_dice" in meta_mls:
#             raw   = meta_mls["tps_raw_dice"]
#             d_raw = raw - (dice_affine or 0)
#             lines.append(f"TPS raw:{raw:.4f} ({d_raw:+.4f})")
#             lines.append(f"Final  :affine(fb)")
#         else:
#             d2 = dice_mls - (dice_affine or 0)
#             lines.append(f"TPS    : {dice_mls:.4f} ({d2:+.4f})")
#     if decomp:
#         lines.append("---")
#         lines.append(f"tx:{decomp['tx']:+.1f} ty:{decomp['ty']:+.1f}")
#         lines.append(f"rot:{decomp['angle_deg']:+.2f}deg")
#         lines.append(f"sx:{decomp['sx']:.3f} sy:{decomp['sy']:.3f}")
#         lines.append(f"shear:{decomp['shear']:.3f}")
#     if meta:
#         lines.append(f"inlr:{meta['n_inliers']}/{meta['n_candidates']}")
#         lines.append(f"ratio:{meta['inlier_ratio']:.3f}")
#         if meta.get("low_conf_fallback"):
#             lines.append("!low_conf_fallback")
#     if meta_mls:
#         lines.append(f"TPS ctrl:{meta_mls.get('tps_n_control_pts', 0)}")
#         if mls_fallback:
#             lines.append("!TPS fallback→affine")
#     if flags:
#         lines.append("WARN:")
#         for f in flags:
#             lines.append(f"  {f[:24]}")

#     flag_color = (80, 80, 255) if flags else (200, 200, 200)
#     for li, line in enumerate(lines):
#         y = 18 + li * 18
#         if y > H - 4:
#             break
#         color = flag_color if (line.startswith("WARN") or line.startswith("  ")) \
#                 else (200, 200, 200)
#         if line.startswith("!"):
#             color = (60, 200, 255)
#         cv2.putText(sidebar, line, (4, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

#     divider = np.ones((H, GAP, 3), dtype=np.uint8) * 60
#     return np.concatenate([sidebar, divider, row], axis=1)


# # ── Save per-pair outputs ─────────────────────────────────────────────────────
# def save_outputs(out_dir, name,
#                  fixed_img, fixed_vessel,
#                  moving_img, moving_vessel,
#                  warped_img_affine, warped_vessel_affine,
#                  warped_img_mls=None, warped_vessel_mls=None,
#                  dice_before=None, dice_affine=None, dice_mls=None,
#                  mls_fallback=False,
#                  pair_index=None, decomp=None, flags=None,
#                  meta=None, meta_mls=None):

#     def write_rgb(fname, img):
#         cv2.imwrite(str(out_dir / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#     def write_mask(fname, m):
#         cv2.imwrite(str(out_dir / fname), (m * 255).astype(np.uint8))

#     write_rgb( f"{name}_affine.png",        warped_img_affine)
#     write_mask(f"{name}_affine_vessel.png", warped_vessel_affine)
#     write_rgb( f"{name}_overlap_affine.png",
#                make_overlap_image(fixed_img, warped_img_affine))

#     if warped_img_mls is not None:
#         suffix = "mls_raw" if mls_fallback else "mls"
#         write_rgb( f"{name}_{suffix}.png",        warped_img_mls)
#         write_mask(f"{name}_{suffix}_vessel.png", warped_vessel_mls)
#         write_rgb( f"{name}_overlap_{suffix}.png",
#                    make_overlap_image(fixed_img, warped_img_mls))
#         write_rgb( f"{name}_vessels_{suffix}.png",
#                    make_overlap_vessels(fixed_vessel, warped_vessel_mls, fixed_img))
#     if warped_img_mls is None:
#         write_rgb( f"{name}_vessels_affine.png",
#                    make_overlap_vessels(fixed_vessel, warped_vessel_affine, fixed_img))

#     row = make_row_grid(
#         fixed_img, fixed_vessel, moving_img, moving_vessel,
#         warped_img_affine, warped_vessel_affine,
#         warped_img_mls=warped_img_mls, warped_vessel_mls=warped_vessel_mls,
#         row_label=f"#{pair_index}" if pair_index is not None else None,
#         dice_before=dice_before, dice_affine=dice_affine, dice_mls=dice_mls,
#         mls_fallback=mls_fallback,
#         decomp=decomp, flags=flags, meta=meta, meta_mls=meta_mls,
#     )
#     write_rgb(f"{name}_grid.png", row)
#     extra = 6 if warped_img_mls is not None else 4
#     print(f"  Saved: {name}_grid.png  (+{extra} individual files)")
#     return row


# # ── Summary pages ─────────────────────────────────────────────────────────────
# def save_summary_page(out_dir, rows, mean_before=None,
#                       mean_affine=None, mean_mls=None, rows_per_page=20):
#     if not rows:
#         return
#     pages_dir = out_dir / "summary_pages"
#     pages_dir.mkdir(parents=True, exist_ok=True)

#     ROW_SEP_H = 4
#     BANNER_H  = 50
#     max_w     = max(r.shape[1] for r in rows)

#     def pad_row(r):
#         if r.shape[1] < max_w:
#             pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
#             r = np.concatenate([r, pad], axis=1)
#         return r

#     padded  = [pad_row(r) for r in rows]
#     sep     = np.full((ROW_SEP_H, max_w, 3), 100, dtype=np.uint8)
#     n_pages = max(1, (len(padded) + rows_per_page - 1) // rows_per_page)

#     for p in range(n_pages):
#         chunk      = padded[p * rows_per_page : (p + 1) * rows_per_page]
#         page_label = f"{p + 1:02d}of{n_pages:02d}"
#         pairs_r    = (f"pairs {p * rows_per_page + 1}-"
#                       f"{min((p + 1) * rows_per_page, len(rows))} / {len(rows)}")

#         txt = f"CoWTracker FA/IR Registration  |  Page {page_label}  |  {pairs_r}"
#         if mean_before is not None and mean_affine is not None:
#             txt += (f"  |  Dice before:{mean_before:.4f}  "
#                     f"affine:{mean_affine:.4f} ({mean_affine-mean_before:+.4f})")
#         if mean_mls is not None:
#             txt += f"  mls:{mean_mls:.4f} ({mean_mls-mean_affine:+.4f})"

#         banner = np.zeros((BANNER_H, max_w, 3), dtype=np.uint8)
#         cv2.putText(banner, txt, (10, 34),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 60), 1, cv2.LINE_AA)

#         parts = [banner]
#         for row in chunk:
#             parts.append(sep)
#             parts.append(row)

#         page     = np.concatenate(parts, axis=0)
#         out_path = pages_dir / f"summary_page_{page_label}.png"
#         cv2.imwrite(str(out_path), cv2.cvtColor(page, cv2.COLOR_RGB2BGR))
#         print(f"  Summary page {page_label} saved: {out_path.name}")

#     print(f"\n  {n_pages} summary page(s) saved -> {pages_dir.resolve()}")


# # ── Main ──────────────────────────────────────────────────────────────────────
# def main():
#     parser = argparse.ArgumentParser(
#         description="CoWTracker Retinal Registration — Affine + MLS")
#     parser.add_argument("--csv",               required=True)
#     parser.add_argument("--output_dir",        default=None)
#     parser.add_argument("--stage",             default="affine",
#                         choices=["affine", "deformable"])
#     parser.add_argument("--moving_col",        default="moving")
#     parser.add_argument("--fixed_col",         default="fixed")
#     parser.add_argument("--moving_vessel_col", default="moving_vessel_mask")
#     parser.add_argument("--fixed_vessel_col",  default="fixed_vessel_mask")
#     parser.add_argument("--height",            type=int,   default=TARGET_H)
#     parser.add_argument("--width",             type=int,   default=TARGET_W)
#     parser.add_argument("--conf_thresh",       type=float, default=0.3)
#     parser.add_argument("--min_points",        type=int,   default=6)
#     parser.add_argument("--max_translation",   type=float, default=80.0)
#     parser.add_argument("--max_rotation",      type=float, default=30.0)
#     parser.add_argument("--scale_lo",          type=float, default=0.4)
#     parser.add_argument("--scale_hi",          type=float, default=1.6)
#     parser.add_argument("--max_shear",         type=float, default=0.25)
#     parser.add_argument("--min_inlier_ratio",  type=float, default=0.05)
#     # TPS (Stage 2)
#     parser.add_argument("--tps_conf_thresh",   type=float, default=0.3)
#     parser.add_argument("--tps_min_points",    type=int,   default=10)
#     parser.add_argument("--tps_max_ctrl",      type=int,   default=500)
#     parser.add_argument("--tps_smoothing",     type=float, default=1.0,
#                         help="TPS regularisation. Higher = smoother, less overfitting.")
#     parser.add_argument("--tps_skip_thresh",   type=float, default=0.20,
#                         help="Skip TPS if affine Dice already >= this value. "
#                              "Set to 1.0 to disable.")
#     args = parser.parse_args()

#     assert args.height % 14 == 0 and args.width % 14 == 0

#     if args.output_dir is None:
#         args.output_dir = ("Results_deformable" if args.stage == "deformable"
#                            else "Results_affine")
#     out_dir = Path(args.output_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     print("=" * 65)
#     print(f"CoWTracker Retinal Registration  [Stage: {args.stage.upper()}]")
#     print("=" * 65)
#     print(f"Device      : {DEVICE}")
#     print(f"Size        : {args.height} x {args.width}")
#     print(f"Output      : {out_dir.resolve()}")
#     print(f"Soft flags  : translation>{args.max_translation}px  "
#           f"rotation>{args.max_rotation}°  "
#           f"scale [{args.scale_lo},{args.scale_hi}]  shear>{args.max_shear}")
#     print(f"Hard reject : |rot|>{REJECT_ABS_ROT}°  "
#           f"scale >{REJECT_SCALE_MAX}/<{REJECT_SCALE_MIN}  "
#           f"inlier_ratio<{args.min_inlier_ratio}")
#     if args.stage == "deformable":
#         print(f"TPS         : conf>{args.tps_conf_thresh}  "
#               f"max_ctrl={args.tps_max_ctrl}  smoothing={args.tps_smoothing}  "
#               f"skip_thresh={args.tps_skip_thresh}  "
#               f"[auto-fallback to affine if TPS < affine]")
#     print()

#     print("Loading CoWTracker model...")
#     model = CoWTracker.from_checkpoint(device=DEVICE, dtype=INF_DTYPE)
#     print("Model ready.\n")

#     df = pd.read_csv(args.csv)
#     print(f"Found {len(df)} image pairs.\n")

#     results  = []
#     all_rows = []

#     for i, row in df.iterrows():
#         moving_path    = row[args.moving_col]
#         fixed_path     = row[args.fixed_col]
#         mv_vessel_path = row[args.moving_vessel_col]
#         fx_vessel_path = row[args.fixed_vessel_col]
#         name = (f"{i:04d}_{Path(moving_path).stem}"
#                 f"_to_{Path(fixed_path).stem}")

#         print(f"[{i+1}/{len(df)}] {Path(moving_path).name} -> "
#               f"{Path(fixed_path).name}")

#         try:
#             fixed_img     = load_image(fixed_path,      args.height, args.width)
#             moving_img    = load_image(moving_path,     args.height, args.width)
#             fixed_vessel  = load_vessel(fx_vessel_path, args.height, args.width)
#             moving_vessel = load_vessel(mv_vessel_path, args.height, args.width)

#             # ── Stage 1: Affine ───────────────────────────────────────────────
#             tracks, vis, conf = run_cowtracker(model, fixed_vessel, moving_vessel)

#             M, decomp, meta, reject_reason = estimate_affine(
#                 tracks, vis, conf,
#                 conf_thresh=args.conf_thresh,
#                 min_points=args.min_points,
#                 min_inlier_ratio=args.min_inlier_ratio,
#             )

#             if M is None:
#                 print(f"  Skipping — {reject_reason}")
#                 results.append(dict(name=name, moving=moving_path,
#                                     fixed=fixed_path, status="degenerate",
#                                     error=reject_reason))
#                 continue

#             flags = flag_affine(decomp,
#                                 max_translation=args.max_translation,
#                                 max_rotation=args.max_rotation,
#                                 scale_range=(args.scale_lo, args.scale_hi),
#                                 max_shear=args.max_shear)
#             if flags:
#                 print(f"  ⚠ Flags: {', '.join(flags)}")

#             h, w = args.height, args.width
#             warped_img_affine = cv2.cvtColor(
#                 warp_with_affine(
#                     cv2.cvtColor(moving_img, cv2.COLOR_RGB2BGR), M, h, w),
#                 cv2.COLOR_BGR2RGB)
#             warped_vessel_affine = (
#                 warp_with_affine((moving_vessel * 255).astype(np.uint8),
#                                  M, h, w, flags=cv2.INTER_NEAREST) > 127
#             ).astype(np.float32)

#             fov         = get_fov_mask(fixed_img)
#             dice_before = dice_score(moving_vessel        * fov, fixed_vessel * fov)
#             dice_affine = dice_score(warped_vessel_affine  * fov, fixed_vessel * fov)

#             print(f"  Dice before:{dice_before:.4f}  "
#                   f"after affine:{dice_affine:.4f}  "
#                   f"({dice_affine - dice_before:+.4f})"
#                   f"{'  ⚠ REGRESSED' if dice_affine < dice_before else ''}")

#             # ── Stage 2: TPS ──────────────────────────────────────────────────
#             warped_img_tps    = None
#             warped_vessel_tps = None
#             dice_tps          = None
#             meta_tps          = None
#             tps_fallback      = False

#             if args.stage == "deformable":
#                 # ── Skip TPS if affine already good enough ────────────────
#                 if dice_affine >= args.tps_skip_thresh:
#                     print(f"  [S2] Skipping TPS — affine Dice {dice_affine:.4f} "
#                           f">= skip_thresh {args.tps_skip_thresh}")
#                     warped_img_tps    = warped_img_affine.copy()
#                     warped_vessel_tps = warped_vessel_affine.copy()
#                     dice_tps          = dice_affine
#                     tps_fallback      = True
#                     meta_tps          = {"tps_n_control_pts": 0,
#                                          "tps_n_confident": 0,
#                                          "tps_raw_dice": dice_affine,
#                                          "tps_skipped": True}
#                 else:
#                     tracks2, vis2, conf2 = run_cowtracker(
#                         model, fixed_vessel, warped_vessel_affine)

#                     fixed_pts, warped_pts, meta_tps = select_tps_control_points(
#                         tracks2, vis2, conf2,
#                         conf_thresh=args.tps_conf_thresh,
#                         min_points=args.tps_min_points,
#                         max_ctrl=args.tps_max_ctrl,
#                     )

#                     if fixed_pts is not None:
#                         print(f"  [S2] Computing TPS map "
#                               f"({meta_tps['tps_n_control_pts']} ctrl pts, "
#                               f"smoothing={args.tps_smoothing})...")

#                         map_x, map_y = compute_tps_map(
#                             fixed_pts, warped_pts, h, w,
#                             smoothing=args.tps_smoothing,
#                         )

#                         # Always compute raw TPS warp for grid display
#                         _warp_img = cv2.cvtColor(
#                             warp_with_tps(
#                                 cv2.cvtColor(warped_img_affine, cv2.COLOR_RGB2BGR),
#                                 map_x, map_y),
#                             cv2.COLOR_BGR2RGB)
#                         _warp_v = (
#                             warp_with_tps(
#                                 (warped_vessel_affine * 255).astype(np.uint8),
#                                 map_x, map_y, is_mask=True) > 127
#                         ).astype(np.float32)
#                         _dice_tps = dice_score(_warp_v * fov, fixed_vessel * fov)

#                         # Grid always shows raw TPS result
#                         warped_img_tps    = _warp_img
#                         warped_vessel_tps = _warp_v
#                         meta_tps["tps_raw_dice"] = round(_dice_tps, 4)

#                         if _dice_tps >= dice_affine:
#                             # TPS improved — use as final result
#                             dice_tps     = _dice_tps
#                             tps_fallback = False
#                             print(f"  Dice after TPS: {dice_tps:.4f}  "
#                                   f"({dice_tps - dice_affine:+.4f} vs affine)")
#                         else:
#                             # TPS regressed — revert to affine,
#                             # grid still shows raw TPS so you can see why
#                             dice_tps     = dice_affine
#                             tps_fallback = True
#                             print(f"  TPS {_dice_tps:.4f} < affine {dice_affine:.4f} "
#                                   f"— fallback to affine (grid shows raw TPS)")
#                     else:
#                         # Not enough control points — duplicate affine for grid
#                         warped_img_tps    = warped_img_affine.copy()
#                         warped_vessel_tps = warped_vessel_affine.copy()
#                         dice_tps          = dice_affine
#                         tps_fallback      = True
#                         if meta_tps is None:
#                             meta_tps = {}
#                         meta_tps["tps_raw_dice"] = dice_affine

#             row_grid = save_outputs(
#                 out_dir, name,
#                 fixed_img, fixed_vessel, moving_img, moving_vessel,
#                 warped_img_affine, warped_vessel_affine,
#                 warped_img_mls=warped_img_tps if args.stage == "deformable" else None,
#                 warped_vessel_mls=warped_vessel_tps if args.stage == "deformable" else None,
#                 dice_before=dice_before, dice_affine=dice_affine,
#                 dice_mls=dice_tps if args.stage == "deformable" else None,
#                 mls_fallback=tps_fallback if args.stage == "deformable" else False,
#                 pair_index=i + 1,
#                 decomp=decomp, flags=flags, meta=meta,
#                 meta_mls=meta_tps if args.stage == "deformable" else None,
#             )
#             all_rows.append(row_grid)

#             rec = dict(
#                 name=name,
#                 moving=moving_path,
#                 fixed=fixed_path,
#                 dice_before=round(dice_before,  4),
#                 dice_affine=round(dice_affine,  4),
#                 delta_affine=round(dice_affine - dice_before, 4),
#                 dice_regressed_affine=(dice_affine < dice_before),
#                 tx=round(decomp["tx"],          2),
#                 ty=round(decomp["ty"],          2),
#                 angle_deg=round(decomp["angle_deg"], 4),
#                 sx=round(decomp["sx"],          4),
#                 sy=round(decomp["sy"],          4),
#                 shear=round(decomp["shear"],    4),
#                 n_candidates=meta["n_candidates"],
#                 n_inliers=meta["n_inliers"],
#                 inlier_ratio=meta["inlier_ratio"],
#                 low_conf_fallback=meta["low_conf_fallback"],
#                 flagged=bool(flags),
#                 flag_reasons="; ".join(flags) if flags else "",
#                 status="ok",
#             )
#             if args.stage == "deformable" and dice_tps is not None:
#                 dice_tps_raw = (meta_tps.get("tps_raw_dice", dice_tps)
#                                 if meta_tps else dice_tps)
#                 rec["dice_tps"]          = round(dice_tps, 4)
#                 rec["dice_tps_raw"]      = round(dice_tps_raw, 4)
#                 rec["delta_tps"]         = round(dice_tps - dice_affine, 4)
#                 rec["delta_total"]       = round(dice_tps - dice_before, 4)
#                 rec["tps_fallback"]      = tps_fallback
#                 rec["tps_n_ctrl"]        = (meta_tps.get("tps_n_control_pts", 0)
#                                             if meta_tps else 0)
#                 rec["tps_skipped"]       = meta_tps.get("tps_skipped", False) \
#                                            if meta_tps else False
#             results.append(rec)

#         except Exception as e:
#             print(f"  ERROR: {e}")
#             results.append(dict(name=name, moving=moving_path,
#                                 fixed=fixed_path, status="error", error=str(e)))

#     results_df = pd.DataFrame(results)
#     results_df.to_csv(out_dir / "results.csv", index=False)

#     ok         = results_df[results_df["status"] == "ok"]
#     degenerate = results_df[results_df["status"] == "degenerate"]
#     errors     = results_df[results_df["status"] == "error"]

#     mean_before = ok["dice_before"].mean() if len(ok) else None
#     mean_affine = ok["dice_affine"].mean()  if len(ok) else None
#     mean_tps    = ok["dice_tps"].mean()     if (len(ok) and "dice_tps" in ok.columns) else None

#     if all_rows:
#         save_summary_page(out_dir, all_rows,
#                           mean_before=mean_before,
#                           mean_affine=mean_affine,
#                           mean_mls=mean_tps)

#     flagged  = ok[ok["flagged"] == True] if "flagged" in ok.columns else pd.DataFrame()
#     reg_aff  = ok[ok["dice_regressed_affine"] == True] if "dice_regressed_affine" in ok.columns else pd.DataFrame()
#     fallback = ok[ok["low_conf_fallback"] == True] if "low_conf_fallback" in ok.columns else pd.DataFrame()

#     print("\n" + "=" * 65)
#     print("Summary")
#     print("=" * 65)
#     print(f"  Total pairs        : {len(df)}")
#     print(f"  OK                 : {len(ok)}")
#     print(f"  Degenerate/skip    : {len(degenerate)}")
#     print(f"  File errors        : {len(errors)}")
#     if len(ok):
#         print(f"  Flagged (soft)     : {len(flagged)}")
#         print(f"  Regressed (affine) : {len(reg_aff)}")
#         print(f"  Low-conf fallback  : {len(fallback)}")
#         print(f"  Mean Dice before   : {mean_before:.4f}")
#         print(f"  Mean Dice affine   : {mean_affine:.4f}  "
#               f"({mean_affine - mean_before:+.4f})")
#         if mean_tps is not None:
#             tps_fb   = ok[ok["tps_fallback"] == True] if "tps_fallback" in ok.columns else pd.DataFrame()
#             tps_skip = ok[ok["tps_skipped"]  == True] if "tps_skipped"  in ok.columns else pd.DataFrame()
#             tps_imp  = ok[(ok["tps_fallback"] == False) & (ok["tps_n_ctrl"] > 0)] if "tps_fallback" in ok.columns else pd.DataFrame()
#             print(f"  Mean Dice TPS      : {mean_tps:.4f}  "
#                   f"({mean_tps - mean_affine:+.4f} vs affine)")
#             print(f"  TPS improved       : {len(tps_imp)}")
#             print(f"  TPS→affine fallback: {len(tps_fb) - len(tps_skip)}")
#             print(f"  TPS skipped (Dice>={args.tps_skip_thresh}): {len(tps_skip)}")
#         print(f"  Mean rotation      : {ok['angle_deg'].mean():.2f}°")
#         print(f"  Mean translation   : "
#               f"tx={ok['tx'].mean():.1f}px  ty={ok['ty'].mean():.1f}px")
#         print(f"  Mean inlier ratio  : {ok['inlier_ratio'].mean():.3f}")
#     print(f"  Results saved      : {out_dir.resolve()}/results.csv")
#     print(f"  Summary pages      : {out_dir.resolve()}/summary_pages/")
#     print("=" * 65)


# if __name__ == "__main__":
#     main()


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
TARGET_H  = 224   # must be multiple of 14
TARGET_W  = 224   # must be multiple of 14

# ── Hard-reject thresholds ────────────────────────────────────────────────────
REJECT_ABS_ROT        = 90.0
REJECT_SCALE_MAX      = 3.0
REJECT_SCALE_MIN      = 0.2
REJECT_NEGATIVE_SCALE = True


# ── I/O ───────────────────────────────────────────────────────────────────────
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


# ── CoWTracker (shared for both stages) ───────────────────────────────────────
def run_cowtracker(model, vessel_a, vessel_b):
    video = np.stack([enrich_vessel(vessel_a), enrich_vessel(vessel_b)], axis=0)
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float().to(DEVICE)
    torch.cuda.empty_cache()
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=INF_DTYPE):
            predictions = model.forward(video=video_tensor, queries=None)
    tracks = predictions["track"][0].cpu()
    vis    = predictions["vis"][0].cpu()
    conf   = predictions["conf"][0].cpu()
    return tracks, vis, conf


# ── Stage 1: Affine estimation ────────────────────────────────────────────────
def estimate_affine(tracks, vis, conf, conf_thresh=0.3, min_points=6,
                    min_inlier_ratio=0.05):
    confidence        = vis[1] * conf[1]
    mask              = confidence > conf_thresh
    low_conf_fallback = False

    if mask.sum() < min_points:
        print(f"  Warning: only {int(mask.sum())} confident points "
              f"(need {min_points}), lowering threshold to median...")
        median_conf = float(confidence[confidence > 0].quantile(0.5)
                            if confidence.max() > 0 else 0)
        mask = confidence > median_conf
        low_conf_fallback = True
        if mask.sum() < 4:
            print(f"  ERROR: still only {int(mask.sum())} points")
            return None, None, None, "too_few_points"

    src          = tracks[0][mask].numpy().astype(np.float32)
    dst          = tracks[1][mask].numpy().astype(np.float32)
    n_candidates = len(src)

    print(f"  [S1] Fitting affine from {n_candidates} correspondences "
          f"(conf>{conf_thresh:.2f})"
          f"{' [fallback]' if low_conf_fallback else ''}")

    M, inlier_mask = cv2.estimateAffine2D(
        dst, src,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.999,
        refineIters=10,
    )
    if M is None:
        print("  ERROR: affine estimation failed")
        return None, None, None, "cv2_failed"

    n_inliers    = int(inlier_mask.sum()) if inlier_mask is not None else 0
    inlier_ratio = n_inliers / max(n_candidates, 1)
    print(f"  [S1] RANSAC inliers: {n_inliers} / {n_candidates}  "
          f"(ratio={inlier_ratio:.3f})")

    if n_candidates > 1000 and inlier_ratio < min_inlier_ratio:
        reason = (f"low inlier ratio {inlier_ratio:.3f} < {min_inlier_ratio} "
                  f"({n_inliers}/{n_candidates})")
        print(f"  ✗ Hard-reject: {reason}")
        return None, None, None, reason

    decomp = decompose_affine(M)
    print(f"  [S1] tx={decomp['tx']:.2f}px  ty={decomp['ty']:.2f}px  "
          f"rot={decomp['angle_deg']:.2f}°  "
          f"sx={decomp['sx']:.4f}  sy={decomp['sy']:.4f}  "
          f"shear={decomp['shear']:.4f}")

    reject_reason = check_degenerate(decomp)
    if reject_reason:
        print(f"  ✗ Hard-reject (degenerate): {reject_reason}")
        return None, None, None, reject_reason

    meta = dict(
        n_candidates=n_candidates,
        n_inliers=n_inliers,
        inlier_ratio=round(inlier_ratio, 4),
        low_conf_fallback=low_conf_fallback,
    )
    return M, decomp, meta, None


# ── Affine decomposition (SVD-based) ─────────────────────────────────────────
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
    A_norm    = R.T @ A
    shear     = float(A_norm[0, 1] / (sx + 1e-8))
    return dict(tx=tx, ty=ty, angle_deg=angle_deg, sx=sx, sy=sy, shear=shear)


def check_degenerate(decomp):
    ang = abs(decomp["angle_deg"])
    if ang > REJECT_ABS_ROT:
        return f"extreme rotation {ang:.1f}° > {REJECT_ABS_ROT}°"
    sx, sy = decomp["sx"], decomp["sy"]
    if REJECT_NEGATIVE_SCALE and sy < 0:
        return f"negative scale sy={sy:.4f} (reflection)"
    for axis, s in [("sx", sx), ("sy", sy)]:
        if abs(s) > REJECT_SCALE_MAX:
            return f"extreme scale {axis}={s:.4f} > {REJECT_SCALE_MAX}"
        if abs(s) < REJECT_SCALE_MIN:
            return f"near-zero scale {axis}={s:.4f} < {REJECT_SCALE_MIN}"
    return None


def flag_affine(decomp,
                max_translation=80.0,
                max_rotation=30.0,
                scale_range=(0.4, 1.6),
                max_shear=0.25):
    flags = []
    tx, ty = abs(decomp["tx"]), abs(decomp["ty"])
    if tx > max_translation or ty > max_translation:
        flags.append(f"large translation ({tx:.1f}, {ty:.1f})px > {max_translation}px")
    ang = abs(decomp["angle_deg"])
    if ang > max_rotation:
        flags.append(f"large rotation {ang:.1f}° > {max_rotation}°")
    for axis, s in [("sx", decomp["sx"]), ("sy", decomp["sy"])]:
        lo, hi = scale_range
        if not (lo <= s <= hi):
            flags.append(f"unusual scale {axis}={s:.4f} outside [{lo},{hi}]")
    if abs(decomp["shear"]) > max_shear:
        flags.append(f"high shear {decomp['shear']:.4f} > {max_shear}")
    return flags


# ── Stage 1: Warping ──────────────────────────────────────────────────────────
def warp_with_affine(img, M, h, w, flags=cv2.INTER_LINEAR):
    return cv2.warpAffine(img, M, (w, h), flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ── Stage 2: MLS — Moving Least Squares (Schaefer et al. 2006) ───────────────
def select_tps_control_points(tracks, vis, conf, conf_thresh=0.3,
                               min_points=10, max_ctrl=500):
    """
    Select control points from Stage 2 CoWTracker correspondences for TPS.

    tracks[0] = points in fixed frame        (target positions)
    tracks[1] = points in affine-warped frame (source positions to sample from)

    Returns fixed_pts, warped_pts, meta (or None, None, None if too few points).
    """
    from scipy.interpolate import RBFInterpolator  # lazy import — check available
    confidence = vis[1] * conf[1]
    mask       = confidence > conf_thresh
    n_conf     = int(mask.sum())

    if n_conf < min_points:
        print(f"  [S2] Only {n_conf} confident points — "
              f"TPS skipped, keeping affine result")
        return None, None, None

    fixed_pts  = tracks[0][mask].numpy().astype(np.float64)
    warped_pts = tracks[1][mask].numpy().astype(np.float64)

    if len(fixed_pts) > max_ctrl:
        idx        = np.random.choice(len(fixed_pts), max_ctrl, replace=False)
        fixed_pts  = fixed_pts[idx]
        warped_pts = warped_pts[idx]

    n_ctrl = len(fixed_pts)
    meta   = dict(tps_n_control_pts=n_ctrl, tps_n_confident=n_conf)
    return fixed_pts, warped_pts, meta


def compute_tps_map(fixed_pts, warped_pts, h, w, smoothing=10.0):
    """
    Compute Thin Plate Spline deformation map using RBFInterpolator.

    fixed_pts   : (N, 2) float64  control points in fixed frame
    warped_pts  : (N, 2) float64  corresponding points in affine-warped frame
    smoothing   : regularisation (higher = smoother, less overfitting)

    Returns map_x, map_y  (H, W) float32 for cv2.remap.
    """
    from scipy.interpolate import RBFInterpolator

    rbf_x = RBFInterpolator(fixed_pts, warped_pts[:, 0],
                             kernel="thin_plate_spline", smoothing=smoothing)
    rbf_y = RBFInterpolator(fixed_pts, warped_pts[:, 1],
                             kernel="thin_plate_spline", smoothing=smoothing)

    yy, xx  = np.mgrid[0:h, 0:w]
    grid    = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)

    map_x = rbf_x(grid).reshape(h, w).astype(np.float32)
    map_y = rbf_y(grid).reshape(h, w).astype(np.float32)

    # Clip to valid image bounds — prevents black border artifacts
    # caused by TPS extrapolating outside image extent
    np.clip(map_x, 0, w - 1, out=map_x)
    np.clip(map_y, 0, h - 1, out=map_y)

    return map_x, map_y


def warp_with_tps(img, map_x, map_y, is_mask=False):
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.remap(img, map_x, map_y, interp,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ── Metrics ───────────────────────────────────────────────────────────────────
def dice_score(a, b):
    inter = (a * b).sum()
    return float((2 * inter) / (a.sum() + b.sum() + 1e-8))


# ── FOV mask ──────────────────────────────────────────────────────────────────
def get_fov_mask(img):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return (mask > 0).astype(np.float32)


# ── Overlap composites ────────────────────────────────────────────────────────
def make_overlap_image(fixed_img, warped_img):
    fov = get_fov_mask(fixed_img)
    def norm(img):
        g = img.astype(np.float32).mean(axis=-1)
        mn, mx = g.min(), g.max()
        return (g - mn) / (mx - mn + 1e-8)
    f   = norm(fixed_img)
    w   = norm(warped_img)
    out = np.zeros((*f.shape, 3), dtype=np.float32)
    out[..., 0] = f
    out[..., 1] = w
    out[..., 2] = f
    out *= fov[..., np.newaxis]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def make_overlap_vessels(fixed_v, warped_v, fixed_img):
    fov = get_fov_mask(fixed_img)
    H, W = fixed_v.shape
    out  = np.zeros((H, W, 3), dtype=np.uint8)
    out[..., 0] = (warped_v * 255).astype(np.uint8)
    out[..., 1] = (fixed_v  * 255).astype(np.uint8)
    return (out * fov[..., np.newaxis]).astype(np.uint8)


# ── Visualisation helpers ─────────────────────────────────────────────────────
def mask_to_rgb(mask):
    g = (mask * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def add_label(img, text):
    out = img.copy()
    for color, thick in [((255, 255, 255), 2), ((0, 0, 0), 1)]:
        cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, thick, cv2.LINE_AA)
    return out


def make_row_grid(fixed_img, fixed_vessel,
                  moving_img, moving_vessel,
                  warped_img_affine, warped_vessel_affine,
                  warped_img_mls=None, warped_vessel_mls=None,
                  row_label=None,
                  dice_before=None, dice_affine=None, dice_mls=None,
                  mls_fallback=False,
                  decomp=None, flags=None, meta=None, meta_mls=None):
    GAP   = 6
    H     = fixed_img.shape[0]
    gap_v = np.ones((H, GAP, 3), dtype=np.uint8) * 60

    ov_affine = make_overlap_image(fixed_img, warped_img_affine)
    vv_affine = make_overlap_vessels(fixed_vessel, warped_vessel_affine, fixed_img)

    if warped_img_mls is not None:
        ov_mls    = make_overlap_image(fixed_img, warped_img_mls)
        vv_mls    = make_overlap_vessels(fixed_vessel, warped_vessel_mls, fixed_img)
        tps_label = "TPS(fallback)" if mls_fallback else "After TPS"
        panels = [
            add_label(fixed_img,                  "Fixed Image"),
            add_label(mask_to_rgb(fixed_vessel),  "Fixed Vessel"),
            add_label(moving_img,                 "Moving Image"),
            add_label(mask_to_rgb(moving_vessel), "Moving Vessel"),
            add_label(warped_img_affine,           "After Affine"),
            add_label(warped_img_mls,              tps_label),
            add_label(ov_affine,                  "Overlap Affine"),
            add_label(ov_mls,                     "Overlap MLS"),
            add_label(vv_mls,                     "Vessel Overlap"),
        ]
    else:
        panels = [
            add_label(fixed_img,                  "Fixed Image"),
            add_label(mask_to_rgb(fixed_vessel),  "Fixed Vessel"),
            add_label(moving_img,                 "Moving Image"),
            add_label(mask_to_rgb(moving_vessel), "Moving Vessel"),
            add_label(warped_img_affine,           "Registered"),
            add_label(ov_affine,                  "Fixed+Reg Overlap"),
            add_label(vv_affine,                  "Vessel Overlap"),
        ]

    row_parts = []
    for idx, panel in enumerate(panels):
        row_parts.append(panel)
        if idx < len(panels) - 1:
            row_parts.append(gap_v)
    row = np.concatenate(row_parts, axis=1)

    SIDEBAR_W = 170
    sidebar   = np.zeros((H, SIDEBAR_W, 3), dtype=np.uint8)
    sidebar[:] = (30, 30, 30)

    lines = []
    if row_label:
        lines.append(row_label)
    if dice_before is not None:
        lines.append(f"Before: {dice_before:.4f}")
    if dice_affine is not None:
        d1 = dice_affine - (dice_before or 0)
        lines.append(f"Affine: {dice_affine:.4f} ({d1:+.4f})")
    if dice_mls is not None:
        if mls_fallback and meta_mls and "tps_raw_dice" in meta_mls:
            raw   = meta_mls["tps_raw_dice"]
            d_raw = raw - (dice_affine or 0)
            lines.append(f"TPS raw:{raw:.4f} ({d_raw:+.4f})")
            lines.append(f"Final  :affine(fb)")
        else:
            d2 = dice_mls - (dice_affine or 0)
            lines.append(f"TPS    : {dice_mls:.4f} ({d2:+.4f})")
    if decomp:
        lines.append("---")
        lines.append(f"tx:{decomp['tx']:+.1f} ty:{decomp['ty']:+.1f}")
        lines.append(f"rot:{decomp['angle_deg']:+.2f}deg")
        lines.append(f"sx:{decomp['sx']:.3f} sy:{decomp['sy']:.3f}")
        lines.append(f"shear:{decomp['shear']:.3f}")
    if meta:
        lines.append(f"inlr:{meta['n_inliers']}/{meta['n_candidates']}")
        lines.append(f"ratio:{meta['inlier_ratio']:.3f}")
        if meta.get("low_conf_fallback"):
            lines.append("!low_conf_fallback")
    if meta_mls:
        lines.append(f"TPS ctrl:{meta_mls.get('tps_n_control_pts', 0)}")
        if mls_fallback:
            lines.append("!TPS fallback→affine")
    if flags:
        lines.append("WARN:")
        for f in flags:
            lines.append(f"  {f[:24]}")

    flag_color = (80, 80, 255) if flags else (200, 200, 200)
    for li, line in enumerate(lines):
        y = 18 + li * 18
        if y > H - 4:
            break
        color = flag_color if (line.startswith("WARN") or line.startswith("  ")) \
                else (200, 200, 200)
        if line.startswith("!"):
            color = (60, 200, 255)
        cv2.putText(sidebar, line, (4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    divider = np.ones((H, GAP, 3), dtype=np.uint8) * 60
    return np.concatenate([sidebar, divider, row], axis=1)


# ── Save per-pair outputs ─────────────────────────────────────────────────────
def save_outputs(out_dir, name,
                 fixed_img, fixed_vessel,
                 moving_img, moving_vessel,
                 warped_img_affine, warped_vessel_affine,
                 warped_img_mls=None, warped_vessel_mls=None,
                 dice_before=None, dice_affine=None, dice_mls=None,
                 mls_fallback=False,
                 pair_index=None, decomp=None, flags=None,
                 meta=None, meta_mls=None):

    def write_rgb(fname, img):
        cv2.imwrite(str(out_dir / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    def write_mask(fname, m):
        cv2.imwrite(str(out_dir / fname), (m * 255).astype(np.uint8))

    write_rgb( f"{name}_affine.png",        warped_img_affine)
    write_mask(f"{name}_affine_vessel.png", warped_vessel_affine)
    write_rgb( f"{name}_overlap_affine.png",
               make_overlap_image(fixed_img, warped_img_affine))

    if warped_img_mls is not None:
        suffix = "mls_raw" if mls_fallback else "mls"
        write_rgb( f"{name}_{suffix}.png",        warped_img_mls)
        write_mask(f"{name}_{suffix}_vessel.png", warped_vessel_mls)
        write_rgb( f"{name}_overlap_{suffix}.png",
                   make_overlap_image(fixed_img, warped_img_mls))
        write_rgb( f"{name}_vessels_{suffix}.png",
                   make_overlap_vessels(fixed_vessel, warped_vessel_mls, fixed_img))
    if warped_img_mls is None:
        write_rgb( f"{name}_vessels_affine.png",
                   make_overlap_vessels(fixed_vessel, warped_vessel_affine, fixed_img))

    row = make_row_grid(
        fixed_img, fixed_vessel, moving_img, moving_vessel,
        warped_img_affine, warped_vessel_affine,
        warped_img_mls=warped_img_mls, warped_vessel_mls=warped_vessel_mls,
        row_label=f"#{pair_index}" if pair_index is not None else None,
        dice_before=dice_before, dice_affine=dice_affine, dice_mls=dice_mls,
        mls_fallback=mls_fallback,
        decomp=decomp, flags=flags, meta=meta, meta_mls=meta_mls,
    )
    write_rgb(f"{name}_grid.png", row)
    extra = 6 if warped_img_mls is not None else 4
    print(f"  Saved: {name}_grid.png  (+{extra} individual files)")
    return row


# ── Summary pages ─────────────────────────────────────────────────────────────
def save_summary_page(out_dir, rows, mean_before=None,
                      mean_affine=None, mean_mls=None, rows_per_page=20):
    if not rows:
        return
    pages_dir = out_dir / "summary_pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    ROW_SEP_H = 4
    BANNER_H  = 50
    max_w     = max(r.shape[1] for r in rows)

    def pad_row(r):
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        return r

    padded  = [pad_row(r) for r in rows]
    sep     = np.full((ROW_SEP_H, max_w, 3), 100, dtype=np.uint8)
    n_pages = max(1, (len(padded) + rows_per_page - 1) // rows_per_page)

    for p in range(n_pages):
        chunk      = padded[p * rows_per_page : (p + 1) * rows_per_page]
        page_label = f"{p + 1:02d}of{n_pages:02d}"
        pairs_r    = (f"pairs {p * rows_per_page + 1}-"
                      f"{min((p + 1) * rows_per_page, len(rows))} / {len(rows)}")

        txt = f"CoWTracker FA/IR Registration  |  Page {page_label}  |  {pairs_r}"
        if mean_before is not None and mean_affine is not None:
            txt += (f"  |  Dice before:{mean_before:.4f}  "
                    f"affine:{mean_affine:.4f} ({mean_affine-mean_before:+.4f})")
        if mean_mls is not None:
            txt += f"  mls:{mean_mls:.4f} ({mean_mls-mean_affine:+.4f})"

        banner = np.zeros((BANNER_H, max_w, 3), dtype=np.uint8)
        cv2.putText(banner, txt, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 60), 1, cv2.LINE_AA)

        parts = [banner]
        for row in chunk:
            parts.append(sep)
            parts.append(row)

        page     = np.concatenate(parts, axis=0)
        out_path = pages_dir / f"summary_page_{page_label}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(page, cv2.COLOR_RGB2BGR))
        print(f"  Summary page {page_label} saved: {out_path.name}")

    print(f"\n  {n_pages} summary page(s) saved -> {pages_dir.resolve()}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="CoWTracker Retinal Registration — Affine + MLS")
    parser.add_argument("--csv",               required=True)
    parser.add_argument("--output_dir",        default=None)
    parser.add_argument("--stage",             default="affine",
                        choices=["affine", "deformable"])
    parser.add_argument("--moving_col",        default="moving")
    parser.add_argument("--fixed_col",         default="fixed")
    parser.add_argument("--moving_vessel_col", default="moving_vessel_mask")
    parser.add_argument("--fixed_vessel_col",  default="fixed_vessel_mask")
    parser.add_argument("--height",            type=int,   default=TARGET_H)
    parser.add_argument("--width",             type=int,   default=TARGET_W)
    parser.add_argument("--conf_thresh",       type=float, default=0.3)
    parser.add_argument("--min_points",        type=int,   default=6)
    parser.add_argument("--max_translation",   type=float, default=80.0)
    parser.add_argument("--max_rotation",      type=float, default=30.0)
    parser.add_argument("--scale_lo",          type=float, default=0.4)
    parser.add_argument("--scale_hi",          type=float, default=1.6)
    parser.add_argument("--max_shear",         type=float, default=0.25)
    parser.add_argument("--min_inlier_ratio",  type=float, default=0.05)
    # TPS (Stage 2)
    parser.add_argument("--tps_conf_thresh",   type=float, default=0.3)
    parser.add_argument("--tps_min_points",    type=int,   default=10)
    parser.add_argument("--tps_max_ctrl",      type=int,   default=500)
    parser.add_argument("--tps_smoothing",     type=float, default=10.0,
                        help="TPS regularisation. Higher = smoother, less overfitting.")
    parser.add_argument("--tps_skip_thresh",   type=float, default=0.20,
                        help="Skip TPS if affine Dice already >= this value. "
                             "Set to 1.0 to disable.")
    args = parser.parse_args()

    assert args.height % 14 == 0 and args.width % 14 == 0

    if args.output_dir is None:
        args.output_dir = ("Results_deformable" if args.stage == "deformable"
                           else "Results_affine")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"CoWTracker Retinal Registration  [Stage: {args.stage.upper()}]")
    print("=" * 65)
    print(f"Device      : {DEVICE}")
    print(f"Size        : {args.height} x {args.width}")
    print(f"Output      : {out_dir.resolve()}")
    print(f"Soft flags  : translation>{args.max_translation}px  "
          f"rotation>{args.max_rotation}°  "
          f"scale [{args.scale_lo},{args.scale_hi}]  shear>{args.max_shear}")
    print(f"Hard reject : |rot|>{REJECT_ABS_ROT}°  "
          f"scale >{REJECT_SCALE_MAX}/<{REJECT_SCALE_MIN}  "
          f"inlier_ratio<{args.min_inlier_ratio}")
    if args.stage == "deformable":
        print(f"TPS         : conf>{args.tps_conf_thresh}  "
              f"max_ctrl={args.tps_max_ctrl}  smoothing={args.tps_smoothing}  "
              f"skip_thresh={args.tps_skip_thresh}  "
              f"[auto-fallback to affine if TPS < affine]")
    print()

    print("Loading CoWTracker model...")
    model = CoWTracker.from_checkpoint(device=DEVICE, dtype=INF_DTYPE)
    print("Model ready.\n")

    df = pd.read_csv(args.csv)
    print(f"Found {len(df)} image pairs.\n")

    results  = []
    all_rows = []

    for i, row in df.iterrows():
        moving_path    = row[args.moving_col]
        fixed_path     = row[args.fixed_col]
        mv_vessel_path = row[args.moving_vessel_col]
        fx_vessel_path = row[args.fixed_vessel_col]
        name = (f"{i:04d}_{Path(moving_path).stem}"
                f"_to_{Path(fixed_path).stem}")

        print(f"[{i+1}/{len(df)}] {Path(moving_path).name} -> "
              f"{Path(fixed_path).name}")

        try:
            fixed_img     = load_image(fixed_path,      args.height, args.width)
            moving_img    = load_image(moving_path,     args.height, args.width)
            fixed_vessel  = load_vessel(fx_vessel_path, args.height, args.width)
            moving_vessel = load_vessel(mv_vessel_path, args.height, args.width)

            # ── Stage 1: Affine ───────────────────────────────────────────────
            tracks, vis, conf = run_cowtracker(model, fixed_vessel, moving_vessel)

            M, decomp, meta, reject_reason = estimate_affine(
                tracks, vis, conf,
                conf_thresh=args.conf_thresh,
                min_points=args.min_points,
                min_inlier_ratio=args.min_inlier_ratio,
            )

            if M is None:
                print(f"  Skipping — {reject_reason}")
                results.append(dict(name=name, moving=moving_path,
                                    fixed=fixed_path, status="degenerate",
                                    error=reject_reason))
                continue

            flags = flag_affine(decomp,
                                max_translation=args.max_translation,
                                max_rotation=args.max_rotation,
                                scale_range=(args.scale_lo, args.scale_hi),
                                max_shear=args.max_shear)
            if flags:
                print(f"  ⚠ Flags: {', '.join(flags)}")

            h, w = args.height, args.width
            warped_img_affine = cv2.cvtColor(
                warp_with_affine(
                    cv2.cvtColor(moving_img, cv2.COLOR_RGB2BGR), M, h, w),
                cv2.COLOR_BGR2RGB)
            warped_vessel_affine = (
                warp_with_affine((moving_vessel * 255).astype(np.uint8),
                                 M, h, w, flags=cv2.INTER_NEAREST) > 127
            ).astype(np.float32)

            fov         = get_fov_mask(fixed_img)
            dice_before = dice_score(moving_vessel        * fov, fixed_vessel * fov)
            dice_affine = dice_score(warped_vessel_affine  * fov, fixed_vessel * fov)

            print(f"  Dice before:{dice_before:.4f}  "
                  f"after affine:{dice_affine:.4f}  "
                  f"({dice_affine - dice_before:+.4f})"
                  f"{'  ⚠ REGRESSED' if dice_affine < dice_before else ''}")

            # ── Stage 2: TPS ──────────────────────────────────────────────────
            warped_img_tps    = None
            warped_vessel_tps = None
            dice_tps          = None
            meta_tps          = None
            tps_fallback      = False

            if args.stage == "deformable":
                # ── Skip TPS if affine already good enough ────────────────
                if dice_affine >= args.tps_skip_thresh:
                    print(f"  [S2] Skipping TPS — affine Dice {dice_affine:.4f} "
                          f">= skip_thresh {args.tps_skip_thresh}")
                    warped_img_tps    = warped_img_affine.copy()
                    warped_vessel_tps = warped_vessel_affine.copy()
                    dice_tps          = dice_affine
                    tps_fallback      = True
                    meta_tps          = {"tps_n_control_pts": 0,
                                         "tps_n_confident": 0,
                                         "tps_raw_dice": dice_affine,
                                         "tps_skipped": True}
                else:
                    tracks2, vis2, conf2 = run_cowtracker(
                        model, fixed_vessel, warped_vessel_affine)

                    fixed_pts, warped_pts, meta_tps = select_tps_control_points(
                        tracks2, vis2, conf2,
                        conf_thresh=args.tps_conf_thresh,
                        min_points=args.tps_min_points,
                        max_ctrl=args.tps_max_ctrl,
                    )

                    if fixed_pts is not None:
                        print(f"  [S2] Computing TPS map "
                              f"({meta_tps['tps_n_control_pts']} ctrl pts, "
                              f"smoothing={args.tps_smoothing})...")

                        map_x, map_y = compute_tps_map(
                            fixed_pts, warped_pts, h, w,
                            smoothing=args.tps_smoothing,
                        )

                        # Always compute raw TPS warp for grid display
                        _warp_img = cv2.cvtColor(
                            warp_with_tps(
                                cv2.cvtColor(warped_img_affine, cv2.COLOR_RGB2BGR),
                                map_x, map_y),
                            cv2.COLOR_BGR2RGB)
                        _warp_v = (
                            warp_with_tps(
                                (warped_vessel_affine * 255).astype(np.uint8),
                                map_x, map_y, is_mask=True) > 127
                        ).astype(np.float32)
                        _dice_tps = dice_score(_warp_v * fov, fixed_vessel * fov)

                        # Grid always shows raw TPS result
                        warped_img_tps    = _warp_img
                        warped_vessel_tps = _warp_v
                        meta_tps["tps_raw_dice"] = round(_dice_tps, 4)

                        if _dice_tps >= dice_affine:
                            # TPS improved — use as final result
                            dice_tps     = _dice_tps
                            tps_fallback = False
                            print(f"  Dice after TPS: {dice_tps:.4f}  "
                                  f"({dice_tps - dice_affine:+.4f} vs affine)")
                        else:
                            # TPS regressed — revert to affine,
                            # grid still shows raw TPS so you can see why
                            dice_tps     = dice_affine
                            tps_fallback = True
                            print(f"  TPS {_dice_tps:.4f} < affine {dice_affine:.4f} "
                                  f"— fallback to affine (grid shows raw TPS)")
                    else:
                        # Not enough control points — duplicate affine for grid
                        warped_img_tps    = warped_img_affine.copy()
                        warped_vessel_tps = warped_vessel_affine.copy()
                        dice_tps          = dice_affine
                        tps_fallback      = True
                        if meta_tps is None:
                            meta_tps = {}
                        meta_tps["tps_raw_dice"] = dice_affine

            row_grid = save_outputs(
                out_dir, name,
                fixed_img, fixed_vessel, moving_img, moving_vessel,
                warped_img_affine, warped_vessel_affine,
                warped_img_mls=warped_img_tps if args.stage == "deformable" else None,
                warped_vessel_mls=warped_vessel_tps if args.stage == "deformable" else None,
                dice_before=dice_before, dice_affine=dice_affine,
                dice_mls=dice_tps if args.stage == "deformable" else None,
                mls_fallback=tps_fallback if args.stage == "deformable" else False,
                pair_index=i + 1,
                decomp=decomp, flags=flags, meta=meta,
                meta_mls=meta_tps if args.stage == "deformable" else None,
            )
            all_rows.append(row_grid)

            rec = dict(
                name=name,
                moving=moving_path,
                fixed=fixed_path,
                dice_before=round(dice_before,  4),
                dice_affine=round(dice_affine,  4),
                delta_affine=round(dice_affine - dice_before, 4),
                dice_regressed_affine=(dice_affine < dice_before),
                tx=round(decomp["tx"],          2),
                ty=round(decomp["ty"],          2),
                angle_deg=round(decomp["angle_deg"], 4),
                sx=round(decomp["sx"],          4),
                sy=round(decomp["sy"],          4),
                shear=round(decomp["shear"],    4),
                n_candidates=meta["n_candidates"],
                n_inliers=meta["n_inliers"],
                inlier_ratio=meta["inlier_ratio"],
                low_conf_fallback=meta["low_conf_fallback"],
                flagged=bool(flags),
                flag_reasons="; ".join(flags) if flags else "",
                status="ok",
            )
            if args.stage == "deformable" and dice_tps is not None:
                dice_tps_raw = (meta_tps.get("tps_raw_dice", dice_tps)
                                if meta_tps else dice_tps)
                rec["dice_tps"]          = round(dice_tps, 4)
                rec["dice_tps_raw"]      = round(dice_tps_raw, 4)
                rec["delta_tps"]         = round(dice_tps - dice_affine, 4)
                rec["delta_total"]       = round(dice_tps - dice_before, 4)
                rec["tps_fallback"]      = tps_fallback
                rec["tps_n_ctrl"]        = (meta_tps.get("tps_n_control_pts", 0)
                                            if meta_tps else 0)
                rec["tps_skipped"]       = meta_tps.get("tps_skipped", False) \
                                           if meta_tps else False
            results.append(rec)

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(dict(name=name, moving=moving_path,
                                fixed=fixed_path, status="error", error=str(e)))

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "results.csv", index=False)

    ok         = results_df[results_df["status"] == "ok"]
    degenerate = results_df[results_df["status"] == "degenerate"]
    errors     = results_df[results_df["status"] == "error"]

    mean_before = ok["dice_before"].mean() if len(ok) else None
    mean_affine = ok["dice_affine"].mean()  if len(ok) else None
    mean_tps    = ok["dice_tps"].mean()     if (len(ok) and "dice_tps" in ok.columns) else None

    if all_rows:
        save_summary_page(out_dir, all_rows,
                          mean_before=mean_before,
                          mean_affine=mean_affine,
                          mean_mls=mean_tps)

    flagged  = ok[ok["flagged"] == True] if "flagged" in ok.columns else pd.DataFrame()
    reg_aff  = ok[ok["dice_regressed_affine"] == True] if "dice_regressed_affine" in ok.columns else pd.DataFrame()
    fallback = ok[ok["low_conf_fallback"] == True] if "low_conf_fallback" in ok.columns else pd.DataFrame()

    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    print(f"  Total pairs        : {len(df)}")
    print(f"  OK                 : {len(ok)}")
    print(f"  Degenerate/skip    : {len(degenerate)}")
    print(f"  File errors        : {len(errors)}")
    if len(ok):
        print(f"  Flagged (soft)     : {len(flagged)}")
        print(f"  Regressed (affine) : {len(reg_aff)}")
        print(f"  Low-conf fallback  : {len(fallback)}")
        print(f"  Mean Dice before   : {mean_before:.4f}")
        print(f"  Mean Dice affine   : {mean_affine:.4f}  "
              f"({mean_affine - mean_before:+.4f})")
        if mean_tps is not None:
            tps_fb   = ok[ok["tps_fallback"] == True] if "tps_fallback" in ok.columns else pd.DataFrame()
            tps_skip = ok[ok["tps_skipped"]  == True] if "tps_skipped"  in ok.columns else pd.DataFrame()
            tps_imp  = ok[(ok["tps_fallback"] == False) & (ok["tps_n_ctrl"] > 0)] if "tps_fallback" in ok.columns else pd.DataFrame()
            print(f"  Mean Dice TPS      : {mean_tps:.4f}  "
                  f"({mean_tps - mean_affine:+.4f} vs affine)")
            print(f"  TPS improved       : {len(tps_imp)}")
            print(f"  TPS→affine fallback: {len(tps_fb) - len(tps_skip)}")
            print(f"  TPS skipped (Dice>={args.tps_skip_thresh}): {len(tps_skip)}")
        print(f"  Mean rotation      : {ok['angle_deg'].mean():.2f}°")
        print(f"  Mean translation   : "
              f"tx={ok['tx'].mean():.1f}px  ty={ok['ty'].mean():.1f}px")
        print(f"  Mean inlier ratio  : {ok['inlier_ratio'].mean():.3f}")
    print(f"  Results saved      : {out_dir.resolve()}/results.csv")
    print(f"  Summary pages      : {out_dir.resolve()}/summary_pages/")
    print("=" * 65)


if __name__ == "__main__":
    main()