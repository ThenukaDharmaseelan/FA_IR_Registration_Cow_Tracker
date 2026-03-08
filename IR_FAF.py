import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import griddata

from cowtracker import CoWTracker

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
INF_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
TARGET_H  = 224
TARGET_W  = 224


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
    dist = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)
    dist = (dist / dist.max() * 255).astype(np.uint8) if dist.max() > 0 \
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
    tracks = predictions["track"][0].cpu()
    vis    = predictions["vis"][0].cpu()
    conf   = predictions["conf"][0].cpu()
    return tracks, vis, conf


# ── Dense flow directly from CoWTracker tracks ───────────────────────────────
def build_dense_flow(tracks, vis, conf, h, w, conf_thresh=0.3, min_points=20):
    confidence = vis[1] * conf[1]
    mask       = confidence > conf_thresh

    if mask.sum() < min_points:
        print(f"  Warning: only {int(mask.sum())} confident points "
              f"(need {min_points}), lowering threshold...")
        if confidence.max() > 0:
            conf_thresh = float(confidence[confidence > 0].quantile(0.5))
        mask = confidence > conf_thresh
        if mask.sum() < 4:
            print(f"  ERROR: still only {int(mask.sum())} points — cannot warp")
            return None, None

    src = tracks[0][mask].numpy()
    dst = tracks[1][mask].numpy()

    print(f"  Building dense flow from {len(src)} CoWTracker correspondences "
          f"(conf>{conf_thresh:.2f})")

    disp_x = dst[:, 0] - src[:, 0]
    disp_y = dst[:, 1] - src[:, 1]

    points   = src[:, ::-1].astype(np.float32)
    grid_r, grid_c = np.mgrid[0:h, 0:w]
    grid_pts = np.stack([grid_r.ravel(), grid_c.ravel()], axis=-1).astype(np.float32)

    dx_dense = griddata(points, disp_x, grid_pts, method="linear").reshape(h, w)
    dy_dense = griddata(points, disp_y, grid_pts, method="linear").reshape(h, w)

    nan_mask = np.isnan(dx_dense)
    if nan_mask.any():
        dx_nn = griddata(points, disp_x, grid_pts, method="nearest").reshape(h, w)
        dy_nn = griddata(points, disp_y, grid_pts, method="nearest").reshape(h, w)
        dx_dense[nan_mask] = dx_nn[nan_mask]
        dy_dense[nan_mask] = dy_nn[nan_mask]

    map_x = (grid_c + dx_dense).astype(np.float32)
    map_y = (grid_r + dy_dense).astype(np.float32)
    return map_x, map_y


def warp_with_flow(img, map_x, map_y, flags=cv2.INTER_LINEAR):
    return cv2.remap(img, map_x, map_y, interpolation=flags,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ── Metrics ───────────────────────────────────────────────────────────────────
def dice_score(a, b):
    inter = (a * b).sum()
    return float((2 * inter) / (a.sum() + b.sum() + 1e-8))


# ── FOV mask ──────────────────────────────────────────────────────────────────
def get_fov_mask(img):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return (mask > 0).astype(np.float32)


# ── Overlap composites ────────────────────────────────────────────────────────
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
    out[..., 1] = (fixed_v  * 255).astype(np.uint8)
    out = (out * fov[..., np.newaxis]).astype(np.uint8)
    return out


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


def make_row_grid(fixed_img, fixed_vessel, moving_img, moving_vessel,
                  warped_img, warped_vessel,
                  row_label=None, dice_before=None, dice_after=None):
    GAP   = 6
    H     = fixed_img.shape[0]
    gap_v = np.ones((H, GAP, 3), dtype=np.uint8) * 60
    ov_img    = make_overlap_image(fixed_img, warped_img)
    ov_vessel = make_overlap_vessels(fixed_vessel, warped_vessel, fixed_img)
    panels = [
        add_label(fixed_img,                  "Fixed Image"),
        add_label(mask_to_rgb(fixed_vessel),  "Fixed Vessel"),
        add_label(moving_img,                 "Moving Image"),
        add_label(mask_to_rgb(moving_vessel), "Moving Vessel"),
        add_label(warped_img,                 "Registered"),
        add_label(ov_img,                     "Fixed+Reg Overlap"),
        add_label(ov_vessel,                  "Vessel Overlap"),
    ]
    row_parts = []
    for idx, panel in enumerate(panels):
        row_parts.append(panel)
        if idx < len(panels) - 1:
            row_parts.append(gap_v)
    row = np.concatenate(row_parts, axis=1)

    if row_label is not None or (dice_before is not None and dice_after is not None):
        SIDEBAR_W = 120
        sidebar   = np.zeros((H, SIDEBAR_W, 3), dtype=np.uint8)
        sidebar[:] = (30, 30, 30)
        lines = []
        if row_label is not None:
            lines.append(row_label)
        if dice_before is not None:
            lines.append(f"Before:{dice_before:.4f}")
        if dice_after is not None:
            delta = dice_after - dice_before if dice_before is not None else 0
            arrow = "+" if delta >= 0 else ""
            lines.append(f"After: {dice_after:.4f}")
            lines.append(f"Delta:{arrow}{delta:.4f}")
        for li, line in enumerate(lines):
            y = 22 + li * 22
            cv2.putText(sidebar, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (200, 200, 200), 1, cv2.LINE_AA)
        divider = np.ones((H, GAP, 3), dtype=np.uint8) * 60
        row = np.concatenate([sidebar, divider, row], axis=1)
    return row


# ── Save per-pair outputs ─────────────────────────────────────────────────────
def save_outputs(out_dir, name, fixed_img, fixed_vessel, moving_img, moving_vessel,
                 warped_img, warped_vessel, dice_before=None, dice_after=None, pair_index=None):
    def write_rgb(fname, img):
        cv2.imwrite(str(out_dir / fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    def write_mask(fname, m):
        cv2.imwrite(str(out_dir / fname), (m * 255).astype(np.uint8))

    write_rgb( f"{name}_registered.png",        warped_img)
    write_mask(f"{name}_registered_vessel.png", warped_vessel)
    write_rgb( f"{name}_overlap_image.png",     make_overlap_image(fixed_img, warped_img))
    write_rgb( f"{name}_overlap_vessels.png",   make_overlap_vessels(fixed_vessel, warped_vessel, fixed_img))

    row_label = f"#{pair_index}" if pair_index is not None else None
    row = make_row_grid(fixed_img, fixed_vessel, moving_img, moving_vessel,
                        warped_img, warped_vessel,
                        row_label=row_label, dice_before=dice_before, dice_after=dice_after)
    write_rgb(f"{name}_grid.png", row)
    print(f"  Saved: {name}_grid.png  (+4 individual files)")
    return row


# ── Summary pages (paginated, 20 rows per page) ───────────────────────────────
def save_summary_pages(out_dir, rows, rows_per_page=20,
                       mean_before=None, mean_after=None):
    """
    Splits all row grids into chunks of `rows_per_page` and saves each chunk
    as a separate summary PNG inside FAF_IR (no subfolders):

        <out_dir>/FAF_IR/summary_page_001.png
        <out_dir>/FAF_IR/summary_page_002.png
        ...

    Also saves a master index.txt listing all page paths.
    """
    if not rows:
        return

    ROW_SEP_H = 4
    sep_color = (100, 100, 100)

    max_w = max(r.shape[1] for r in rows)

    def pad_row(r):
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        return r

    sep       = np.full((ROW_SEP_H, max_w, 3), sep_color, dtype=np.uint8)
    pages_dir = out_dir / "IR_FAF"
    pages_dir.mkdir(parents=True, exist_ok=True)

    total_pages = (len(rows) + rows_per_page - 1) // rows_per_page
    saved_paths = []

    for page_idx in range(total_pages):
        start = page_idx * rows_per_page
        end   = min(start + rows_per_page, len(rows))
        chunk = [pad_row(rows[i]) for i in range(start, end)]

        page_num = page_idx + 1

        # ── Header banner ─────────────────────────────────────────────────────
        BANNER_H   = 50
        banner     = np.zeros((BANNER_H, max_w, 3), dtype=np.uint8)
        banner_txt = (f"CoWTracker Retinal Registration  |  "
                      f"Page {page_num}/{total_pages}  |  "
                      f"Rows {start+1}–{end} of {len(rows)}")
        if mean_before is not None and mean_after is not None:
            banner_txt += (f"   |   Mean Dice  before:{mean_before:.4f} "
                           f"after:{mean_after:.4f} "
                           f"delta:{mean_after - mean_before:+.4f}")
        cv2.putText(banner, banner_txt, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 220, 60), 1, cv2.LINE_AA)

        parts = [banner]
        for row in chunk:
            parts.append(sep)
            parts.append(row)

        page_img = np.concatenate(parts, axis=0)
        out_path = pages_dir / f"summary_page_{page_num:03d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(page_img, cv2.COLOR_RGB2BGR))
        saved_paths.append(out_path)
        print(f"  Summary page {page_num}/{total_pages}: {out_path.resolve()}")

    # ── Master index ──────────────────────────────────────────────────────────
    index_path = pages_dir / "index.txt"
    with open(index_path, "w") as f:
        f.write(f"Total pages : {total_pages}\n")
        f.write(f"Rows/page   : {rows_per_page}\n")
        f.write(f"Total rows  : {len(rows)}\n\n")
        for p in saved_paths:
            f.write(str(p) + "\n")
    print(f"  Index written : {index_path.resolve()}")

    return saved_paths


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="CoWTracker Retinal Registration — dense warp from tracks only")
    parser.add_argument("--csv",               required=True)
    parser.add_argument("--output_dir",        default="IR_FAF_outputs")
    parser.add_argument("--moving_col",        default="fixed")
    parser.add_argument("--fixed_col",         default="moving")
    parser.add_argument("--moving_vessel_col", default="fixed_vessel_mask")
    parser.add_argument("--fixed_vessel_col",  default="moving_vessel_mask")
    parser.add_argument("--height",       type=int,   default=TARGET_H)
    parser.add_argument("--width",        type=int,   default=TARGET_W)
    parser.add_argument("--conf_thresh",  type=float, default=0.3)
    parser.add_argument("--min_points",   type=int,   default=20)
    parser.add_argument("--rows_per_page", type=int,  default=20,
                        help="Number of image rows per summary page (default: 20)")
    args = parser.parse_args()

    assert args.height % 14 == 0, f"Height must be multiple of 14, got {args.height}"
    assert args.width  % 14 == 0, f"Width must be multiple of 14, got {args.width}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CoWTracker Retinal Registration  [no homography / no TPS]")
    print("=" * 60)
    print(f"Device       : {DEVICE}")
    print(f"Size         : {args.height} x {args.width}")
    print(f"Rows/page    : {args.rows_per_page}")
    print(f"Output       : {out_dir.resolve()}\n")

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

        name = f"{i:04d}_{Path(moving_path).stem}_to_{Path(fixed_path).stem}"
        print(f"[{i+1}/{len(df)}] {Path(moving_path).name} -> {Path(fixed_path).name}")

        try:
            h, w = args.height, args.width
            fixed_img     = load_image(fixed_path,      h, w)
            moving_img    = load_image(moving_path,     h, w)
            fixed_vessel  = load_vessel(fx_vessel_path, h, w)
            moving_vessel = load_vessel(mv_vessel_path, h, w)

            tracks, vis, conf = run_cowtracker(model, fixed_vessel, moving_vessel)

            map_x, map_y = build_dense_flow(
                tracks, vis, conf, h, w,
                conf_thresh=args.conf_thresh,
                min_points=args.min_points,
            )

            if map_x is None:
                print("  Skipping — could not build dense flow")
                results.append(dict(name=name, moving=moving_path,
                                    fixed=fixed_path, status="skip",
                                    error="dense flow failed"))
                continue

            warped_img = warp_with_flow(
                cv2.cvtColor(moving_img, cv2.COLOR_RGB2BGR), map_x, map_y)
            warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)

            warped_vessel = warp_with_flow(
                (moving_vessel * 255).astype(np.uint8), map_x, map_y,
                flags=cv2.INTER_NEAREST)
            warped_vessel = (warped_vessel > 127).astype(np.float32)

            fov         = get_fov_mask(fixed_img)
            dice_before = dice_score(moving_vessel * fov, fixed_vessel * fov)
            dice_after  = dice_score(warped_vessel * fov, fixed_vessel * fov)
            delta       = dice_after - dice_before
            arrow       = "▲" if delta >= 0 else "▼"
            print(f"  Dice before: {dice_before:.4f}  after: {dice_after:.4f}  "
                  f"{arrow}{abs(delta):.4f}")

            row_grid = save_outputs(
                out_dir, name,
                fixed_img,  fixed_vessel,
                moving_img, moving_vessel,
                warped_img, warped_vessel,
                dice_before=dice_before,
                dice_after=dice_after,
                pair_index=i + 1,
            )
            all_rows.append(row_grid)

            results.append(dict(
                name=name,
                moving=moving_path,
                fixed=fixed_path,
                dice_before=round(dice_before, 4),
                dice_after=round(dice_after,   4),
                dice_delta=round(delta,         4),
                status="ok",
            ))

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(dict(name=name, moving=moving_path,
                                fixed=fixed_path, status="error", error=str(e)))

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "results.csv", index=False)

    ok = results_df[results_df["status"] == "ok"]

    if all_rows:
        mean_before = ok["dice_before"].mean() if len(ok) else None
        mean_after  = ok["dice_after"].mean()  if len(ok) else None
        save_summary_pages(out_dir, all_rows,
                           rows_per_page=args.rows_per_page,
                           mean_before=mean_before,
                           mean_after=mean_after)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Pairs OK        : {len(ok)} / {len(df)}")
    if len(ok):
        print(f"  Mean Dice before: {ok['dice_before'].mean():.4f}")
        print(f"  Mean Dice after : {ok['dice_after'].mean():.4f}")
        print(f"  Mean Dice delta : {ok['dice_delta'].mean():+.4f}")
    print(f"  Results saved   : {out_dir.resolve()}/results.csv")
    print(f"  Summary pages   : {out_dir.resolve()}/FAF_IR/")
    print("=" * 60)


if __name__ == "__main__":
    main()

