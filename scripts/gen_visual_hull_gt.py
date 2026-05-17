#!/usr/bin/env python3
"""
יוצר GT מסוג visual hull occupancy עבור דגמי מטוסים של ShapeNet R2N2.

עבור כל נקודת גריד תלת-ממדית, מקרין לכל המבטים ובודק אם היא נמצאת בתוך המסכה.
נקודה נחשבת "בפנים" אם היא בתוך המסכה בלפחות min_views מבטים.

לא נדרשת רשת משולשים — משתמש רק ברינדורים, מסכות ופרמטרי מצלמה שכבר קיימים בנתוני R2N2.
שומר תוצאה כ-{model_id}_hull_{resolution}.npy (מערך בוליאני בגודל=(R,R,R)).

שימוש:
  python scripts/gen_visual_hull_gt.py \
      --data-root data_shapenet \
      --resolution 128 \
      --min-views 6 \
      --workers 4
"""

import sys, argparse, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed


# ── כלי מצלמה (אותה קונבנציה כמו shapenet_r2n2_dataset.py) ─────────────────

def _parse_metadata(path):
    cameras = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            az, el, tilt, dist, fov = (float(x) for x in parts[:5])
            cameras.append({"azimuth": az, "elevation": el, "tilt": tilt,
                             "distance": dist, "fov": fov})
    return cameras


def _camera_to_matrices(cam, img_h=137, img_w=137):
    az   = math.radians(cam["azimuth"])
    el   = math.radians(cam["elevation"])
    tilt = math.radians(cam["tilt"])
    dist = cam["distance"]
    fov  = math.radians(cam["fov"])

    cx_w = dist * math.cos(el) * math.sin(az)
    cy_w = dist * math.sin(el)
    cz_w = dist * math.cos(el) * math.cos(az)
    cam_pos = np.array([cx_w, cy_w, cz_w], dtype=np.float32)

    z_axis = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)
    up = np.array([0., 1., 0.], dtype=np.float32)
    x_axis = np.cross(z_axis, up)
    xn = np.linalg.norm(x_axis)
    if xn < 1e-6:
        up = np.array([0., 0., 1.], dtype=np.float32)
        x_axis = np.cross(z_axis, up)
        xn = np.linalg.norm(x_axis)
    x_axis /= xn
    y_axis = np.cross(x_axis, z_axis)

    if abs(tilt) > 1e-6:
        ct, st = math.cos(tilt), math.sin(tilt)
        x_axis, y_axis = ct * x_axis + st * y_axis, -st * x_axis + ct * y_axis

    R = np.stack([x_axis, y_axis, z_axis], axis=0).astype(np.float32)
    t = -R @ cam_pos

    f = (img_w / 2.0) / math.tan(fov / 2.0)
    K = np.array([[f, 0., img_w/2.], [0., f, img_h/2.], [0., 0., 1.]], dtype=np.float32)
    return K, R, t


# ── הול חזותי עבור דגם יחיד ───────────────────────────────────────────────────

def compute_visual_hull(model_id, render_dir, resolution, min_views, n_views=24):
    """
    מחזיר מערך בוליאני בגודל (R, R, R): True = בתוך האובייקט.
    משתמש בכל המבטים הזמינים (עד n_views).
    """
    rend_path = render_dir / model_id / "rendering"
    meta_path = rend_path / "rendering_metadata.txt"
    cameras = _parse_metadata(meta_path)
    n_avail = min(len(cameras), n_views)

    # בניית גריד תלת-ממדי צפוף ב-[-0.5, 0.5]^3
    R = resolution
    lin = np.linspace(-0.5, 0.5, R, dtype=np.float32)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (R^3, 3)

    inside_count = np.zeros(len(pts), dtype=np.int16)

    for vi in range(n_avail):
        img_path = rend_path / f"{vi:02d}.png"
        img = Image.open(img_path).convert("RGBA")
        W, H = img.size
        mask = (np.array(img)[..., 3] > 127).astype(np.uint8)  # (H, W)

        K, Rmat, t = _camera_to_matrices(cameras[vi], img_h=H, img_w=W)

        # המרת נקודות למערכת מצלמה: p_cam = R @ p + t
        p_cam = pts @ Rmat.T + t[None, :]  # (N, 3)

        # לקחת רק נקודות שנמצאות לפני המצלמה
        z = p_cam[:, 2]
        valid = z > 0.01

        # הקרנה לקואורדינטות פיקסלים
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u = np.where(valid, fx * (p_cam[:, 0] / np.where(valid, z, 1.)) + cx, -1.)
        v = np.where(valid, fy * (p_cam[:, 1] / np.where(valid, z, 1.)) + cy, -1.)

        # חיפוש פיקסל (שכן קרוב ביותר)
        ui = np.round(u).astype(np.int32)
        vi_ = np.round(v).astype(np.int32)

        in_bounds = valid & (ui >= 0) & (ui < W) & (vi_ >= 0) & (vi_ < H)
        # לנקודות בתוך גבולות התמונה, בדיקה מול המסכה
        ui_safe  = np.clip(ui,  0, W - 1)
        vi_safe  = np.clip(vi_, 0, H - 1)
        in_mask  = in_bounds & (mask[vi_safe, ui_safe] > 0)
        inside_count += in_mask.astype(np.int16)

    occupied = inside_count >= min_views
    return occupied.reshape(R, R, R)


# ── פונקציית עובד (לריבוי תהליכים) ───────────────────────────────────────────

def _worker(args):
    model_id, render_dir, out_dir, resolution, min_views = args
    out_path = out_dir / f"{model_id}_hull_{resolution}.npy"
    if out_path.exists():
        return model_id, "skip"
    try:
        hull = compute_visual_hull(model_id, render_dir, resolution, min_views)
        np.save(out_path, hull)
        n_occ = int(hull.sum())
        return model_id, f"ok ({n_occ} voxels)"
    except Exception as e:
        return model_id, f"ERROR: {e}"


# ── ראשי ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",   default="data_shapenet")
    parser.add_argument("--category",    default="02691156")
    parser.add_argument("--resolution",  type=int, default=128)
    parser.add_argument("--min-views",   type=int, default=6,
                        help="Point is inside if visible in >= this many views")
    parser.add_argument("--n-views",     type=int, default=24,
                        help="How many rendered views to use (max 24)")
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--limit",       type=int, default=0,
                        help="Process only first N models (0=all, for testing)")
    args = parser.parse_args()

    root       = Path(args.data_root)
    render_dir = root / "ShapeNetRendering" / args.category
    out_dir    = root / "ShapeNetHull" / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    model_ids = sorted(m.name for m in render_dir.iterdir() if m.is_dir())
    if args.limit > 0:
        model_ids = model_ids[:args.limit]

    print(f"[INFO] {len(model_ids)} models | resolution={args.resolution}³ | "
          f"min_views={args.min_views} | workers={args.workers}")

    tasks = [(mid, render_dir, out_dir, args.resolution, args.min_views)
             for mid in model_ids]

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            mid, status = fut.result()
            done += 1
            if done % 50 == 0 or done <= 5:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(tasks) - done) / max(rate, 1e-6)
                print(f"[{done}/{len(tasks)}] {mid[:16]} → {status} | "
                      f"{rate:.1f} m/s | ETA {eta/60:.1f} min", flush=True)
            elif "ERROR" in status:
                print(f"[{done}/{len(tasks)}] {mid[:16]} → {status}", flush=True)

    print(f"[DONE] {len(tasks)} models in {(time.time()-t0)/60:.1f} min")
    print(f"[OUT]  {out_dir}")


if __name__ == "__main__":
    main()
