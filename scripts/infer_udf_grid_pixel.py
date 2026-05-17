#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

from src.mv_model_occ_pixel import PixelAlignedOccupancyNet


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_ply_ascii(path: Path, points: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def load_sample(sample_dir: Path, device: torch.device):
    images = np.load(sample_dir / "images.npy").astype(np.float32) / 255.0
    masks = (np.load(sample_dir / "masks.npy") > 0).astype(np.float32)
    K = np.load(sample_dir / "K.npy").astype(np.float32)
    T_path = sample_dir / "T_w2c_norm.npy"
    if not T_path.exists():
        T_path = sample_dir / "T_w2c.npy"
    T_w2c = np.load(T_path).astype(np.float32)

    images_t = torch.from_numpy(images).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    masks_t = torch.from_numpy(masks).unsqueeze(1).unsqueeze(0).to(device)
    K_t = torch.from_numpy(K).unsqueeze(0).to(device)
    T_t = torch.from_numpy(T_w2c).unsqueeze(0).to(device)
    return images_t, masks_t, K_t, T_t


def robust_bbox_from_points(points: np.ndarray, low_q: float, high_q: float, pad_frac: float):
    q_low = np.quantile(points, low_q, axis=0).astype(np.float32)
    q_high = np.quantile(points, high_q, axis=0).astype(np.float32)
    extent = q_high - q_low
    pad = np.maximum(extent * pad_frac, 1e-3).astype(np.float32)
    return q_low - pad, q_high + pad


def make_grid_from_bbox(bmin: np.ndarray, bmax: np.ndarray, resolution: int):
    xs = np.linspace(bmin[0], bmax[0], resolution, dtype=np.float32)
    ys = np.linspace(bmin[1], bmax[1], resolution, dtype=np.float32)
    zs = np.linspace(bmin[2], bmax[2], resolution, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)


def largest_component_mask_3d(mask: np.ndarray):
    n = mask.shape[0]
    labels = np.zeros(mask.shape, dtype=np.int32)
    sizes = []
    comp_id = 0
    flat_mask = mask.reshape(-1)
    flat_labels = labels.reshape(-1)
    nn = n * n
    for seed in np.flatnonzero(flat_mask):
        if flat_labels[seed] != 0:
            continue
        comp_id += 1
        size = 0
        stack = [int(seed)]
        flat_labels[seed] = comp_id
        while stack:
            idx = stack.pop()
            size += 1
            x = idx // nn
            yz = idx - x * nn
            y = yz // n
            z = yz - y * n
            for j in (
                idx - nn if x > 0 else -1,
                idx + nn if x + 1 < n else -1,
                idx - n if y > 0 else -1,
                idx + n if y + 1 < n else -1,
                idx - 1 if z > 0 else -1,
                idx + 1 if z + 1 < n else -1,
            ):
                if j >= 0 and flat_mask[j] and flat_labels[j] == 0:
                    flat_labels[j] = comp_id
                    stack.append(j)
        sizes.append(size)
    if not sizes:
        return mask, {"num_components": 0, "largest_component_voxels": 0}
    best = int(np.argmax(np.asarray(sizes))) + 1
    keep = labels == best
    return keep, {"num_components": int(len(sizes)), "largest_component_voxels": int(sizes[best - 1])}


def main():
    parser = argparse.ArgumentParser(description="Dense-grid inference for pixel-aligned UDF model")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--out-dir", default="predictions_udf_pixel")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--iso-dist", type=float, default=0.02)
    parser.add_argument("--surface-band", type=float, default=0.0,
                        help="If >0, extract only the surface shell: points with "
                             "iso-dist - surface-band <= pred_dist <= iso-dist. "
                             "Produces a thin surface instead of a filled volume.")
    parser.add_argument("--surface-only", action="store_true",
                        help="If set, after thresholding keep only boundary voxels: "
                             "occupied voxels that have at least one empty 6-neighbour. "
                             "Produces a clean 1-voxel-thick surface shell.")
    parser.add_argument("--batch-query", type=int, default=24576)
    parser.add_argument("--bbox-pad-frac", type=float, default=0.10)
    parser.add_argument("--bbox-low-q", type=float, default=0.02)
    parser.add_argument("--bbox-high-q", type=float, default=0.98)
    parser.add_argument("--keep-largest-component", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["config"]
    model = PixelAlignedOccupancyNet(
        image_feat_dim=config["image_feat_dim"],
        hidden_dim=config["hidden_dim"],
        query_num_freqs=int(config.get("query_num_freqs", 0)),
        query_freq_scale=float(config.get("query_freq_scale", 1.0)),
        cam_pe_freqs=int(config.get("cam_pe_freqs", 0)),
        cam_pe_scale=float(config.get("cam_pe_scale", 1.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sample_dir = Path(args.sample_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_t, masks_t, K_t, T_t = load_sample(sample_dir, device)
    pts_path = sample_dir / "points_norm.npy"
    if not pts_path.exists():
        pts_path = sample_dir / "points.npy"
    gt_points = np.load(pts_path).astype(np.float32)
    bmin, bmax = robust_bbox_from_points(gt_points, args.bbox_low_q, args.bbox_high_q, args.bbox_pad_frac)
    grid = make_grid_from_bbox(bmin, bmax, args.resolution)
    print(f"[INFO] grid_points={grid.shape[0]} resolution={args.resolution}")

    mode = config.get("mode", "udf")
    prob_all = []
    with torch.no_grad():
        for i in range(0, grid.shape[0], args.batch_query):
            q = torch.from_numpy(grid[i:i + args.batch_query]).unsqueeze(0).to(device)
            logits = model(images_t, masks_t, K_t, T_t, q)
            if mode == "bce":
                prob = torch.sigmoid(logits).squeeze(0).cpu().numpy().astype(np.float32)
            else:
                prob = F.softplus(logits).squeeze(0).cpu().numpy().astype(np.float32)
            prob_all.append(prob)
    pred_dist = np.concatenate(prob_all, axis=0)
    if mode == "bce":
        # עבור BCE: סף על הסתברות occupancy (ערך גבוה = יותר "בפנים")
        occ_mask = pred_dist >= args.iso_dist
    elif args.surface_band > 0.0:
        occ_mask = pred_dist <= args.iso_dist
        inner = args.iso_dist - args.surface_band
        if inner > 0.0:
            occ_mask = occ_mask & (pred_dist >= inner)
    else:
        occ_mask = pred_dist <= args.iso_dist

    cc_stats = {"num_components": 0, "largest_component_voxels": int(occ_mask.sum())}
    if args.keep_largest_component and occ_mask.any():
        occ_3d = occ_mask.reshape(args.resolution, args.resolution, args.resolution)
        occ_3d, cc_stats = largest_component_mask_3d(occ_3d)
        occ_mask = occ_3d.reshape(-1)

    if args.surface_only and occ_mask.any():
        R = args.resolution
        occ_3d = occ_mask.reshape(R, R, R)
        padded = np.pad(occ_3d, 1, mode="constant", constant_values=False)
        interior = (
            padded[:-2, 1:-1, 1:-1]  # x-1
            & padded[2:,  1:-1, 1:-1]  # x+1
            & padded[1:-1, :-2, 1:-1]  # y-1
            & padded[1:-1, 2:,  1:-1]  # y+1
            & padded[1:-1, 1:-1, :-2]  # z-1
            & padded[1:-1, 1:-1, 2:]   # z+1
        )
        occ_3d = occ_3d & ~interior
        occ_mask = occ_3d.reshape(-1)

    occ_points = grid[occ_mask]
    sample_name = sample_dir.name

    np.save(out_dir / f"{sample_name}_grid.npy", grid)
    np.save(out_dir / f"{sample_name}_pred_dist.npy", pred_dist)
    np.save(out_dir / f"{sample_name}_occ_points.npy", occ_points)
    write_ply_ascii(out_dir / f"{sample_name}_occ_points.ply", occ_points)

    meta = {
        "sample_name": sample_name,
        "model": "pixel_aligned_udf",
        "resolution": int(args.resolution),
        "iso_dist": float(args.iso_dist),
        "num_grid_points": int(grid.shape[0]),
        "num_occupied_points": int(occ_points.shape[0]),
        "bbox_min": bmin.tolist(),
        "bbox_max": bmax.tolist(),
        "bbox_low_q": float(args.bbox_low_q),
        "bbox_high_q": float(args.bbox_high_q),
        "keep_largest_component": bool(args.keep_largest_component),
        "connected_components": cc_stats,
    }
    (out_dir / f"{sample_name}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[OK] occupied points: {occ_points.shape[0]}")
    print(f"[OK] saved: {out_dir / f'{sample_name}_occ_points.ply'}")


if __name__ == "__main__":
    main()
