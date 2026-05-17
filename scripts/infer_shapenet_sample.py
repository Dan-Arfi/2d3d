#!/usr/bin/env python3
"""אינפרנס מהיר על דוגמת ולידציה יחידה של ShapeNet. שומר PLY ומייצר תצוגת Plotly."""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from src.mv_model_occ_pixel import PixelAlignedOccupancyNet
from src.shapenet_r2n2_dataset import ShapeNetR2N2Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       default="checkpoints_shapenet_airplane/best.pt")
    parser.add_argument("--data-root",  default="data_shapenet")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--out",        default="predictions_shapenet")
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    model = PixelAlignedOccupancyNet(image_feat_dim=256, hidden_dim=cfg["hidden_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = ShapeNetR2N2Dataset(root=args.data_root, split="val", n_views=12, n_query=512)
    sample = ds[args.sample_idx]
    model_id = sample["sample_id"]
    print(f"[INFO] model_id={model_id}")

    # גריד צפוף ב-[-0.5, 0.5]^3
    R = args.resolution
    lin = np.linspace(-0.5, 0.5, R, dtype=np.float32)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    grid = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    imgs  = sample["images"].unsqueeze(0).to(device)
    masks = sample["masks"].unsqueeze(0).to(device)
    K     = sample["K"].unsqueeze(0).to(device)
    T     = sample["T_w2c"].unsqueeze(0).to(device)

    probs = []
    batch = 32768
    with torch.no_grad():
        for i in range(0, len(grid), batch):
            q = torch.from_numpy(grid[i:i+batch]).unsqueeze(0).to(device)
            logits = model(imgs, masks, K, T, q)
            probs.append(torch.sigmoid(logits).squeeze(0).cpu().numpy())
    probs = np.concatenate(probs)

    occ = probs >= args.threshold
    pts = grid[occ]
    print(f"[INFO] occupied={len(pts)} / {len(grid)}  (threshold={args.threshold})")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    ply_path = out / f"{model_id}_pred.ply"
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in pts:
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
    print(f"[OK] saved {ply_path}")

    # שמירת תמונות הקלט כרצועת PNG
    import torchvision.utils as vutils
    from PIL import Image as PILImage
    imgs_np = (sample["images"].permute(0,2,3,1).numpy() * 255).clip(0,255).astype(np.uint8)
    strip = np.concatenate([imgs_np[i] for i in range(min(6, len(imgs_np)))], axis=1)
    PILImage.fromarray(strip).save(out / f"{model_id}_input.png")
    print(f"[OK] saved input strip")

    # תצוגה מקדימה עם Plotly
    try:
        import plotly.graph_objects as go
        if len(pts) > 20000:
            idx = np.random.choice(len(pts), 20000, replace=False)
            show_pts = pts[idx]
        else:
            show_pts = pts
        fig = go.Figure(data=[go.Scatter3d(
            x=show_pts[:,0], y=show_pts[:,1], z=show_pts[:,2],
            mode="markers", marker=dict(size=1.5, color=show_pts[:,2], colorscale="Viridis", opacity=0.8)
        )])
        fig.update_layout(title=f"Predicted airplane — {model_id[:16]}", height=600,
                          scene=dict(bgcolor="black"), paper_bgcolor="black")
        fig.write_html(out / f"{model_id}_preview.html")
        print(f"[OK] saved HTML preview → open: {out / f'{model_id}_preview.html'}")
    except ImportError:
        print("[WARN] plotly not installed, skipping preview")

if __name__ == "__main__":
    main()
