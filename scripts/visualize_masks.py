#!/usr/bin/env python3
"""
הצגת מסכות ShapeNet-R2N2 לכל מבט עבור מזהה דגם יחיד.

דוגמת שימוש:
  python v3/scripts/visualize_masks.py \
    --model-id 1a04e3eab45ca15dd86060f189eb133 \
    --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _load_rgba(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    return np.array(img, dtype=np.uint8)


def _mask_edge(mask: np.ndarray) -> np.ndarray:
    """מחזיר את פיקסלי הגבול של מסכה בינארית."""
    center = mask
    up = np.roll(mask, -1, axis=0)
    down = np.roll(mask, 1, axis=0)
    left = np.roll(mask, -1, axis=1)
    right = np.roll(mask, 1, axis=1)
    interior = center & up & down & left & right
    return center & (~interior)


def _compose_mask_view(
    rgba: np.ndarray,
    bg_rgb: tuple[int, int, int],
    show_mask_overlay: bool,
    overlay_alpha: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """
    rgba: (H, W, 4) uint8
    מחזיר:
      overlay_view: (H, W, 3) uint8
      binary_mask_view: (H, W, 3) uint8
    """
    rgb = rgba[..., :3].astype(np.float32)
    alpha = (rgba[..., 3:4].astype(np.float32) / 255.0)
    mask = (alpha[..., 0] > 0.5)

    bg = np.zeros_like(rgb, dtype=np.float32)
    bg[..., 0] = float(bg_rgb[0])
    bg[..., 1] = float(bg_rgb[1])
    bg[..., 2] = float(bg_rgb[2])

    composed = rgb * alpha + bg * (1.0 - alpha)

    if show_mask_overlay:
        mask_f = mask.astype(np.float32)
        overlay = np.zeros_like(composed, dtype=np.float32)
        overlay[..., 0] = 255.0  # שכבת מג'נטה לשיפור נראות
        overlay[..., 2] = 255.0
        composed = composed * (1.0 - overlay_alpha * mask_f[..., None]) + overlay * (overlay_alpha * mask_f[..., None])

        # מוסיף קו מתאר בהיר כדי שגבול המסכה יהיה תמיד נראה.
        edge = _mask_edge(mask)
        composed[edge] = np.array([0.0, 255.0, 255.0], dtype=np.float32)  # גבול בצבע ציאן

    mask_panel = np.zeros_like(composed, dtype=np.float32)
    mask_panel[mask] = np.array([255.0, 255.0, 255.0], dtype=np.float32)

    return np.clip(composed, 0, 255).astype(np.uint8), np.clip(mask_panel, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize 12 masked renders for a ShapeNet model id.")
    parser.add_argument("--model-id", required=True, help="ShapeNet model id folder name.")
    parser.add_argument("--data-root", default="v3/data_shapenet", help="Path to dataset root.")
    parser.add_argument("--category", default="02691156", help="ShapeNet category id (default: airplanes).")
    parser.add_argument("--n-views", type=int, default=12, help="Number of views to show.")
    parser.add_argument("--bg", default="20,20,20", help="Background RGB, e.g. 20,20,20")
    parser.add_argument("--no-overlay", action="store_true", help="Disable mask color overlay.")
    parser.add_argument("--save", default="", help="Output PNG path. Defaults to predictions_demo/<id>_mask_views.png")
    parser.add_argument("--show", action="store_true", help="Open image with default viewer.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    render_dir = data_root / "ShapeNetRendering" / args.category / args.model_id / "rendering"
    if not render_dir.exists():
        raise SystemExit(f"[ERR] Rendering folder not found: {render_dir}")

    bg_parts = [p.strip() for p in args.bg.split(",")]
    if len(bg_parts) != 3:
        raise SystemExit("[ERR] --bg must be in format R,G,B")
    bg_rgb = tuple(int(x) for x in bg_parts)

    views_overlay: list[np.ndarray] = []
    views_mask: list[np.ndarray] = []
    for i in range(args.n_views):
        p = render_dir / f"{i:02d}.png"
        if not p.exists():
            break
        rgba = _load_rgba(p)
        overlay, mask_only = _compose_mask_view(
            rgba, bg_rgb=bg_rgb, show_mask_overlay=not args.no_overlay
        )
        views_overlay.append(overlay)
        views_mask.append(mask_only)

    if not views_overlay:
        raise SystemExit(f"[ERR] No views found in: {render_dir}")

    cols = 4
    rows_per_strip = int(np.ceil(len(views_overlay) / cols))
    h, w = views_overlay[0].shape[:2]
    pad = 6
    title_h = 26
    strip_h = rows_per_strip * h + (rows_per_strip + 1) * pad
    canvas_h = title_h + strip_h + title_h + strip_h + pad
    canvas_w = cols * w + (cols + 1) * pad
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[..., 0] = bg_rgb[0]
    canvas[..., 1] = bg_rgb[1]
    canvas[..., 2] = bg_rgb[2]

    y_overlay0 = title_h
    y_mask0 = title_h + strip_h + title_h + pad

    for idx, v in enumerate(views_overlay):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (w + pad)
        y0 = y_overlay0 + pad + r * (h + pad)
        canvas[y0:y0 + h, x0:x0 + w] = v

    for idx, v in enumerate(views_mask):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (w + pad)
        y0 = y_mask0 + pad + r * (h + pad)
        canvas[y0:y0 + h, x0:x0 + w] = v

    if args.save:
        out_path = Path(args.save)
    else:
        out_dir = Path("v3/predictions_demo")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.model_id}_mask_views.png"

    out_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(out_img)
    draw.text((8, 5), f"Overlay Views (model={args.model_id})", fill=(255, 255, 0))
    draw.text((8, y_mask0 - 22), "Binary Masks", fill=(255, 255, 0))

    out_img.save(out_path)
    print(f"[OK] Saved: {out_path.resolve()}")

    if args.show:
        out_img.show()


if __name__ == "__main__":
    main()
