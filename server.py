#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import io
import os
import re
import secrets
import sqlite3
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import websockets
from PIL import Image, ImageDraw

from protocol import make_message, parse_message
from src.mv_model_occ_pixel import PixelAlignedOccupancyNet
from src.shapenet_r2n2_dataset import (
    ShapeNetR2N2Dataset,
    _camera_to_matrices,
    _parse_metadata,
    read_binvox,
)


V3 = Path(__file__).parent
CKPT = V3 / "checkpoints_shapenet_v3" / "best.pt"
CKPT_HMAC = V3 / "checkpoints_shapenet_v3" / "best.pt.hmac"
EXPECTED_CKPT_SHA256 = "77003a6e546ce601806b118294dff386dd2688dc0969ebb6b1d5206e58724e69"
DATA_ROOT = V3 / "data_shapenet"
CATEGORY = "02691156"
PRED_DIR = V3 / "predictions_demo_ws"
PRED_DIR.mkdir(exist_ok=True)
DB_PATH = V3 / "users.db"
USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")

RENDER_DIR = DATA_ROOT / "ShapeNetRendering" / CATEGORY
VOX_DIR = DATA_ROOT / "ShapeNetVox32" / CATEGORY

DEVICE = torch.device("cpu")
_model: PixelAlignedOccupancyNet | None = None
_ds_train: ShapeNetR2N2Dataset | None = None
_ds_val: ShapeNetR2N2Dataset | None = None
_sessions: dict[int, dict[str, Any] | None] = {}


def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hmac_sha256_file(path: Path, key: bytes) -> str:
    mac = hmac.new(key, digestmod=hashlib.sha256)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            mac.update(chunk)
    return mac.hexdigest()


def verify_checkpoint_integrity(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"[SECURITY][CKPT] checkpoint not found: {path}")

    actual_sha256 = _sha256_file(path)
    if not hmac.compare_digest(actual_sha256, EXPECTED_CKPT_SHA256):
        raise RuntimeError(
            "[SECURITY][CKPT] SHA-256 mismatch. "
            f"expected={EXPECTED_CKPT_SHA256} actual={actual_sha256}"
        )
    print(f"[SECURITY][CKPT] SHA-256 OK: {actual_sha256}")

    hmac_key = os.getenv("MODEL_HMAC_KEY", "").strip()
    if not hmac_key:
        print("[SECURITY][CKPT] MODEL_HMAC_KEY not set; HMAC verification skipped")
        return

    if not CKPT_HMAC.exists():
        raise RuntimeError(
            "[SECURITY][CKPT] MODEL_HMAC_KEY set but signature file missing: "
            f"{CKPT_HMAC}"
        )

    expected_sig = CKPT_HMAC.read_text(encoding="utf-8").strip().lower()
    actual_sig = _hmac_sha256_file(path, bytes.fromhex(hmac_key))
    if not hmac.compare_digest(actual_sig, expected_sig):
        raise RuntimeError(
            "[SECURITY][CKPT] HMAC mismatch. "
            f"expected={expected_sig} actual={actual_sig}"
        )
    print(f"[SECURITY][CKPT] HMAC OK: {CKPT_HMAC}")


def init_auth_db() -> None:
    with _db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                resolution INTEGER NOT NULL,
                threshold_mult REAL NOT NULL,
                pred_points INTEGER NOT NULL,
                gt_voxels INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )


def _hash_password(password: str, salt_hex: str | None = None) -> tuple[str, str]:
    if salt_hex is None:
        salt_hex = secrets.token_hex(16)
    salt = bytes.fromhex(salt_hex)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return salt_hex, pwd_hash.hex()


def register_user(username: str, password: str) -> tuple[bool, str]:
    username = (username or "").strip()
    password = password or ""

    if not USERNAME_RE.match(username):
        return False, f"שם המשתמש חייב להכיל 3-32 תווים (אותיות/ספרות/._-)"
    if len(password) < 6:
        return False, "הסיסמה חייבת להכיל לפחות 6 תווים"

    salt_hex, pwd_hash = _hash_password(password)
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    try:
        with _db_conn() as conn:
            conn.execute(
                "INSERT INTO users(username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
                (username, pwd_hash, salt_hex, now),
            )
    except sqlite3.IntegrityError:
        return False, f"חשבון כבר קיים עבור '{username}'"
    return True, "נרשם בהצלחה"


def authenticate_user(username: str, password: str) -> tuple[bool, dict[str, Any] | None, str]:
    username = (username or "").strip()
    password = password or ""
    with _db_conn() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, salt FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if row is None:
        return False, None, f"לא נמצא חשבון עבור '{username}'"
    _, candidate_hash = _hash_password(password, row["salt"])
    if not hmac.compare_digest(candidate_hash, row["password_hash"]):
        return False, None, "סיסמה שגויה"
    return True, {"id": int(row["id"]), "username": row["username"]}, "התחברת בהצלחה"


def add_generation_history(
    user_id: int,
    model_id: str,
    resolution: int,
    threshold_mult: float,
    pred_points: int,
    gt_voxels: int,
) -> None:
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    with _db_conn() as conn:
        conn.execute(
            """
            INSERT INTO generations(user_id, model_id, resolution, threshold_mult, pred_points, gt_voxels, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, model_id, resolution, threshold_mult, pred_points, gt_voxels, now),
        )


def fetch_generation_history(user_id: int, limit: int = 100) -> list[dict[str, Any]]:
    with _db_conn() as conn:
        rows = conn.execute(
            """
            SELECT created_at, model_id, resolution, threshold_mult, pred_points, gt_voxels
            FROM generations
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [
        {
            "created_at": r["created_at"],
            "model_id": r["model_id"],
            "resolution": int(r["resolution"]),
            "threshold_mult": float(r["threshold_mult"]),
            "pred_points": int(r["pred_points"]),
            "gt_voxels": int(r["gt_voxels"]),
        }
        for r in rows
    ]


def load_runtime() -> None:
    global _model, _ds_train, _ds_val
    if _model is not None:
        return
    verify_checkpoint_integrity(CKPT)
    ckpt = torch.load(str(CKPT), map_location=DEVICE, weights_only=False)
    model = PixelAlignedOccupancyNet(hidden_dim=ckpt["config"]["hidden_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    _model = model

    _ds_train = ShapeNetR2N2Dataset(root=str(DATA_ROOT), split="train", n_views=12, n_query=64)
    _ds_val = ShapeNetR2N2Dataset(root=str(DATA_ROOT), split="val", n_views=12, n_query=64)


def _mask_edge(mask: np.ndarray) -> np.ndarray:
    center = mask
    up = np.roll(mask, -1, axis=0)
    down = np.roll(mask, 1, axis=0)
    left = np.roll(mask, -1, axis=1)
    right = np.roll(mask, 1, axis=1)
    interior = center & up & down & left & right
    return center & (~interior)


def _render_frame_mode(rgba: np.ndarray, mode: str) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    mask = alpha[..., 0] > 0.5
    bg = np.full_like(rgb, 20.0)
    composed = rgb * alpha + bg * (1.0 - alpha)

    if mode == "binary":
        out = np.zeros_like(composed, dtype=np.float32)
        out[mask] = np.array([255.0, 255.0, 255.0], dtype=np.float32)
        return out.astype(np.uint8)

    if mode in ("mask_edge", "compare"):
        overlay = np.zeros_like(composed, dtype=np.float32)
        overlay[..., 0] = 255.0
        overlay[..., 2] = 255.0
        mask_f = mask.astype(np.float32)
        composed = composed * (1.0 - 0.55 * mask_f[..., None]) + overlay * (0.55 * mask_f[..., None])
        edge = _mask_edge(mask)
        composed[edge] = np.array([0.0, 255.0, 255.0], dtype=np.float32)

    if mode == "compare":
        bin_img = np.zeros_like(composed, dtype=np.float32)
        bin_img[mask] = np.array([255.0, 255.0, 255.0], dtype=np.float32)
        pad = 4
        h, w, _ = composed.shape
        merged = np.zeros((h, w * 2 + pad, 3), dtype=np.float32)
        merged[:, :w] = composed
        merged[:, w + pad :] = bin_img
        return np.clip(merged, 0, 255).astype(np.uint8)

    return np.clip(composed, 0, 255).astype(np.uint8)


def get_frames(model_id: str, n: int = 12, mode: str = "mask_edge") -> list[str]:
    rend = RENDER_DIR / model_id / "rendering"
    encoded: list[str] = []
    for i in range(n):
        p = rend / f"{i:02d}.png"
        if not p.exists():
            break
        rgba = np.array(Image.open(p).convert("RGBA"), dtype=np.uint8)
        vis = _render_frame_mode(rgba, mode=mode)
        bg = Image.fromarray(vis, mode="RGB").resize((160, 160), Image.BILINEAR)
        draw = ImageDraw.Draw(bg)
        draw.text((4, 4), f"view {i+1}", fill=(255, 220, 0))
        buf = io.BytesIO()
        bg.save(buf, format="PNG")
        encoded.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return encoded


def get_thumbnail(model_id: str) -> str:
    img_path = RENDER_DIR / model_id / "rendering" / "00.png"
    rgba = Image.open(img_path).convert("RGBA")
    bg = Image.new("RGB", rgba.size, (20, 20, 20))
    bg.paste(rgba, mask=rgba.split()[3])
    bg = bg.resize((128, 128), Image.BILINEAR)
    buf = io.BytesIO()
    bg.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def get_library(split: str, search: str = "", limit: int = 96) -> list[dict[str, str]]:
    assert _ds_train is not None and _ds_val is not None
    if split == "train":
        ids = _ds_train.model_ids
    else:
        ids = _ds_val.model_ids
    q = (search or "").strip()
    if q:
        ids = [mid for mid in ids if q in mid]

    items: list[dict[str, str]] = []
    for mid in ids[:limit]:
        try:
            items.append({"model_id": mid, "thumb": get_thumbnail(mid)})
        except Exception:
            continue
    return items


def run_inference(
    model_id: str,
    resolution: int = 64,
    threshold_mult: float = 1.0,
    progress_cb: Callable[[int, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    assert _model is not None

    def _progress(pct: int, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(pct, msg)

    _progress(5, "reading metadata")
    rend = RENDER_DIR / model_id / "rendering"
    meta = _parse_metadata(rend / "rendering_metadata.txt")
    n_use = min(12, len(meta))

    images_list, masks_list, k_list, t_list = [], [], [], []
    for vi in range(n_use):
        img = Image.open(rend / f"{vi:02d}.png").convert("RGBA")
        arr = np.array(img, dtype=np.float32) / 255.0
        rgb = arr[..., :3]
        mask = (arr[..., 3] > 0.5).astype(np.float32)
        h, w = rgb.shape[:2]
        k, t = _camera_to_matrices(meta[vi], img_h=h, img_w=w)
        images_list.append(rgb)
        masks_list.append(mask)
        k_list.append(k)
        t_list.append(t)
        if n_use > 0:
            pct = 8 + int((vi + 1) / n_use * 17)
            _progress(pct, f"loading views {vi+1}/{n_use}")

    _progress(28, "building tensors")
    imgs_t = torch.from_numpy(np.stack(images_list)).permute(0, 3, 1, 2).unsqueeze(0)
    masks_t = torch.from_numpy(np.stack(masks_list)).unsqueeze(1).unsqueeze(0)
    k_t = torch.from_numpy(np.stack(k_list)).unsqueeze(0)
    t_t = torch.from_numpy(np.stack(t_list)).unsqueeze(0)

    _progress(35, "building query grid")
    lin = np.linspace(-0.5, 0.5, resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), axis=-1).reshape(-1, 3)

    _progress(40, "running model")
    probs = []
    chunk_size = 16384
    n_chunks = (len(grid) + chunk_size - 1) // chunk_size
    with torch.no_grad():
        for ci, i in enumerate(range(0, len(grid), chunk_size), start=1):
            q = torch.from_numpy(grid[i : i + chunk_size]).unsqueeze(0)
            probs.append(torch.sigmoid(_model(imgs_t, masks_t, k_t, t_t, q)).squeeze(0).cpu().numpy()) # קריאה של המודל
            pct = 40 + int(ci / n_chunks * 45)
            _progress(min(pct, 85), f"inference chunks {ci}/{n_chunks}")
    probs = np.concatenate(probs)

    _progress(88, "post-processing predictions")
    vox = read_binvox(VOX_DIR / model_id / "model.binvox") 
    gt_count = int(vox.sum()) * (resolution // 32) ** 3
    n_keep = max(100, int(gt_count * threshold_mult))
    n_keep = min(n_keep, len(probs))
    thresh = float(np.sort(probs)[::-1][n_keep - 1])
    keep = probs >= thresh
    pred_pts = grid[keep]
    pred_probs = probs[keep]
    gt_pts = np.argwhere(vox).astype(np.float32) / 32.0 - 0.5
    _progress(92, "finalizing output")
    return grid, keep, pred_pts, pred_probs, gt_pts, int(vox.sum())


def _style_scene(fig: go.Figure, scene_name: str) -> None:
    fig.update_layout(
        **{
            scene_name: dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="black",
                aspectmode="data",
            )
        }
    )


def make_pred_figure(
    grid_pts: np.ndarray,
    in_mask: np.ndarray,
    pred_pts: np.ndarray,
    pred_probs: np.ndarray,
    gt_pts: np.ndarray | None,
    color_mode: str = "height",
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Full 3D Grid", "In/Out Classification", "Final Prediction"),
    )

    # Panel 1: all query points in the grid.
    fig.add_trace(
        go.Scatter3d(
            x=grid_pts[:, 0],
            y=grid_pts[:, 1],
            z=grid_pts[:, 2],
            mode="markers",
            name="grid points",
            marker=dict(size=1, color="#6ea8ff", opacity=0.25),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Panel 2: all query points split into out/in for separate opacity control.
    out_pts = grid_pts[~in_mask]
    in_pts = grid_pts[in_mask]
    if len(out_pts):
        fig.add_trace(
            go.Scatter3d(
                x=out_pts[:, 0],
                y=out_pts[:, 1],
                z=out_pts[:, 2],
                mode="markers",
                name="out points",
                marker=dict(size=1.3, color="#ffffff", opacity=0.08),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    if len(in_pts):
        fig.add_trace(
            go.Scatter3d(
                x=in_pts[:, 0],
                y=in_pts[:, 1],
                z=in_pts[:, 2],
                mode="markers",
                name="in points",
                marker=dict(size=1.8, color="#ff2b2b", opacity=0.9),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Panel 3: current final view.
    if gt_pts is not None and len(gt_pts):
        fig.add_trace(
            go.Scatter3d(
                x=gt_pts[:, 0],
                y=gt_pts[:, 1],
                z=gt_pts[:, 2],
                mode="markers",
                name="GT",
                marker=dict(size=3, color="cyan", opacity=0.65),
                showlegend=True,
            ),
            row=1,
            col=3,
        )
    if len(pred_pts):
        if color_mode == "confidence":
            marker_color = pred_probs
            marker_scale = [[0.0, "#ff0000"], [1.0, "#ffffff"]]
            marker_cmin, marker_cmax = 0.0, 1.0
            cbar_title = "confidence"
        else:
            marker_color = pred_pts[:, 2]
            marker_scale = "Hot"
            marker_cmin, marker_cmax = None, None
            cbar_title = "height z"
        fig.add_trace(
            go.Scatter3d(
                x=pred_pts[:, 0],
                y=pred_pts[:, 1],
                z=pred_pts[:, 2],
                mode="markers",
                name="prediction",
                marker=dict(
                    size=2,
                    color=marker_color,
                    cmin=marker_cmin,
                    cmax=marker_cmax,
                    colorscale=marker_scale,
                    opacity=0.8,
                    colorbar=dict(title=cbar_title),
                ),
                showlegend=True,
            ),
            row=1,
            col=3,
        )

    _style_scene(fig, "scene")
    _style_scene(fig, "scene2")
    _style_scene(fig, "scene3")
    fig.update_layout(
        paper_bgcolor="black",
        plot_bgcolor="black",
        margin=dict(l=0, r=0, t=38, b=0),
        height=560,
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
        font=dict(color="white"),
    )
    return fig


async def send_progress(ws: websockets.WebSocketServerProtocol, message: str, pct: int) -> None:
    print(f"[ws] progress: {pct}% - {message}")
    await ws.send(make_message("progress", message=message, pct=pct))


async def handle_message(ws: websockets.WebSocketServerProtocol, msg: dict[str, Any]) -> None:
    t = msg["type"]
    session_key = id(ws)
    user = _sessions.get(session_key)

    if t == "register":
        username = str(msg.get("username", ""))
        password = str(msg.get("password", ""))
        ok, status = register_user(username, password)
        print(f"[ws] register: username='{username}' ok={ok} status='{status}'")
        await ws.send(make_message("auth_result", action="register", ok=ok, message=status))
        return

    if t == "login":
        username = str(msg.get("username", ""))
        password = str(msg.get("password", ""))
        ok, user_data, status = authenticate_user(username, password)
        print(f"[ws] login: username='{username}' ok={ok} status='{status}'")
        if ok and user_data is not None:
            _sessions[session_key] = user_data
            await ws.send(make_message("auth_result", action="login", ok=True, message=status, user=user_data))
        else:
            await ws.send(make_message("auth_result", action="login", ok=False, message=status))
        return

    if t == "logout":
        username = user["username"] if user else "?"
        print(f"[ws] logout: username='{username}'")
        _sessions[session_key] = None
        await ws.send(make_message("auth_result", action="logout", ok=True, message="התנתקת בהצלחה"))
        return

    if t == "history":
        if not user:
            print("[ws] history: denied (not logged in)")
            await ws.send(make_message("history_result", ok=False, message="יש להתחבר תחילה", rows=[]))
            return
        rows = fetch_generation_history(int(user["id"]))
        print(f"[ws] history: user='{user['username']}' rows={len(rows)}")
        await ws.send(make_message("history_result", ok=True, message=f"{len(rows)} rows", rows=rows))
        return

    if t == "list_models":
        print("[ws] list_models")
        assert _ds_train is not None and _ds_val is not None
        await ws.send(
            make_message(
                "models",
                train=_ds_train.model_ids[:500],
                val=_ds_val.model_ids[:500],
            )
        )
        return

    if t == "library":
        split = str(msg.get("split", "val"))
        search = str(msg.get("search", ""))
        print(f"[ws] library: split='{split}' search='{search}'")
        items = await asyncio.to_thread(get_library, split, search, 96)
        print(f"[ws] library: returning {len(items)} items")
        await ws.send(make_message("library", split=split, items=items))
        return

    if t == "load_views":
        model_id = str(msg["model_id"])
        mode = str(msg.get("view_mode", "mask_edge"))
        print(f"[ws] load_views: model='{model_id}' mode='{mode}'")
        frames = await asyncio.to_thread(get_frames, model_id, 12, mode)
        print(f"[ws] load_views: returning {len(frames)} frames")
        await ws.send(make_message("views", model_id=model_id, frames=frames))
        return

    if t == "infer":
        if not user:
            print("[ws] infer: denied (not logged in)")
            await ws.send(make_message("error", message="יש להתחבר לפני הרצת מודל"))
            return
        model_id = str(msg["model_id"])
        resolution = int(msg.get("resolution", 64))
        threshold_mult = float(msg.get("threshold_mult", 1.0))
        print(f"[ws] infer: user='{user['username']}' model='{model_id}' res={resolution} thresh={threshold_mult}")
        show_gt = bool(msg.get("show_gt", True))
        color_mode = str(msg.get("color_mode", "height"))

        progress_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _thread_progress(pct: int, message: str) -> None:
            loop.call_soon_threadsafe(progress_queue.put_nowait, (pct, message))

        async def _pump_progress(task: asyncio.Task) -> None:
            while not task.done():
                try:
                    pct, message = await asyncio.wait_for(progress_queue.get(), timeout=0.2)
                    await send_progress(ws, message, pct)
                except asyncio.TimeoutError:
                    continue
            while not progress_queue.empty():
                pct, message = await progress_queue.get()
                await send_progress(ws, message, pct)

        infer_task = asyncio.create_task(
            asyncio.to_thread(run_inference, model_id, resolution, threshold_mult, _thread_progress) # הרצה של המודל
        )
        pump_task = asyncio.create_task(_pump_progress(infer_task))
        try:
            grid_pts, in_mask, pred_pts, pred_probs, gt_pts, gt_vox = await infer_task
        finally:
            await pump_task
        await send_progress(ws, "building 3d preview", 95)
        fig = make_pred_figure(
            grid_pts,
            in_mask,
            pred_pts,
            pred_probs,
            gt_pts if show_gt else None,
            color_mode=color_mode,
        )
        out_html = PRED_DIR / f"{model_id}_r{resolution}.html"
        PRED_DIR.mkdir(exist_ok=True)
        # Use self-contained HTML so embedded Qt viewer can render without CDN dependencies.
        await asyncio.to_thread(fig.write_html, str(out_html), include_plotlyjs=True)
        await send_progress(ws, "done", 100)
        add_generation_history(
            user_id=int(user["id"]),
            model_id=model_id,
            resolution=resolution,
            threshold_mult=threshold_mult,
            pred_points=int(len(pred_pts)),
            gt_voxels=gt_vox,
        )
        print(f"[ws] infer: done model='{model_id}' pred_points={len(pred_pts)} gt_voxels={gt_vox}")
        await ws.send(
            make_message(
                "result",
                model_id=model_id,
                html_path=str(out_html.resolve()),
                pred_points=int(len(pred_pts)),
                gt_voxels=gt_vox,
            )
        )
        return

    print(f"[ws] unknown message type: '{t}'")
    await ws.send(make_message("error", message=f"Unknown message type: {t}"))


async def ws_handler(ws: websockets.WebSocketServerProtocol) -> None:
    addr = ws.remote_address
    print(f"[ws] client connected: {addr}")
    await ws.send(make_message("status", message="connected"))
    _sessions[id(ws)] = None
    try:
        async for raw in ws:
            try:
                msg = parse_message(raw)
                await handle_message(ws, msg)
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc(limit=1)
                print(f"[ws] error: {exc}")
                await ws.send(make_message("error", message=f"{exc}", detail=tb))
    finally:
        _sessions.pop(id(ws), None)
        print(f"[ws] client disconnected: {addr}")


async def main() -> None:
    init_auth_db()
    load_runtime()
    host = "0.0.0.0"
    port = 8765
    print(f"[server] listening on ws://{host}:{port}")
    async with websockets.serve(ws_handler, host, port, max_size=10_000_000):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
