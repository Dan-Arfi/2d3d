#!/usr/bin/env python3
import math
import struct
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


# ---------------------------------------------------------------------------
# קורא קבצי binvox (ללא תלות חיצונית)
# ---------------------------------------------------------------------------

def read_binvox(path: Path) -> np.ndarray:
    """קורא קובץ .binvox ומחזיר מערך בוליאני בגודל (D,H,W)."""
    with open(path, "rb") as f:
        line = f.readline().strip()
        assert line.startswith(b"#binvox"), f"Not a binvox file: {path}"
        dims_line = f.readline().strip().split()
        dims = tuple(int(x) for x in dims_line[1:])
        f.readline()  # translate
        f.readline()  # scale
        f.readline()  # data
        raw = f.read()
    D = dims[0]
    voxels = np.zeros(D * D * D, dtype=bool)
    idx = 0
    i = 0
    while i < len(raw) - 1:
        value = raw[i]
        count = raw[i + 1]
        voxels[idx: idx + count] = bool(value)
        idx += count
        i += 2
    return voxels.reshape(dims)


# ---------------------------------------------------------------------------
# פענוח פרמטרי מצלמה (קובץ rendering_metadata.txt של 3D-R2N2)
# ---------------------------------------------------------------------------

def _parse_metadata(path: Path):
    """
    כל שורה ב-rendering_metadata.txt היא:
    azimuth elevation tilt distance fov
    מחזיר רשימת מילונים עם השדות האלה כ-float.
    """
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


def _camera_to_matrices(cam: dict, img_h: int = 137, img_w: int = 137):
    """
    ממיר פרמטרי מצלמה ספריים של 3D-R2N2 ל:
    K (3×3) ו-T_w2c (4×4).

    קונבנציה: האובייקט ממורכז בראשית, והמצלמה מסתכלת לראשית.
    המטריצה K מניחה קואורדינטות פיקסל עם cx=img_w/2, cy=img_h/2.
    """
    az  = math.radians(cam["azimuth"])
    el  = math.radians(cam["elevation"])
    tilt = math.radians(cam["tilt"])
    dist = cam["distance"]
    fov  = math.radians(cam["fov"])

    # מיקום המצלמה במערכת העולם (ספריות -> קרטזיות)
    cx_w = dist * math.cos(el) * math.sin(az)
    cy_w = dist * math.sin(el)
    cz_w = dist * math.cos(el) * math.cos(az)
    cam_pos = np.array([cx_w, cy_w, cz_w], dtype=np.float32)

    # רוטציה: המצלמה מסתכלת אל הראשית
    z_axis = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)  # קדימה (לתוך הסצנה)
    up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = np.cross(z_axis, up_world)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        x_axis = np.cross(z_axis, up_world)
        x_norm = np.linalg.norm(x_axis)
    x_axis /= x_norm
    y_axis = np.cross(x_axis, z_axis)

    # החלת סיבוב tilt סביב ציר z
    if abs(tilt) > 1e-6:
        ct, st = math.cos(tilt), math.sin(tilt)
        x_axis_new = ct * x_axis + st * y_axis
        y_axis_new = -st * x_axis + ct * y_axis
        x_axis, y_axis = x_axis_new, y_axis_new

    R = np.stack([x_axis, y_axis, z_axis], axis=0).astype(np.float32)   # (3,3) rows=cam axes
    t = -R @ cam_pos                                                       # (3,)

    T_w2c = np.eye(4, dtype=np.float32)
    T_w2c[:3, :3] = R
    T_w2c[:3, 3]  = t

    # אינטרינזיים: מצלמת חריר, פיקסלים ריבועיים, נקודה עיקרית במרכז התמונה
    f = (img_w / 2.0) / math.tan(fov / 2.0)
    K = np.array([
        [f,   0.0, img_w / 2.0],
        [0.0, f,   img_h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return K, T_w2c


# ---------------------------------------------------------------------------
# דגימת Occupancy מתוך גריד ווקסלים
# ---------------------------------------------------------------------------

def _neighbour_counts(voxel: np.ndarray) -> np.ndarray:
    """לכל ווקסל מאוכלס, סופר כמה שכנים מאוכלסים יש לו (6-שכנות)."""
    occ = voxel.astype(np.int8)
    counts = np.zeros_like(occ)
    counts[1:,  :,  :] += occ[:-1, :,   :]
    counts[:-1, :,  :] += occ[1:,  :,   :]
    counts[:,  1:,  :] += occ[:,  :-1,  :]
    counts[:, :-1,  :] += occ[:,   1:,  :]
    counts[:,   :, 1:] += occ[:,    :, :-1]
    counts[:,   :,:-1] += occ[:,    :,  1:]
    return counts  # (D,D,D)


def _sample_occupancy(voxel: np.ndarray, n_query: int,
                      inside_frac: float = 0.5,
                      surface_expand: int = 1) -> tuple:
    """
    דוגם n_query נקודות תלת-ממד ב-[-0.5, 0.5]^3 ומחזיר (points, labels).
    labels: 1=מאוכלס, 0=ריק.

    אסטרטגיה:
      - inside_frac מהנקודות נדגמות ליד ווקסלים מאוכלסים
      - ווקסלים דקים/דלילים (כנפיים, קצוות) מקבלים דגימת-יתר בעזרת משקל הפוך לשכנים
      - השאר נדגם בצורה אחידה (ברובו שלילי)
    """
    D = voxel.shape[0]
    rng = np.random.default_rng()

    n_pos = int(n_query * inside_frac)
    n_neg = n_query - n_pos

    # מרכזי ווקסלים מאוכלסים ב-[-0.5, 0.5]
    occ_idx = np.argwhere(voxel)   # (P,3)
    if len(occ_idx) == 0:
        # מקרה קצה: דגימה אחידה בלבד
        pts = rng.uniform(-0.5, 0.5, (n_query, 3)).astype(np.float32)
        labels = np.zeros(n_query, dtype=np.float32)
        return pts, labels

    # משקל לכל ווקסל מאוכלס נקבע הפוך למספר השכנים שלו
    # ווקסלים מבודדים/דקים (למשל כנפיים) -> מעט שכנים -> משקל גבוה -> נדגמים יותר
    nb = _neighbour_counts(voxel)
    nb_occ = nb[occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]].astype(np.float32)
    weights = 1.0 / (nb_occ + 1.0)   # +1 to avoid div-by-zero for isolated voxels
    weights /= weights.sum()

    # דגימת נקודות ליד ווקסלים מאוכלסים עם משקול שמדגיש אזורים דקים
    pick = rng.choice(len(occ_idx), size=n_pos, p=weights)
    chosen = occ_idx[pick].astype(np.float32)              # (n_pos,3) voxel indices
    jitter = rng.uniform(-0.5, 0.5, chosen.shape).astype(np.float32)
    pts_pos = (chosen + 0.5 + jitter) / D - 0.5           # map to [-0.5,0.5]

    # דגימת נקודות שליליות באופן אחיד
    pts_neg = rng.uniform(-0.5, 0.5, (n_neg * 3, 3)).astype(np.float32)
    # תיוג לפי בדיקת ווקסל מתאים
    idx_neg = np.floor((pts_neg + 0.5) * D).astype(int).clip(0, D - 1)
    is_occ  = voxel[idx_neg[:, 0], idx_neg[:, 1], idx_neg[:, 2]]
    pts_neg = pts_neg[~is_occ][:n_neg]
    if len(pts_neg) < n_neg:
        extra = rng.uniform(-0.5, 0.5, (n_neg, 3)).astype(np.float32)
        pts_neg = np.concatenate([pts_neg, extra], axis=0)[:n_neg]

    pts = np.concatenate([pts_pos, pts_neg], axis=0)
    labels = np.concatenate([
        np.ones(len(pts_pos), dtype=np.float32),
        np.zeros(len(pts_neg), dtype=np.float32),
    ])
    # ערבוב
    perm = rng.permutation(len(pts))
    return pts[perm].astype(np.float32), labels[perm]


# ---------------------------------------------------------------------------
# מחלקת הדאטהסט הראשית
# ---------------------------------------------------------------------------

class ShapeNetR2N2Dataset(Dataset):
    """
    דאטהסט ShapeNet רב-מבטי שמשתמש ברינדורים של 3D-R2N2 וב-GT מסוג binvox.

    ארגומנטים:
        root:       נתיב שמכיל את ShapeNetRendering/ ואת ShapeNetVox32/
        split:      "train" | "val" | "test"
        category:   מזהה קטגוריה ב-ShapeNet, ברירת מחדל "02691156" (מטוס)
        n_views:    מספר המבטים (מתוך 24) לשימוש בכל דוגמה
        n_query:    מספר נקודות שאילתה של occupancy לכל דוגמה
        img_size:   שינוי גודל תמונה ל-(H, W). אם None נשאר 137×137 מקורי
        train_frac: החלק היחסי של דגמי train (והשאר ל-val/test)
        seed:       זרע אקראי לפיצולים שחוזרים על עצמם
    """

    AIRPLANE = "02691156"

    def __init__(
        self,
        root: str,
        split: str = "train",
        category: str = AIRPLANE,
        n_views: int = 12,
        n_query: int = 4096,
        img_size=None,
        train_frac: float = 0.80,
        val_frac: float = 0.10,
        seed: int = 42,
        use_256: bool = False,
    ):
        self.root      = Path(root)
        self.split     = split
        self.category  = category
        self.n_views   = n_views
        self.n_query   = n_query
        self.img_size  = img_size  # (H, W) or None

        render_dir  = self.root / "ShapeNetRendering" / category
        vox_dir     = self.root / "ShapeNetVox32"    / category
        vox256_dir  = self.root / "ShapeNetCore.v2"  / category

        if not render_dir.exists():
            raise FileNotFoundError(f"Renders not found: {render_dir}")

        # העדפה לווקסלים 256³ מ-ShapeNetCore.v2, ונפילה חזרה ל-32³ אם אין
        self.use_256 = use_256 and vox256_dir.exists()
        if self.use_256:
            self.vox256_dir = vox256_dir
            print(f"[ShapeNetR2N2] Using 256³ voxels from ShapeNetCore.v2")
        elif not vox_dir.exists():
            raise FileNotFoundError(f"No voxels found at {vox256_dir} or {vox_dir}")

        # איסוף מזהי דגמים שיש להם גם רינדורים וגם ווקסלים
        def _has_vox(mid):
            if self.use_256:
                return (vox256_dir / mid / "models" / "model_normalized.256.solid.binvox").exists()
            return (vox_dir / mid / "model.binvox").exists()

        all_ids = sorted(
            m.name for m in render_dir.iterdir()
            if m.is_dir() and _has_vox(m.name)
        )
        if not all_ids:
            raise ValueError(f"No models found in {render_dir}")

        # פיצול train/val/test דטרמיניסטי
        rng = random.Random(seed)
        ids_shuffled = list(all_ids)
        rng.shuffle(ids_shuffled)

        n_total = len(ids_shuffled)
        n_train = int(n_total * train_frac)
        n_val   = int(n_total * val_frac)

        splits = {
            "train": ids_shuffled[:n_train],
            "val":   ids_shuffled[n_train: n_train + n_val],
            "test":  ids_shuffled[n_train + n_val:],
        }
        if split not in splits:
            raise ValueError(f"split must be train/val/test, got '{split}'")

        self.model_ids  = splits[split]
        self.render_dir = render_dir
        self.vox_dir    = vox_dir if not self.use_256 else None

        print(f"[ShapeNetR2N2] category={category} split={split} "
              f"n_models={len(self.model_ids)} n_views={n_views}")

    def __len__(self):
        return len(self.model_ids)

    def __getitem__(self, idx: int):
        model_id  = self.model_ids[idx]
        rend_path = self.render_dir / model_id / "rendering"

        # --- טעינה ופענוח של המצלמות ---
        meta_path = rend_path / "rendering_metadata.txt"
        cameras   = _parse_metadata(meta_path)   # list of dicts, one per render

        # --- בחירת n_views מבטים אקראיים ---
        n_avail = min(len(cameras), 24)
        view_indices = sorted(random.sample(range(n_avail), min(self.n_views, n_avail)))

        images_list = []
        masks_list  = []
        K_list      = []
        T_list      = []

        for vi in view_indices:
            img_path = rend_path / f"{vi:02d}.png"
            img = Image.open(img_path).convert("RGBA")

            if self.img_size is not None:
                img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)

            arr  = np.array(img, dtype=np.float32) / 255.0   # (H,W,4)
            rgb  = arr[..., :3]                               # (H,W,3)
            mask = (arr[..., 3] > 0.5).astype(np.float32)    # (H,W) alpha → mask

            H, W = rgb.shape[:2]
            K, T_w2c = _camera_to_matrices(cameras[vi], img_h=H, img_w=W)

            images_list.append(rgb)
            masks_list.append(mask)
            K_list.append(K)
            T_list.append(T_w2c)

        images = np.stack(images_list, axis=0)   # (V,H,W,3)
        masks  = np.stack(masks_list,  axis=0)   # (V,H,W)
        Ks     = np.stack(K_list,      axis=0)   # (V,3,3)
        Ts     = np.stack(T_list,      axis=0)   # (V,4,4)

        # --- טעינת ווקסלים ודגימת occupancy ---
        if self.use_256:
            vox_path = self.vox256_dir / model_id / "models" / "model_normalized.256.solid.binvox"
        else:
            vox_path = self.vox_dir / model_id / "model.binvox"
        voxel = read_binvox(vox_path)             # (D,D,D) bool
        pts, labels = _sample_occupancy(voxel, self.n_query, inside_frac=0.15)

        return {
            # הטנסורים שהמודל PixelAlignedOccupancyNet מצפה להם
            "images":     torch.from_numpy(images).permute(0, 3, 1, 2).contiguous(),  # (V,3,H,W)
            "masks":      torch.from_numpy(masks).unsqueeze(1).contiguous(),           # (V,1,H,W)
            "K":          torch.from_numpy(Ks),                                        # (V,3,3)
            "T_w2c":      torch.from_numpy(Ts),                                        # (V,4,4)
            "occ_points": torch.from_numpy(pts),                                       # (N,3)
            "occ_labels": torch.from_numpy(labels),                                    # (N,)
            "sample_id":  model_id,
            "category":   self.category,
        }


def shapenet_collate_fn(batch):
    return {
        "images":     torch.stack([b["images"]     for b in batch], dim=0),
        "masks":      torch.stack([b["masks"]      for b in batch], dim=0),
        "K":          torch.stack([b["K"]          for b in batch], dim=0),
        "T_w2c":      torch.stack([b["T_w2c"]      for b in batch], dim=0),
        "occ_points": torch.stack([b["occ_points"] for b in batch], dim=0),
        "occ_labels": torch.stack([b["occ_labels"] for b in batch], dim=0),
        "sample_id":  [b["sample_id"] for b in batch],
        "category":   [b["category"]  for b in batch],
    }
