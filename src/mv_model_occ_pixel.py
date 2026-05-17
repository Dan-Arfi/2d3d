#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torchvision_models


class ResNet18Encoder(nn.Module):
    """
    מקודד ResNet18 מאומן מראש וקפוא.
    הפלט הוא מפות פיצ'רים בשלוש רזולוציות:
      f2: (B*V, 64,  H/4,  W/4)  אחרי layer1 (פרטים מרחביים עדינים)
      f3: (B*V, 128, H/8,  W/8)  אחרי layer2 (פיצ'רים ברמת ביניים)
      f4: (B*V, 256, H/16, W/16) אחרי layer3 (פיצ'רים סמנטיים/גסים)

      

    נרמול ImageNet נעשה פנימית.
    הקלט x צריך להיות בטווח [0, 1].
    משקלי ה-backbone קפואים; רק שכבות ה-MLP/head בהמשך מתאמנות.
    """
    # מספר הערוצים בכל סקייל של הפלט.
    DIMS = (64, 128, 256)

    def __init__(self):
        super().__init__()
        backbone = torchvision_models.resnet18(weights=torchvision_models.ResNet18_Weights.IMAGENET1K_V1)
        # שכבת stem: מתקבל H/4 עם 64 ערוצים.
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1   # H/4,  64 ערוצים
        self.layer2 = backbone.layer2   # H/8,  128 ערוצים
        self.layer3 = backbone.layer3   # H/16, 256 ערוצים

        # קיבוע כל משקלי ה-backbone.
        for p in self.parameters():
            p.requires_grad_(False)

        # באפרים לנרמול ImageNet (עוברים אוטומטית ל-device המתאים).
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.stem(x)
        f2 = self.layer1(x)   # עדין: H/4, 64 ערוצים
        f3 = self.layer2(f2)  # ביניים: H/8, 128 ערוצים
        f4 = self.layer3(f3)  # גס/סמנטי: H/16, 256 ערוצים
        return f2, f3, f4


def transform_world_to_camera(points_world: torch.Tensor, T_w2c: torch.Tensor) -> torch.Tensor:
    """
    המרה ממערכת צירי עולם למערכת צירי מצלמה.

    points_world: (B,V,N,3)
    T_w2c:        (B,V,4,4)
    מחזיר:        (B,V,N,3) נקודות במערכת מצלמה
    """
    R = T_w2c[:, :, :3, :3]
    t = T_w2c[:, :, :3, 3]
    x_cam = torch.einsum("bvij,bvnj->bvni", R, points_world) + t.unsqueeze(2)
    return x_cam


def project_camera_points_to_pixels(points_cam: torch.Tensor, K: torch.Tensor):
    """
    הקרנת נקודות ממערכת מצלמה למישור פיקסלים.

    points_cam: (B,V,N,3)
    K:          (B,V,3,3)
    מחזיר:
      uv: (B,V,N,2) קואורדינטות פיקסלים
      z:  (B,V,N)
    """
    x = points_cam[..., 0]
    y = points_cam[..., 1]
    z = points_cam[..., 2].clamp_min(1e-6)

    fx = K[:, :, 0, 0].unsqueeze(-1)
    fy = K[:, :, 1, 1].unsqueeze(-1)
    cx = K[:, :, 0, 2].unsqueeze(-1)
    cy = K[:, :, 1, 2].unsqueeze(-1)

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    uv = torch.stack([u, v], dim=-1)
    return uv, z


def pixels_to_grid(uv: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """
    המרת קואורדינטות פיקסלים לגריד מנורמל עבור grid_sample.

    uv: (B,V,N,2) קואורדינטות פיקסלים
    פלט: (B,V,N,2) בטווח [-1,1]
    """
    u = uv[..., 0]
    v = uv[..., 1]
    gx = 2.0 * (u / max(width - 1, 1)) - 1.0
    gy = 2.0 * (v / max(height - 1, 1)) - 1.0
    return torch.stack([gx, gy], dim=-1)

# המודל הראשי של הפרויקט: עבור כל נקודה תלת מימד, נעזר ב12 תמנות, הדאטא שלהן ומשתמבש באנקודר resnet18 כדי להבין אם הנקודה נמצאת בתוך הדגם או לא
class PixelAlignedOccupancyNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        query_num_freqs: int = 0,
        query_freq_scale: float = 1.0,
        cam_pe_freqs: int = 0,
        cam_pe_scale: float = 1.0,
    ):
        super().__init__()
        self.query_num_freqs = int(query_num_freqs)
        self.query_freq_scale = float(query_freq_scale)
        self.cam_pe_freqs = int(cam_pe_freqs)
        self.cam_pe_scale = float(cam_pe_scale)
        self.image_encoder = ResNet18Encoder()

        # ממד פיצרים רב סקייל מ ResNet18.
        ms_feat_dim = 64 + 128 + 256  # = 448

        # רשת MLP לכל מבט:
        # פיצ'רים + מסכה + תקפות + uv + z_cam (עומק בלבד).
        # z_cam הוא עומק ביחס למצלמה (תלוי-מבט).
        # x_cam/y_cam לא נכנסים כדי למנוע קיצור דרך למיקום עולם.
        # קלט: ms_feat_dim + mask(1) + valid(1) + uv_norm(2) + z_cam(1)
        self.point_view_mlp = nn.Sequential(
            nn.Linear(ms_feat_dim + 5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.view_confidence = nn.Linear(hidden_dim, 1)

        # שכבת הפלט על פיצ'רים ממוזגים בלבד.
        # בלי PE במרחב עולם ובלי קיצורי דרך גאומטריים.
        # bias=0 -> sigmoid(0)=0.5 (אתחול ניטרלי ל-BCE).
        _out = nn.Linear(hidden_dim // 2, 1)
        nn.init.constant_(_out.bias, 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            _out,
        )

    def encode_cam_pe(self, q_cam: torch.Tensor) -> torch.Tensor:
        """
        נשמר להתאמת API.
        לא בשימוש כאשר cam_pe_freqs=0.
        """
        if self.cam_pe_freqs <= 0:
            return q_cam
        outs = [q_cam]
        for i in range(self.cam_pe_freqs):
            w = (2.0 ** i) * self.cam_pe_scale
            qw = q_cam * w
            outs.append(torch.sin(qw))
            outs.append(torch.cos(qw))
        return torch.cat(outs, dim=-1)

    def encode_query_world(self, q_world: torch.Tensor) -> torch.Tensor:
        """
        PE קטן במרחב עולם עבור ה-head.
        מוסיף מבנה מרחבי בלי לעודד "צורת ממוצע".
        """
        if self.query_num_freqs <= 0:
            return q_world
        outs = [q_world]
        for i in range(self.query_num_freqs):
            w = (2.0 ** i) * self.query_freq_scale
            qw = q_world * w
            outs.append(torch.sin(qw))
            outs.append(torch.cos(qw))
        return torch.cat(outs, dim=-1)

    def sample_feature_map(self, feat_map: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """
        דגימת פיצ'רים מפת-פיצ'רים לפי קואורדינטות גריד.

        feat_map: (B*V, C, Hf, Wf)
        grid:     (B*V, N, 2) מנורמל ל-[-1,1]
        מחזיר:    (B*V, N, C)
        """
        grid_4d = grid.unsqueeze(2)  # (B*V, N, 1, 2)
        sampled = F.grid_sample(
            feat_map, grid_4d,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )  # (B*V, C, N, 1)
        return sampled.squeeze(-1).transpose(1, 2).contiguous()  # (B*V, N, C)

    def sample_mask_map(self, mask: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """
        דגימת מסכה באותן נקודות דגימה.

        mask: (B*V, 1, H, W)
        grid: (B*V, N, 2)
        מחזיר: (B*V, N, 1)
        """
        grid_4d = grid.unsqueeze(2) # (B*V, N, 1, 2)
        sampled = F.grid_sample(
            mask, grid_4d,
            mode="nearest", padding_mode="zeros", align_corners=True,
        )  # (B*V, 1, N, 1)
        return sampled.squeeze(-1).transpose(1, 2).contiguous()  # (B*V, N, 1)

    def _uv_to_feat_grid(self, uv: torch.Tensor, W: int, H: int, Wf: int, Hf: int) -> torch.Tensor:
        """
        התאמת קואורדינטות uv מרזולוציית תמונה לרזולוציית מפת-פיצ'רים.
        מחזיר גריד מנורמל ל-grid_sample.
        """
        uv_s = uv.clone()
        if W > 1 and Wf > 1:
            uv_s[..., 0] = uv_s[..., 0] * ((Wf - 1) / (W - 1))
        if H > 1 and Hf > 1:
            uv_s[..., 1] = uv_s[..., 1] * ((Hf - 1) / (H - 1))
        return pixels_to_grid(uv_s, width=Wf, height=Hf)

    def forward(self, images, masks, K, T_w2c, query_points_world, return_debug: bool = False):
        """
        מעבר קדימה מלא של המודל.

        images:             (B,V,3,H,W)
        masks:              (B,V,1,H,W)
        K:                  (B,V,3,3)
        T_w2c:              (B,V,4,4)
        query_points_world: (B,N,3)
        מחזיר logits:       (B,N)
        """

        B, V, _, H, W = images.shape
        N = query_points_world.shape[1]

        # --- קידוד תמונות בכמה סקיילים ---
        x = images.reshape(B * V, 3, H, W)
        f2, f3, f4 = self.image_encoder(x)   # שלוש מפות פיצ'רים
        Hf2, Wf2 = f2.shape[2], f2.shape[3]  # H/4
        Hf3, Wf3 = f3.shape[2], f3.shape[3]  # H/8
        Hf4, Wf4 = f4.shape[2], f4.shape[3]  # H/16

        masks_bv = masks.reshape(B * V, 1, H, W)

        # --- הקרנת נקודות שאילתה לכל מבט ---
        q = query_points_world.unsqueeze(1).expand(B, V, N, 3)
        q_cam = transform_world_to_camera(q, T_w2c)            # (B,V,N,3)
        uv, z = project_camera_points_to_pixels(q_cam, K)      # (B,V,N,2), (B,V,N)

        # מסכת תקפות הקרנה (בגריד רזולוציית תמונה).
        grid_img = pixels_to_grid(uv, width=W, height=H)       # (B,V,N,2)
        valid = (
            (z > 0.0)
            & (grid_img[..., 0] >= -1.0) & (grid_img[..., 0] <= 1.0)
            & (grid_img[..., 1] >= -1.0) & (grid_img[..., 1] <= 1.0)
        )  # (B,V,N)

        # --- דגימת פיצ'רים בכל סקייל ---
        img_grid_bv  = grid_img.reshape(B * V, N, 2)
        grid_f2_bv   = self._uv_to_feat_grid(uv, W, H, Wf2, Hf2).reshape(B * V, N, 2)
        grid_f3_bv   = self._uv_to_feat_grid(uv, W, H, Wf3, Hf3).reshape(B * V, N, 2)
        grid_f4_bv   = self._uv_to_feat_grid(uv, W, H, Wf4, Hf4).reshape(B * V, N, 2)

        feat_s2 = self.sample_feature_map(f2, grid_f2_bv).reshape(B, V, N, 64)     # סקייל עדין
        feat_s3 = self.sample_feature_map(f3, grid_f3_bv).reshape(B, V, N, 128)    # סקייל ביניים
        feat_s4 = self.sample_feature_map(f4, grid_f4_bv).reshape(B, V, N, 256)    # סקייל גס
        mask_s  = self.sample_mask_map(masks_bv, img_grid_bv).reshape(B, V, N, 1)

        # --- שרשור פיצ'רים + uv + z_cam (עומק בלבד, בלי x/y של מצלמה) ---
        valid_feat = valid.unsqueeze(-1).float()   # (B,V,N,1)
        uvn = grid_img                             # (B,V,N,2)
        z_cam = q_cam[..., 2:3].clamp(min=0.01)   # (B,V,N,1) עומק תלוי-מבט

        view_feat = torch.cat([feat_s2, feat_s3, feat_s4, mask_s, valid_feat, uvn, z_cam], dim=-1)
        # ממד כולל: feature_multi_scale + מסכה + תקפות + uv + עומק.

        view_encoded = self.point_view_mlp(view_feat)              # (B,V,N,H)

        # --- מיזוג רב-מבט לפי משקלי ביטחון ---
        conf_logits = self.view_confidence(view_encoded).squeeze(-1)  # (B,V,N)
        conf_logits = torch.where(valid, conf_logits, torch.full_like(conf_logits, -1e4))
        attn = torch.softmax(conf_logits, dim=1)
        attn = attn * valid.float()
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        fused = (view_encoded * attn.unsqueeze(-1)).sum(dim=1)        # (B,N,H)

        # --- דקוד: רק פיצ'רים ויזואליים ממוזגים, בלי קיצורי דרך גאומטריים ---
        logits = self.head(fused).squeeze(-1)                         # (B,N)

        if not return_debug:
            return logits

        # פלט דיבוג קל לבדיקה ויזואלית.
        with torch.no_grad():
            # סיכום ביטחון לכל מבט (ממוצע על פני נקודות).
            view_conf = attn.mean(dim=2)  # (B, V)

            # מפות אקטיבציה מהדוגמה הראשונה והמבט הראשון.
            # ממוצע ערך מוחלט על פני ערוצים -> מפת 2D.
            idx0 = 0
            act_f2 = f2[idx0].abs().mean(dim=0).detach().cpu()  # (H/4, W/4)
            act_f3 = f3[idx0].abs().mean(dim=0).detach().cpu()  # (H/8, W/8)
            act_f4 = f4[idx0].abs().mean(dim=0).detach().cpu()  # (H/16, W/16)

            # הערוצים הבולטים ביותר בייצוג הממוזג.
            fused_mean = fused.mean(dim=1)  # (B, H)
            topk_vals, topk_idx = torch.topk(fused_mean[0].abs(), k=min(10, fused_mean.shape[1]))

        debug = {
            "view_confidence": view_conf.detach().cpu(),           # (B, V)
            "activation_maps": {
                "f2": act_f2.numpy(),
                "f3": act_f3.numpy(),
                "f4": act_f4.numpy(),
            },
            "top_channels": {
                "indices": topk_idx.detach().cpu().numpy(),
                "values": topk_vals.detach().cpu().numpy(),
            },
            "shapes": {
                "images": tuple(images.shape),
                "f2": tuple(f2.shape),
                "f3": tuple(f3.shape),
                "f4": tuple(f4.shape),
                "view_feat": tuple(view_feat.shape),
                "view_encoded": tuple(view_encoded.shape),
                "fused": tuple(fused.shape),
                "logits": tuple(logits.shape),
            },
        }
        return logits, debug
