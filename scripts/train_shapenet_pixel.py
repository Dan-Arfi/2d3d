#!/usr/bin/env python3
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.mv_model_occ_pixel import PixelAlignedOccupancyNet
from src.shapenet_r2n2_dataset import ShapeNetR2N2Dataset, shapenet_collate_fn


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val: float, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val":        best_val,
        "config":          config,
    }, path)


@torch.no_grad()
# שלב 6 - בדיקה על דאטא לא מאומן 
def evaluate(model, loader, device, pos_weight):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        images     = batch["images"].to(device)
        masks      = batch["masks"].to(device)
        K          = batch["K"].to(device)
        T_w2c      = batch["T_w2c"].to(device)
        occ_points = batch["occ_points"].to(device)
        occ_labels = batch["occ_labels"].to(device)

        logits = model(images, masks, K, T_w2c, occ_points)
        loss   = F.binary_cross_entropy_with_logits(logits, occ_labels, pos_weight=pos_weight)
        total += loss.item() * images.shape[0]
        n     += images.shape[0]
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",    required=True,
                        help="Root containing ShapeNetRendering/ and ShapeNetVox32/")
    parser.add_argument("--category",     default="02691156",  help="ShapeNet category ID (default: airplane)")
    parser.add_argument("--n-views",      type=int,   default=12)
    parser.add_argument("--n-query",      type=int,   default=4096)
    parser.add_argument("--batch-size",   type=int,   default=4)
    parser.add_argument("--num-workers",  type=int,   default=4)
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim",   type=int,   default=384)
    parser.add_argument("--pos-weight",   type=float, default=5.7,
                        help="BCE pos_weight; interior voxels < free space, so >1 balances")
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--out-dir",      default="checkpoints_shapenet_airplane")
    parser.add_argument("--log-every",    type=int,   default=20)
    parser.add_argument("--resume",       default=None,
                        help="Path to checkpoint to resume from (e.g. checkpoints_shapenet_v3/latest.pt)")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"[INFO] device={device}")
    # שלב 1 - הכנת הדאטא 
    train_ds = ShapeNetR2N2Dataset(
        root=args.data_root, split="train",
        category=args.category, n_views=args.n_views, n_query=args.n_query,
    )
    val_ds = ShapeNetR2N2Dataset(
        root=args.data_root, split="val",
        category=args.category, n_views=args.n_views, n_query=args.n_query,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=shapenet_collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=shapenet_collate_fn,
    )

    # ל-ResNet18 יש ממדי פלט קבועים (לא צריך image_feat_dim בפועל)
    # image_feat_dim נשמר לתאימות עם checkpoints, אבל לא משפיע בתוך המודל
    model = PixelAlignedOccupancyNet(
        image_feat_dim=256,   # informational only — ResNet18 always gives 256
        hidden_dim=args.hidden_dim,
    ).to(device)

    # מאמנים רק את ה-MLP/head; ה-backbone קפוא בתוך ResNet18Encoder
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[INFO] trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = torch.tensor(args.pos_weight, device=device)

    config = {
        "category":   args.category,
        "n_views":    args.n_views,
        "n_query":    args.n_query,
        "hidden_dim": args.hidden_dim,
        "lr":         args.lr,
        "pos_weight": args.pos_weight,
        "encoder":    "resnet18_frozen",
        "image_feat_dim": 256,
        "mode":       "bce",
    }

    best_val = float("inf")
    start_epoch = 1
    out_dir  = Path(args.out_dir)

    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        best_val    = resume_ckpt.get("best_val", float("inf"))
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        print(f"[INFO] resumed from epoch {start_epoch - 1}, best_val={best_val:.5f}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        total, n = 0.0, 0

        for step, batch in enumerate(train_loader, start=1):
            images     = batch["images"].to(device)
            masks      = batch["masks"].to(device)
            K          = batch["K"].to(device)
            T_w2c      = batch["T_w2c"].to(device)
            occ_points = batch["occ_points"].to(device)
            occ_labels = batch["occ_labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            # שלב 2
            logits = model(images, masks, K, T_w2c, occ_points)
            # שלב 3
            loss   = F.binary_cross_entropy_with_logits(logits, occ_labels, pos_weight=pos_weight)
            # שלב 4
            loss.backward()
            # שלב 5 (gradient descent)
            optimizer.step()

            total += loss.item() * images.shape[0]
            n     += images.shape[0]

            if args.log_every > 0 and step % args.log_every == 0:
                print(
                    f"[train] epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={loss.item():.5f}",
                    flush=True,
                )

        val_loss = evaluate(model, val_loader, device, pos_weight)
        sec = time.time() - t0
        print(
            f"[epoch {epoch:03d}] train_loss={total/max(n,1):.5f} "
            f"val_loss={val_loss:.5f} time={sec:.1f}s",
            flush=True,
        )
        # שמירת המשקלים שנתנו את התוצאה הטובה ביותר
        save_checkpoint(out_dir / "latest.pt", model, optimizer, epoch, best_val, config)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, best_val, config)
            print(f"[INFO] new best val={best_val:.5f} → saved", flush=True)


if __name__ == "__main__":
    main()
