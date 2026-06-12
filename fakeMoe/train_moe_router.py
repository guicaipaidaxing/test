import os
import argparse
import random
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from moe_router_model import (
    FeatureMoERouter,
    build_transforms
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


def stratified_split_from_imagefolder(dataset, val_ratio=0.2, seed=42):
    """
    对 ImageFolder 做分层划分，保证 train/val 中 AE 和 DMG 比例基本一致。

    dataset.samples:
        [(path, label), ...]
    """
    targets = np.array([label for _, label in dataset.samples])

    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []

    classes = np.unique(targets)

    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        rng.shuffle(cls_indices)

        val_size = int(len(cls_indices) * val_ratio)

        val_cls_indices = cls_indices[:val_size]
        train_cls_indices = cls_indices[val_size:]

        train_indices.extend(train_cls_indices.tolist())
        val_indices.extend(val_cls_indices.tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def train_one_epoch(model, loader, optimizer, device, alpha_aux=0.3):
    model.train()

    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_num = 0

    pbar = tqdm(loader, desc="Train", ncols=120)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits, gate_weights, aux_logits = model(
            images,
            return_aux=True
        )

        main_loss = ce(logits, labels)

        aux_loss = 0.0
        for aux in aux_logits:
            aux_loss += ce(aux, labels)

        aux_loss = aux_loss / len(aux_logits)

        loss = main_loss + alpha_aux * aux_loss

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()

        total_loss += loss.item() * images.size(0)
        total_correct += correct
        total_num += images.size(0)

        pbar.set_postfix({
            "loss": f"{total_loss / total_num:.4f}",
            "acc": f"{total_correct / total_num:.4f}"
        })

    return total_loss / total_num, total_correct / total_num


@torch.no_grad()
def validate(model, loader, device):
    model.eval()

    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_num = 0

    all_gate_weights = []

    pbar = tqdm(loader, desc="Val", ncols=120)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, gate_weights = model(
            images,
            return_aux=False
        )

        loss = ce(logits, labels)

        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()

        total_loss += loss.item() * images.size(0)
        total_correct += correct
        total_num += images.size(0)

        all_gate_weights.append(gate_weights.detach().cpu())

        pbar.set_postfix({
            "loss": f"{total_loss / total_num:.4f}",
            "acc": f"{total_correct / total_num:.4f}"
        })

    all_gate_weights = torch.cat(all_gate_weights, dim=0)
    avg_gate = all_gate_weights.mean(dim=0)

    return total_loss / total_num, total_correct / total_num, avg_gate


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    val_acc,
    class_to_idx,
    img_size,
    feature_dim,
    args
):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "val_acc": val_acc,
        "class_to_idx": class_to_idx,
        "img_size": img_size,
        "feature_dim": feature_dim,
        "model_version": "moe_router_v2",
        "expert_names": [
            "RGB",
            "NOISE",
            "FFT",
            "DCT",
            "BOUNDARY"
        ],
        "train_args": vars(args),
        "preprocess": {
            "convert_rgb": True,
            "pad_to_square": True,
            "resize": img_size,
            "to_tensor": True
        }
    }, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_moe_v2")

    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--feature_dim", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    parser.add_argument("--alpha_aux", type=float, default=0.3)

    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("MoE Router V2 Training")
    print("=" * 70)
    print("Using device :", device)
    print("Data dir     :", args.data_dir)
    print("Save dir     :", args.save_dir)
    print("Image size   :", args.img_size)
    print("Feature dim  :", args.feature_dim)
    print("=" * 70)

    train_transform, val_transform, _ = build_transforms(args.img_size)

    base_dataset = datasets.ImageFolder(
        root=args.data_dir,
        transform=None
    )

    print("Class mapping:", base_dataset.class_to_idx)
    print("Total images :", len(base_dataset))

    train_indices, val_indices = stratified_split_from_imagefolder(
        dataset=base_dataset,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    train_dataset_full = datasets.ImageFolder(
        root=args.data_dir,
        transform=train_transform
    )

    val_dataset_full = datasets.ImageFolder(
        root=args.data_dir,
        transform=val_transform
    )

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    print("Train images :", len(train_dataset))
    print("Val images   :", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = FeatureMoERouter(
        feature_dim=args.feature_dim,
        pretrained_rgb=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    start_epoch = 1
    best_acc = 0.0

    if args.resume is not None:
        print("Resume from:", args.resume)
        checkpoint = torch.load(args.resume, map_location=device)

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if checkpoint.get("scheduler_state_dict", None) is not None:
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_acc = checkpoint.get("val_acc", 0.0)

        print("Resume epoch:", start_epoch)
        print("Resume best acc:", best_acc)

    for epoch in range(start_epoch, args.epochs + 1):
        print("\n" + "=" * 70)
        print(f"Epoch [{epoch}/{args.epochs}]")
        print("=" * 70)

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha_aux=args.alpha_aux
        )

        val_loss, val_acc, avg_gate = validate(
            model=model,
            loader=val_loader,
            device=device
        )

        scheduler.step()

        print("\nEpoch Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        print(
            "Average Gate Weights: "
            f"RGB={avg_gate[0]:.4f}, "
            f"NOISE={avg_gate[1]:.4f}, "
            f"FFT={avg_gate[2]:.4f}, "
            f"DCT={avg_gate[3]:.4f}, "
            f"BOUNDARY={avg_gate[4]:.4f}"
        )

        latest_path = os.path.join(
            args.save_dir,
            "latest_moe_router_v2.pth"
        )

        save_checkpoint(
            path=latest_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            val_acc=val_acc,
            class_to_idx=base_dataset.class_to_idx,
            img_size=args.img_size,
            feature_dim=args.feature_dim,
            args=args
        )

        if val_acc > best_acc:
            best_acc = val_acc

            best_path = os.path.join(
                args.save_dir,
                "best_moe_router_v2.pth"
            )

            save_checkpoint(
                path=best_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                val_acc=val_acc,
                class_to_idx=base_dataset.class_to_idx,
                img_size=args.img_size,
                feature_dim=args.feature_dim,
                args=args
            )

            print(f"Best model saved. Best Val Acc = {best_acc:.4f}")

    print("\nTraining finished.")
    print(f"Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
