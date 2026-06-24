# train_moe_router_v3.py
import os
import json
import time
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from moe_router_model_v3 import build_model_v3


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02)
        ], p=0.3),
        transforms.ToTensor(),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return train_tf, val_tf


def compute_threshold_by_macro_f1(margins, labels):
    margins = np.asarray(margins)
    labels = np.asarray(labels)
    candidates = np.unique(np.percentile(margins, np.linspace(0, 100, 501)))

    best_thr = 0.0
    best_f1 = -1.0
    for thr in candidates:
        preds = (margins >= thr).astype(np.int64)
        mf1 = f1_score(labels, preds, average="macro", zero_division=0)
        if mf1 > best_f1:
            best_f1 = mf1
            best_thr = float(thr)
    return best_thr, best_f1


def evaluate(model, loader, device, criterion, alpha_aux=0.0, class_names=None):
    model.eval()
    losses = []
    all_labels = []
    all_preds = []
    all_probs = []
    all_margins = []
    all_gates = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, aux_logits, gates = model(imgs)
            loss = criterion(logits, labels)

            if alpha_aux > 0:
                aux_loss = 0.0
                for aux in aux_logits:
                    aux_loss = aux_loss + criterion(aux, labels)
                aux_loss = aux_loss / len(aux_logits)
                loss = loss + alpha_aux * aux_loss

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            margins = logits[:, 1] - logits[:, 0]

            losses.append(loss.item())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
            all_margins.extend(margins.cpu().numpy().tolist())
            all_gates.append(gates.cpu().numpy())

    labels = np.array(all_labels)
    preds = np.array(all_preds)
    probs = np.array(all_probs)
    margins = np.array(all_margins)
    gates = np.concatenate(all_gates, axis=0)

    acc = accuracy_score(labels, preds)
    bacc = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0
    )

    try:
        roc_auc = roc_auc_score(labels, probs)
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(labels, probs)
    except Exception:
        pr_auc = float("nan")

    best_thr, best_thr_mf1 = compute_threshold_by_macro_f1(margins, labels)

    thr_preds = (margins >= best_thr).astype(np.int64)
    thr_acc = accuracy_score(labels, thr_preds)
    thr_bacc = balanced_accuracy_score(labels, thr_preds)
    thr_macro_f1 = f1_score(labels, thr_preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    return {
        "loss": float(np.mean(losses)),
        "acc": acc,
        "balanced_acc": bacc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_threshold": best_thr,
        "best_threshold_macro_f1": best_thr_mf1,
        "threshold_acc": thr_acc,
        "threshold_balanced_acc": thr_bacc,
        "threshold_macro_f1": thr_macro_f1,
        "confusion_matrix": cm.tolist(),
        "gate_mean": gates.mean(axis=0).tolist(),
    }


def print_metrics(title, m, expert_names):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Loss              : {m['loss']:.6f}")
    print(f"Accuracy          : {m['acc']:.4f}")
    print(f"Balanced Accuracy : {m['balanced_acc']:.4f}")
    print(f"Macro-F1          : {m['macro_f1']:.4f}")
    print(f"Weighted-F1       : {m['weighted_f1']:.4f}")
    print(f"ROC-AUC           : {m['roc_auc']:.4f}")
    print(f"PR-AUC            : {m['pr_auc']:.4f}")
    print(f"Best Threshold    : {m['best_threshold']:.8f}")
    print(f"Best Thr Macro-F1 : {m['best_threshold_macro_f1']:.4f}")
    print(f"Thr Balanced Acc  : {m['threshold_balanced_acc']:.4f}")

    print("\nPer-class:")
    for i, name in enumerate(["AE", "DMG"]):
        print(f"{name}: P={m['per_class_precision'][i]:.4f}, "
              f"R={m['per_class_recall'][i]:.4f}, "
              f"F1={m['per_class_f1'][i]:.4f}, "
              f"N={m['support'][i]}")

    print("\nGate mean:")
    for n, g in zip(expert_names, m["gate_mean"]):
        print(f"  {n:12s}: {g:.4f}")

    print("\nConfusion Matrix [GT rows, Pred cols]:")
    print(np.array(m["confusion_matrix"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="ImageFolder root. Example: data/train_all/AE, data/train_all/DMG")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_moe_v3")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--expert_width", type=int, default=48)
    parser.add_argument("--dct_num_coeff", type=int, default=32)
    parser.add_argument("--use_rgb_expert", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.25)

    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--alpha_aux", type=float, default=0.25)
    parser.add_argument("--lambda_gate_balance", type=float, default=0.02)
    parser.add_argument("--lambda_gate_entropy", type=float, default=0.003)

    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_tf, val_tf = build_transforms(args.img_size)

    full_train_ds = datasets.ImageFolder(args.data_dir, transform=train_tf)
    full_val_ds = datasets.ImageFolder(args.data_dir, transform=val_tf)

    class_to_idx = full_train_ds.class_to_idx
    print("class_to_idx:", class_to_idx)

    # Require AE=0, DMG=1 ideally. If ImageFolder sorts alphabetically, AE then DMG is okay.
    targets = np.array(full_train_ds.targets)
    indices = np.arange(len(targets))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=targets,
    )

    train_ds = Subset(full_train_ds, train_idx)
    val_ds = Subset(full_val_ds, val_idx)

    if args.use_weighted_sampler:
        train_targets = targets[train_idx]
        class_counts = np.bincount(train_targets, minlength=2)
        class_weights_np = 1.0 / np.maximum(class_counts, 1)
        sample_weights = class_weights_np[train_targets]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model_v3(
        num_classes=2,
        feature_dim=args.feature_dim,
        use_rgb_expert=args.use_rgb_expert,
        dct_num_coeff=args.dct_num_coeff,
        expert_width=args.expert_width,
        dropout=args.dropout,
    ).to(device)

    print("Experts:", model.expert_names)

    if args.use_class_weights:
        class_counts = np.bincount(targets[train_idx], minlength=2)
        weights = class_counts.sum() / np.maximum(class_counts, 1)
        weights = weights / weights.mean()
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        print("Class weights:", weights)
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_macro_f1 = -1.0
    best_bacc = -1.0

    config = vars(args)
    config["class_to_idx"] = class_to_idx
    config["expert_names"] = model.expert_names
    with open(os.path.join(args.save_dir, "config_v3.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_losses = []
        train_labels = []
        train_preds = []

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits, aux_logits, gates = model(imgs)
                loss = criterion(logits, labels)

                if args.alpha_aux > 0:
                    aux_loss = 0.0
                    for aux in aux_logits:
                        aux_loss = aux_loss + criterion(aux, labels)
                    aux_loss = aux_loss / len(aux_logits)
                    loss = loss + args.alpha_aux * aux_loss

                if args.lambda_gate_balance > 0:
                    gate_mean = gates.mean(dim=0)
                    target = torch.full_like(gate_mean, 1.0 / gate_mean.numel())
                    balance_loss = F.mse_loss(gate_mean, target)
                    loss = loss + args.lambda_gate_balance * balance_loss

                if args.lambda_gate_entropy > 0:
                    entropy = -torch.sum(gates * torch.log(gates + 1e-8), dim=1).mean()
                    # Negative encourages higher entropy when added as -lambda * entropy.
                    loss = loss - args.lambda_gate_entropy * entropy

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(logits.detach(), dim=1)
            train_losses.append(loss.item())
            train_labels.extend(labels.detach().cpu().numpy().tolist())
            train_preds.extend(preds.cpu().numpy().tolist())

        scheduler.step()

        train_acc = accuracy_score(train_labels, train_preds)
        train_mf1 = f1_score(train_labels, train_preds, average="macro", zero_division=0)

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
            alpha_aux=args.alpha_aux,
        )

        elapsed = time.time() - t0

        print(f"\nEpoch [{epoch}/{args.epochs}] "
              f"time={elapsed:.1f}s "
              f"lr={optimizer.param_groups[0]['lr']:.2e} "
              f"train_loss={np.mean(train_losses):.5f} "
              f"train_acc={train_acc:.4f} "
              f"train_mf1={train_mf1:.4f}")

        print_metrics("Validation", val_metrics, model.expert_names)

        latest_path = os.path.join(args.save_dir, "latest_moe_router_v3.pth")
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
            "class_to_idx": class_to_idx,
            "expert_names": model.expert_names,
            "img_size": args.img_size,
            "feature_dim": args.feature_dim,
            "use_rgb_expert": args.use_rgb_expert,
            "dct_num_coeff": args.dct_num_coeff,
            "expert_width": args.expert_width,
            "dropout": args.dropout,
            "best_threshold": val_metrics["best_threshold"],
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, latest_path)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            path = os.path.join(args.save_dir, "best_macro_f1_moe_router_v3.pth")
            torch.save(ckpt, path)
            print(f"[SAVE] best macro-F1 checkpoint: {path}")

        if val_metrics["balanced_acc"] > best_bacc:
            best_bacc = val_metrics["balanced_acc"]
            path = os.path.join(args.save_dir, "best_balanced_acc_moe_router_v3.pth")
            torch.save(ckpt, path)
            print(f"[SAVE] best balanced-acc checkpoint: {path}")

    print("\nTraining finished.")
    print(f"Best val macro-F1: {best_macro_f1:.4f}")
    print(f"Best val balanced-acc: {best_bacc:.4f}")


if __name__ == "__main__":
    import torch.nn.functional as F
    main()
