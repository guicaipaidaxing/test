import os
import argparse
import random
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

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
    targets = np.array([label for _, label in dataset.samples])
    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []

    for cls in np.unique(targets):
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


def get_subset_targets(dataset, indices=None):
    if indices is None:
        return np.array([label for _, label in dataset.samples])

    return np.array([dataset.samples[i][1] for i in indices])


def make_weighted_sampler_from_targets(targets):
    class_counts = np.bincount(targets)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler, class_counts


def compute_class_weights(targets, device):
    class_counts = np.bincount(targets)
    weights = class_counts.sum() / np.maximum(class_counts, 1)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32, device=device), class_counts


def gate_regularization(
    gate_weights,
    lambda_gate_balance=0.02,
    lambda_rgb_penalty=0.08,
    lambda_gate_entropy=0.005,
    rgb_target=0.38,
    rgb_index=0
):
    if gate_weights is None:
        return torch.tensor(0.0)

    eps = 1e-8
    loss = torch.tensor(0.0, device=gate_weights.device)

    num_experts = gate_weights.shape[1]

    if lambda_gate_balance > 0:
        mean_gate = gate_weights.mean(dim=0)
        target = torch.ones_like(mean_gate) / num_experts
        loss = loss + lambda_gate_balance * F.mse_loss(mean_gate, target)

    if lambda_rgb_penalty > 0:
        mean_rgb = gate_weights[:, rgb_index].mean()
        rgb_loss = F.relu(mean_rgb - rgb_target) ** 2
        loss = loss + lambda_rgb_penalty * rgb_loss

    if lambda_gate_entropy > 0:
        entropy = -(gate_weights * torch.log(gate_weights + eps)).sum(dim=1).mean()
        loss = loss - lambda_gate_entropy * entropy

    return loss


def find_best_logit_margin_threshold(logits, labels, metric="macro_f1"):
    score = logits[:, 1] - logits[:, 0]

    thresholds = np.unique(
        np.quantile(score, np.linspace(0.001, 0.999, 2000))
    )

    best = None
    best_value = -1.0

    for th in thresholds:
        preds = (score > th).astype(np.int64)

        acc = accuracy_score(labels, preds)
        bal_acc = balanced_accuracy_score(labels, preds)

        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="macro",
            zero_division=0
        )

        p_each, r_each, f_each, _ = precision_recall_fscore_support(
            labels,
            preds,
            average=None,
            zero_division=0
        )

        if metric == "balanced_acc":
            value = bal_acc
        else:
            value = f_macro

        if value > best_value:
            best_value = value
            best = {
                "threshold": float(th),
                "acc": float(acc),
                "balanced_acc": float(bal_acc),
                "macro_precision": float(p_macro),
                "macro_recall": float(r_macro),
                "macro_f1": float(f_macro),
                "p_each": p_each.tolist(),
                "r_each": r_each.tolist(),
                "f_each": f_each.tolist(),
                "confusion_matrix": confusion_matrix(labels, preds).tolist(),
            }

    return best


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    criterion,
    alpha_aux=0.3,
    lambda_gate_balance=0.02,
    lambda_rgb_penalty=0.08,
    lambda_gate_entropy=0.005,
    rgb_target=0.38,
    amp=False
):
    model.train()

    total_loss = 0.0
    total_main = 0.0
    total_aux = 0.0
    total_gate_reg = 0.0

    total_correct = 0
    total_num = 0

    pbar = tqdm(loader, desc="Train", ncols=140)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits, gate_weights, aux_logits = model(
                images,
                return_aux=True
            )

            main_loss = criterion(logits, labels)

            aux_loss = 0.0
            for aux in aux_logits:
                aux_loss = aux_loss + criterion(aux, labels)
            aux_loss = aux_loss / max(len(aux_logits), 1)

            gate_reg = gate_regularization(
                gate_weights,
                lambda_gate_balance=lambda_gate_balance,
                lambda_rgb_penalty=lambda_rgb_penalty,
                lambda_gate_entropy=lambda_gate_entropy,
                rgb_target=rgb_target,
                rgb_index=0
            )

            loss = main_loss + alpha_aux * aux_loss + gate_reg

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        scaler.step(optimizer)
        scaler.update()

        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()

        bs = images.size(0)

        total_loss += loss.item() * bs
        total_main += main_loss.item() * bs
        total_aux += aux_loss.item() * bs
        total_gate_reg += gate_reg.item() * bs

        total_correct += correct
        total_num += bs

        pbar.set_postfix({
            "loss": f"{total_loss / total_num:.4f}",
            "main": f"{total_main / total_num:.4f}",
            "aux": f"{total_aux / total_num:.4f}",
            "gate": f"{total_gate_reg / total_num:.4f}",
            "acc": f"{total_correct / total_num:.4f}",
        })

    return {
        "loss": total_loss / total_num,
        "main_loss": total_main / total_num,
        "aux_loss": total_aux / total_num,
        "gate_reg": total_gate_reg / total_num,
        "acc": total_correct / total_num,
    }


@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()

    all_logits = []
    all_labels = []
    all_gate_weights = []

    total_loss = 0.0
    total_num = 0

    pbar = tqdm(loader, desc="Val", ncols=120)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)

        logits, gate_weights = model(
            images,
            return_aux=False
        )

        loss = criterion(logits, labels_device)

        total_loss += loss.item() * images.size(0)
        total_num += images.size(0)

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())
        all_gate_weights.append(gate_weights.detach().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    gates = torch.cat(all_gate_weights, dim=0).numpy()

    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds_argmax = probs.argmax(axis=1)

    acc = accuracy_score(labels, preds_argmax)
    bal_acc = balanced_accuracy_score(labels, preds_argmax)

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        labels,
        preds_argmax,
        average="macro",
        zero_division=0
    )

    p_each, r_each, f_each, _ = precision_recall_fscore_support(
        labels,
        preds_argmax,
        average=None,
        zero_division=0
    )

    cm = confusion_matrix(labels, preds_argmax)

    score = logits[:, 1] - logits[:, 0]

    try:
        roc_auc = roc_auc_score(labels, score)
        pr_auc = average_precision_score(labels, score)
    except Exception:
        roc_auc = -1.0
        pr_auc = -1.0

    best_th = find_best_logit_margin_threshold(
        logits,
        labels,
        metric="macro_f1"
    )

    return {
        "loss": total_loss / total_num,

        "argmax_acc": acc,
        "argmax_balanced_acc": bal_acc,
        "argmax_macro_precision": p_macro,
        "argmax_macro_recall": r_macro,
        "argmax_macro_f1": f_macro,
        "argmax_p_each": p_each,
        "argmax_r_each": r_each,
        "argmax_f_each": f_each,
        "argmax_cm": cm,

        "roc_auc": roc_auc,
        "pr_auc": pr_auc,

        "logits": logits,
        "labels": labels,
        "gates": gates,
        "avg_gate": gates.mean(axis=0),
        "std_gate": gates.std(axis=0),

        "best_threshold": best_th,
    }


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    class_to_idx,
    img_size,
    feature_dim,
    args,
    val_result,
    best_score
):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,

        "class_to_idx": class_to_idx,
        "img_size": img_size,
        "feature_dim": feature_dim,

        "model_version": "moe_router_v2_domain_retrain",
        "expert_names": [
            "RGB",
            "NOISE",
            "FFT",
            "DCT",
            "BOUNDARY"
        ],

        "val_loss": val_result["loss"],
        "val_argmax_acc": val_result["argmax_acc"],
        "val_argmax_balanced_acc": val_result["argmax_balanced_acc"],
        "val_argmax_macro_f1": val_result["argmax_macro_f1"],
        "val_roc_auc": val_result["roc_auc"],
        "val_pr_auc": val_result["pr_auc"],

        "best_logit_margin_threshold": val_result["best_threshold"]["threshold"],
        "best_threshold_result": val_result["best_threshold"],

        "best_score": best_score,
        "train_args": vars(args),

        "preprocess": {
            "convert_rgb": True,
            "pad_to_square": True,
            "resize": img_size,
            "to_tensor": True,
            "external_imagenet_normalize": False
        }
    }, path)


def build_datasets(args):
    train_transform, val_transform, _ = build_transforms(args.img_size)

    if args.train_dir is not None and args.val_dir is not None:
        train_dataset = datasets.ImageFolder(
            root=args.train_dir,
            transform=train_transform
        )

        val_dataset = datasets.ImageFolder(
            root=args.val_dir,
            transform=val_transform
        )

        base_class_to_idx = train_dataset.class_to_idx

        if train_dataset.class_to_idx != val_dataset.class_to_idx:
            raise ValueError("train_dir and val_dir class_to_idx mismatch.")

        train_targets = np.array(train_dataset.targets)
        val_targets = np.array(val_dataset.targets)

        return train_dataset, val_dataset, base_class_to_idx, train_targets, val_targets

    base_dataset = datasets.ImageFolder(
        root=args.data_dir,
        transform=None
    )

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

    train_targets = get_subset_targets(base_dataset, train_indices)
    val_targets = get_subset_targets(base_dataset, val_indices)

    return train_dataset, val_dataset, base_dataset.class_to_idx, train_targets, val_targets


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--val_dir", type=str, default=None)

    parser.add_argument("--save_dir", type=str, default="./checkpoints_moe_v2_domain")

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
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")

    parser.add_argument("--lambda_gate_balance", type=float, default=0.02)
    parser.add_argument("--lambda_rgb_penalty", type=float, default=0.08)
    parser.add_argument("--lambda_gate_entropy", type=float, default=0.005)
    parser.add_argument("--rgb_target", type=float, default=0.38)

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    if args.train_dir is None or args.val_dir is None:
        if args.data_dir is None:
            raise ValueError("Either --data_dir or both --train_dir and --val_dir must be provided.")

    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"

    print("=" * 80)
    print("MoE Router V2 Domain-Robust Training")
    print("=" * 80)
    print("Device     :", device)
    print("AMP        :", use_amp)
    print("Data dir   :", args.data_dir)
    print("Train dir  :", args.train_dir)
    print("Val dir    :", args.val_dir)
    print("Save dir   :", args.save_dir)
    print("Image size :", args.img_size)
    print("Feature dim:", args.feature_dim)
    print("=" * 80)

    train_dataset, val_dataset, class_to_idx, train_targets, val_targets = build_datasets(args)

    print("Class mapping:", class_to_idx)
    print("Train images :", len(train_dataset), "counts:", np.bincount(train_targets))
    print("Val images   :", len(val_dataset), "counts:", np.bincount(val_targets))

    if class_to_idx.get("AE", None) != 0 or class_to_idx.get("DMG", None) != 1:
        print("\nWARNING: Expected class_to_idx = {'AE': 0, 'DMG': 1}.")
        print("Current:", class_to_idx)

    if args.use_weighted_sampler:
        sampler, class_counts = make_weighted_sampler_from_targets(train_targets)
        shuffle = False
        print("Using WeightedRandomSampler. Class counts:", class_counts)
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
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

    checkpoint = None

    if args.resume is not None:
        print("Resume from:", args.resume)
        checkpoint = torch.load(args.resume, map_location=device)

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )

    if args.use_class_weights:
        class_weights, class_counts = compute_class_weights(train_targets, device)
        print("Using class weights:", class_weights.detach().cpu().numpy())
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_score = -1.0

    if args.resume is not None and checkpoint is not None:
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print("Warning: failed to load optimizer:", e)

        if checkpoint.get("scheduler_state_dict", None) is not None:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                print("Warning: failed to load scheduler:", e)

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_score = checkpoint.get("best_score", checkpoint.get("val_argmax_macro_f1", -1.0))

        print("Resume epoch:", start_epoch)
        print("Resume best score:", best_score)

    for epoch in range(start_epoch, args.epochs + 1):
        print("\n" + "=" * 80)
        print(f"Epoch [{epoch}/{args.epochs}]")
        print("=" * 80)

        train_result = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            criterion=criterion,
            alpha_aux=args.alpha_aux,
            lambda_gate_balance=args.lambda_gate_balance,
            lambda_rgb_penalty=args.lambda_rgb_penalty,
            lambda_gate_entropy=args.lambda_gate_entropy,
            rgb_target=args.rgb_target,
            amp=use_amp
        )

        val_result = validate(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion
        )

        scheduler.step()

        best_th = val_result["best_threshold"]
        current_score = best_th["macro_f1"]

        print("\nEpoch Summary:")
        print(f"Train Loss: {train_result['loss']:.4f} | Train Acc: {train_result['acc']:.4f}")
        print(f"Val Loss  : {val_result['loss']:.4f}")

        print("\nVal Argmax:")
        print(f"  Acc       : {val_result['argmax_acc']:.4f}")
        print(f"  Bal Acc   : {val_result['argmax_balanced_acc']:.4f}")
        print(f"  Macro F1  : {val_result['argmax_macro_f1']:.4f}")
        print(f"  ROC-AUC   : {val_result['roc_auc']:.4f}")
        print(f"  PR-AUC    : {val_result['pr_auc']:.4f}")

        print("\nVal Best Logit-Margin Threshold:")
        print(f"  Threshold : {best_th['threshold']:.8f}")
        print(f"  Acc       : {best_th['acc']:.4f}")
        print(f"  Bal Acc   : {best_th['balanced_acc']:.4f}")
        print(f"  Macro F1  : {best_th['macro_f1']:.4f}")
        print(f"  AE  P/R/F1: {best_th['p_each'][0]:.4f} / {best_th['r_each'][0]:.4f} / {best_th['f_each'][0]:.4f}")
        print(f"  DMG P/R/F1: {best_th['p_each'][1]:.4f} / {best_th['r_each'][1]:.4f} / {best_th['f_each'][1]:.4f}")

        avg_gate = val_result["avg_gate"]
        std_gate = val_result["std_gate"]

        print("\nAverage Gate Weights:")
        print(f"RGB     : {avg_gate[0]:.4f} +/- {std_gate[0]:.4f}")
        print(f"NOISE   : {avg_gate[1]:.4f} +/- {std_gate[1]:.4f}")
        print(f"FFT     : {avg_gate[2]:.4f} +/- {std_gate[2]:.4f}")
        print(f"DCT     : {avg_gate[3]:.4f} +/- {std_gate[3]:.4f}")
        print(f"BOUNDARY: {avg_gate[4]:.4f} +/- {std_gate[4]:.4f}")

        latest_path = os.path.join(
            args.save_dir,
            "latest_moe_router_v2_domain.pth"
        )

        save_checkpoint(
            path=latest_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            class_to_idx=class_to_idx,
            img_size=args.img_size,
            feature_dim=args.feature_dim,
            args=args,
            val_result=val_result,
            best_score=best_score
        )

        if current_score > best_score:
            best_score = current_score

            best_path = os.path.join(
                args.save_dir,
                "best_moe_router_v2_domain.pth"
            )

            save_checkpoint(
                path=best_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                class_to_idx=class_to_idx,
                img_size=args.img_size,
                feature_dim=args.feature_dim,
                args=args,
                val_result=val_result,
                best_score=best_score
            )

            print(f"\nBest model saved. Best Val Macro-F1 with threshold = {best_score:.4f}")

    print("\nTraining finished.")
    print(f"Best Val Threshold Macro-F1: {best_score:.4f}")


if __name__ == "__main__":
    main()
