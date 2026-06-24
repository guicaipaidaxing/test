# test_moe_router_v3.py
import argparse
import time
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def compute_metrics(labels, preds, probs, margins):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    probs = np.asarray(probs)
    margins = np.asarray(margins)

    acc = accuracy_score(labels, preds)
    bacc = balanced_accuracy_score(labels, preds)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    p, r, f1, support = precision_recall_fscore_support(
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

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    return {
        "acc": acc,
        "balanced_acc": bacc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "per_class_precision": p,
        "per_class_recall": r,
        "per_class_f1": f1,
        "support": support,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
    }


def print_report(title, m, class_names=("AE", "DMG")):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Overall Accuracy : {m['acc']:.4f}")
    print(f"Balanced Accuracy: {m['balanced_acc']:.4f}")

    print("\n--- Macro Average ---")
    print(f"Precision: {m['macro_precision']:.4f}")
    print(f"Recall   : {m['macro_recall']:.4f}")
    print(f"F1 Score : {m['macro_f1']:.4f}")

    print("\n--- Weighted Average ---")
    print(f"Precision: {m['weighted_precision']:.4f}")
    print(f"Recall   : {m['weighted_recall']:.4f}")
    print(f"F1 Score : {m['weighted_f1']:.4f}")

    print("\n--- AUC ---")
    print(f"ROC-AUC  : {m['roc_auc']:.4f}")
    print(f"PR-AUC   : {m['pr_auc']:.4f}")

    print("\n--- Per-class Metrics ---")
    for i, name in enumerate(class_names):
        print(f"{name}:")
        print(f"  Precision: {m['per_class_precision'][i]:.4f}")
        print(f"  Recall   : {m['per_class_recall'][i]:.4f}")
        print(f"  F1 Score : {m['per_class_f1'][i]:.4f}")
        print(f"  Support  : {m['support'][i]}")

    print("\n--- Confusion Matrix ---")
    print("Rows = GT, Cols = Pred")
    print(m["confusion_matrix"])


def sweep_threshold(labels, margins, probs):
    labels = np.asarray(labels)
    margins = np.asarray(margins)
    probs = np.asarray(probs)

    candidates = np.unique(np.percentile(margins, np.linspace(0, 100, 1001)))

    best_f1 = -1
    best_f1_thr = 0
    best_f1_metrics = None

    best_bacc = -1
    best_bacc_thr = 0
    best_bacc_metrics = None

    for thr in candidates:
        preds = (margins >= thr).astype(np.int64)
        m = compute_metrics(labels, preds, probs, margins)

        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_f1_thr = float(thr)
            best_f1_metrics = m

        if m["balanced_acc"] > best_bacc:
            best_bacc = m["balanced_acc"]
            best_bacc_thr = float(thr)
            best_bacc_metrics = m

    return best_f1_thr, best_f1_metrics, best_bacc_thr, best_bacc_metrics


def cuda_event_timing(model, loader, device, warmup_batches=5, max_timing_batches=None):
    if device.type != "cuda":
        print("[Timing] CUDA not available. Skip cuda event timing.")
        return

    model.eval()

    # Warmup
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= warmup_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            _ = model(imgs)

    torch.cuda.synchronize()

    total_ms = 0.0
    total_images = 0
    timed_batches = 0

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if max_timing_batches is not None and timed_batches >= max_timing_batches:
                break

            imgs = imgs.to(device, non_blocking=True)

            starter.record()
            _ = model(imgs)
            ender.record()

            torch.cuda.synchronize()

            elapsed_ms = starter.elapsed_time(ender)
            total_ms += elapsed_ms
            total_images += imgs.size(0)
            timed_batches += 1

    avg_batch_ms = total_ms / max(timed_batches, 1)
    avg_img_ms = total_ms / max(total_images, 1)
    throughput = 1000.0 / avg_img_ms if avg_img_ms > 0 else 0

    print("\n" + "=" * 80)
    print("CUDA Event Inference Timing")
    print("=" * 80)
    print(f"Warmup batches ignored : {warmup_batches}")
    print(f"Timed batches          : {timed_batches}")
    print(f"Timed images           : {total_images}")
    print(f"Avg batch forward time : {avg_batch_ms:.4f} ms")
    print(f"Avg image forward time : {avg_img_ms:.4f} ms")
    print(f"Throughput             : {throughput:.2f} images/s")
    print("Note: timing only counts model forward, not dataloader or H2D copy.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold on logit margin: logit_DMG - logit_AE. Pred DMG if margin >= threshold.")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--timing_batches", type=int, default=None)
    parser.add_argument("--warmup_batches", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    img_size = args.img_size or ckpt.get("img_size", 256)
    feature_dim = ckpt.get("feature_dim", 256)
    use_rgb_expert = ckpt.get("use_rgb_expert", False)
    dct_num_coeff = ckpt.get("dct_num_coeff", 32)
    expert_width = ckpt.get("expert_width", 48)
    dropout = ckpt.get("dropout", 0.25)

    model = build_model_v3(
        num_classes=2,
        feature_dim=feature_dim,
        use_rgb_expert=use_rgb_expert,
        dct_num_coeff=dct_num_coeff,
        expert_width=expert_width,
        dropout=dropout,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    print("Loaded checkpoint:", args.checkpoint)
    print("Experts:", model.expert_names)
    print("Checkpoint best_threshold:", ckpt.get("best_threshold", None))

    ds = datasets.ImageFolder(args.test_dir, transform=build_transform(img_size))
    print("test class_to_idx:", ds.class_to_idx)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    labels_all = []
    preds_all = []
    probs_all = []
    margins_all = []
    gates_all = []

    t0 = time.time()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)

            logits, aux_logits, gates = model(imgs)
            probs = torch.softmax(logits, dim=1)
            margins = logits[:, 1] - logits[:, 0]

            if args.threshold is None:
                preds = torch.argmax(logits, dim=1)
            else:
                preds = (margins >= args.threshold).long()

            labels_all.extend(labels.numpy().tolist())
            preds_all.extend(preds.cpu().numpy().tolist())
            probs_all.extend(probs[:, 1].cpu().numpy().tolist())
            margins_all.extend(margins.cpu().numpy().tolist())
            gates_all.append(gates.cpu().numpy())

    elapsed = time.time() - t0

    labels_all = np.array(labels_all)
    preds_all = np.array(preds_all)
    probs_all = np.array(probs_all)
    margins_all = np.array(margins_all)
    gates_all = np.concatenate(gates_all, axis=0)

    title = "Test Result"
    if args.threshold is not None:
        title += f" | threshold={args.threshold:.8f}"
    else:
        title += " | argmax"

    metrics = compute_metrics(labels_all, preds_all, probs_all, margins_all)
    print_report(title, metrics)

    print("\n" + "=" * 80)
    print("Gate Mean")
    print("=" * 80)
    for name, val in zip(model.expert_names, gates_all.mean(axis=0)):
        print(f"{name:12s}: {val:.4f}")

    print("\n" + "=" * 80)
    print("Runtime Including DataLoader")
    print("=" * 80)
    print(f"Total images : {len(ds)}")
    print(f"Elapsed      : {elapsed:.2f} s")
    print(f"Images/s     : {len(ds) / elapsed:.2f}")

    if args.sweep:
        best_f1_thr, best_f1_metrics, best_bacc_thr, best_bacc_metrics = sweep_threshold(
            labels_all, margins_all, probs_all
        )

        print("\n" + "=" * 80)
        print("Best Threshold by Macro-F1")
        print("=" * 80)
        print(f"Threshold: {best_f1_thr:.8f}")
        print_report("Best Macro-F1 Sweep Result", best_f1_metrics)

        print("\n" + "=" * 80)
        print("Best Threshold by Balanced Accuracy")
        print("=" * 80)
        print(f"Threshold: {best_bacc_thr:.8f}")
        print_report("Best Balanced-Acc Sweep Result", best_bacc_metrics)

    if args.timing:
        cuda_event_timing(
            model,
            loader,
            device,
            warmup_batches=args.warmup_batches,
            max_timing_batches=args.timing_batches,
        )


if __name__ == "__main__":
    main()
