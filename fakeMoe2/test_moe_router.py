import argparse
import time
import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
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


@torch.no_grad()
def run_inference(model, loader, device, warmup_batches=10, timing=False):
    model.eval()

    all_logits = []
    all_labels = []
    all_gate_weights = []

    forward_times_ms = []
    total_images_timed = 0

    pbar = tqdm(loader, desc="Test", ncols=120)

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)

        if timing and device.type == "cuda":
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            logits, gate_weights = model(
                images,
                return_aux=False
            )

            end_event.record()
            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(end_event)

        elif timing:
            start_time = time.time()

            logits, gate_weights = model(
                images,
                return_aux=False
            )

            elapsed_ms = (time.time() - start_time) * 1000.0

        else:
            logits, gate_weights = model(
                images,
                return_aux=False
            )

            elapsed_ms = None

        if timing and batch_idx >= warmup_batches:
            forward_times_ms.append(elapsed_ms)
            total_images_timed += images.size(0)

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())
        all_gate_weights.append(gate_weights.detach().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    gates = torch.cat(all_gate_weights, dim=0).numpy()

    timing_result = None

    if timing and len(forward_times_ms) > 0 and total_images_timed > 0:
        total_time_ms = sum(forward_times_ms)

        timing_result = {
            "warmup_batches": warmup_batches,
            "timed_batches": len(forward_times_ms),
            "timed_images": total_images_timed,
            "avg_batch_time_ms": total_time_ms / len(forward_times_ms),
            "avg_image_time_ms": total_time_ms / total_images_timed,
            "throughput": 1000.0 / (total_time_ms / total_images_timed),
        }

    return logits, labels, gates, timing_result


def compute_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0
    )

    p_each, r_each, f_each, support = precision_recall_fscore_support(
        labels,
        preds,
        average=None,
        zero_division=0
    )

    cm = confusion_matrix(labels, preds)

    return {
        "acc": acc,
        "balanced_acc": bal_acc,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f_macro,
        "weighted_precision": p_weighted,
        "weighted_recall": r_weighted,
        "weighted_f1": f_weighted,
        "p_each": p_each,
        "r_each": r_each,
        "f_each": f_each,
        "support": support,
        "cm": cm,
    }


def print_metrics(result, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(f"Overall Accuracy : {result['acc']:.4f}")
    print(f"Balanced Accuracy: {result['balanced_acc']:.4f}")

    print("\n--- Macro Average ---")
    print(f"Precision: {result['macro_precision']:.4f}")
    print(f"Recall   : {result['macro_recall']:.4f}")
    print(f"F1 Score : {result['macro_f1']:.4f}")

    print("\n--- Weighted Average ---")
    print(f"Precision: {result['weighted_precision']:.4f}")
    print(f"Recall   : {result['weighted_recall']:.4f}")
    print(f"F1 Score : {result['weighted_f1']:.4f}")

    print("\n--- Per-class Metrics ---")
    names = ["AE", "DMG"]

    for i, name in enumerate(names):
        print(f"{name}:")
        print(f"  Precision: {result['p_each'][i]:.4f}")
        print(f"  Recall   : {result['r_each'][i]:.4f}")
        print(f"  F1 Score : {result['f_each'][i]:.4f}")
        print(f"  Support  : {result['support'][i]}")

    cm = result["cm"]

    print("\n--- Confusion Matrix ---")
    print("Rows = GT, Cols = Pred")
    print("               AE      DMG")
    print(f"      AE   {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"     DMG   {cm[1,0]:6d}   {cm[1,1]:6d}")


def sweep_logit_margin(logits, labels, num_thresholds=3000):
    score = logits[:, 1] - logits[:, 0]

    thresholds = np.unique(
        np.quantile(score, np.linspace(0.0005, 0.9995, num_thresholds))
    )

    best_macro = None
    best_macro_f1 = -1.0

    best_bal = None
    best_bal_acc = -1.0

    for th in thresholds:
        preds = (score > th).astype(np.int64)

        result = compute_metrics(labels, preds)

        item = {
            "threshold": float(th),
            "metrics": result
        }

        if result["macro_f1"] > best_macro_f1:
            best_macro_f1 = result["macro_f1"]
            best_macro = item

        if result["balanced_acc"] > best_bal_acc:
            best_bal_acc = result["balanced_acc"]
            best_bal = item

    return best_macro, best_bal


def probability_analysis(logits, labels):
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    p_dmg = probs[:, 1]

    print("\n" + "=" * 80)
    print("Probability Distribution Analysis")
    print("=" * 80)

    for cls_idx, cls_name in [(0, "AE"), (1, "DMG")]:
        arr = p_dmg[labels == cls_idx]

        print(f"\n--- True Class: {cls_name} (n={len(arr)}) ---")
        print("DMG Probability Distribution:")
        print(f"  Mean   : {np.mean(arr):.6f}")
        print(f"  Median : {np.median(arr):.6f}")
        print(f"  Std    : {np.std(arr):.6f}")
        print(f"  Min    : {np.min(arr):.6f}")
        print(f"  Max    : {np.max(arr):.6f}")

        print("  Quantiles:")
        for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
            print(f"    {q:.2f}: {np.quantile(arr, q):.8f}")


def logit_margin_analysis(logits, labels):
    score = logits[:, 1] - logits[:, 0]

    print("\n" + "=" * 80)
    print("Logit Margin Analysis")
    print("score = logit_DMG - logit_AE")
    print("=" * 80)

    for cls_idx, cls_name in [(0, "AE"), (1, "DMG")]:
        arr = score[labels == cls_idx]

        print(f"\n--- True Class: {cls_name} (n={len(arr)}) ---")
        print(f"  Mean   : {np.mean(arr):.6f}")
        print(f"  Median : {np.median(arr):.6f}")
        print(f"  Std    : {np.std(arr):.6f}")
        print(f"  Min    : {np.min(arr):.6f}")
        print(f"  Max    : {np.max(arr):.6f}")

        print("  Quantiles:")
        for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
            print(f"    {q:.2f}: {np.quantile(arr, q):.8f}")


def gate_analysis(gates, labels):
    names = ["RGB", "NOISE", "FFT", "DCT", "BOUNDARY"]

    print("\n" + "=" * 80)
    print("Gate Weight Analysis")
    print("=" * 80)

    print("\n--- Average Gate Weights ---")
    mean = gates.mean(axis=0)
    std = gates.std(axis=0)

    for i, name in enumerate(names):
        print(f"{name:>8}: {mean[i]:.4f} +/- {std[i]:.4f}")

    print("\n--- Gate Weights by True Class ---")

    for cls_idx, cls_name in [(0, "AE"), (1, "DMG")]:
        g = gates[labels == cls_idx]

        mean = g.mean(axis=0)
        std = g.std(axis=0)

        print(f"\n{cls_name}:")
        for i, name in enumerate(names):
            print(f"{name:>8}: {mean[i]:.4f} +/- {std[i]:.4f}")


def roc_auc_analysis(logits, labels):
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    p_dmg = probs[:, 1]
    score = logits[:, 1] - logits[:, 0]
    dmg_logit = logits[:, 1]

    print("\n" + "=" * 80)
    print("ROC-AUC Analysis")
    print("=" * 80)

    try:
        auc_prob = roc_auc_score(labels, p_dmg)
        auc_margin = roc_auc_score(labels, score)
        auc_logit = roc_auc_score(labels, dmg_logit)
        pr_auc = average_precision_score(labels, score)

        print(f"Using DMG Probability       : {auc_prob:.4f}")
        print(f"Using Logit Margin DMG-AE   : {auc_margin:.4f}")
        print(f"Using DMG Logit             : {auc_logit:.4f}")
        print(f"PR-AUC using Logit Margin   : {pr_auc:.4f}")

    except Exception as e:
        print("Failed to compute ROC-AUC:", e)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--sweep", action="store_true")

    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--warmup_batches", type=int, default=10)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("MoE Router V2 Domain Test")
    print("=" * 80)
    print("Device    :", device)
    print("Test dir  :", args.test_dir)
    print("Checkpoint:", args.checkpoint)
    print("=" * 80)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    checkpoint_img_size = checkpoint.get("img_size", args.img_size)
    feature_dim = checkpoint.get("feature_dim", 256)

    if checkpoint_img_size != args.img_size:
        print(
            f"Warning: checkpoint img_size={checkpoint_img_size}, "
            f"but input img_size={args.img_size}."
        )

    _, _, test_transform = build_transforms(args.img_size)

    test_dataset = datasets.ImageFolder(
        root=args.test_dir,
        transform=test_transform
    )

    print("Test class mapping      :", test_dataset.class_to_idx)
    print("Checkpoint class_to_idx :", checkpoint.get("class_to_idx", "unknown"))
    print("Test images             :", len(test_dataset))

    checkpoint_class_to_idx = checkpoint.get("class_to_idx", None)

    if checkpoint_class_to_idx is not None:
        if checkpoint_class_to_idx != test_dataset.class_to_idx:
            print("\nWARNING:")
            print("Checkpoint class_to_idx and test_dataset class_to_idx are different.")
            print("Please make sure AE/DMG label order is correct.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = FeatureMoERouter(
        feature_dim=feature_dim,
        pretrained_rgb=False
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    print("\nLoaded checkpoint:", args.checkpoint)
    print("Checkpoint epoch :", checkpoint.get("epoch", "unknown"))
    print("Model version    :", checkpoint.get("model_version", "unknown"))
    print("Val argmax acc   :", checkpoint.get("val_argmax_acc", checkpoint.get("val_acc", "unknown")))
    print("Val macro f1     :", checkpoint.get("val_argmax_macro_f1", "unknown"))
    print("Val ROC-AUC      :", checkpoint.get("val_roc_auc", "unknown"))
    print("Saved threshold  :", checkpoint.get("best_logit_margin_threshold", "unknown"))

    logits, labels, gates, timing_result = run_inference(
        model=model,
        loader=test_loader,
        device=device,
        warmup_batches=args.warmup_batches,
        timing=args.timing
    )

    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    argmax_preds = probs.argmax(axis=1)

    argmax_result = compute_metrics(labels, argmax_preds)
    print_metrics(argmax_result, "Argmax / Default Softmax Result")

    roc_auc_analysis(logits, labels)
    probability_analysis(logits, labels)
    logit_margin_analysis(logits, labels)
    gate_analysis(gates, labels)

    threshold = args.threshold

    if threshold is None:
        threshold = checkpoint.get("best_logit_margin_threshold", None)

    if threshold is not None:
        threshold = float(threshold)
        score = logits[:, 1] - logits[:, 0]
        preds_th = (score > threshold).astype(np.int64)

        th_result = compute_metrics(labels, preds_th)

        print_metrics(
            th_result,
            f"Fixed Logit-Margin Threshold Result | threshold = {threshold:.8f}"
        )
    else:
        print("\nNo fixed threshold found. Use --threshold or train checkpoint with threshold.")

    if args.sweep:
        best_macro, best_bal = sweep_logit_margin(
            logits,
            labels,
            num_thresholds=3000
        )

        print("\n" + "=" * 80)
        print("Best Threshold by Macro-F1")
        print("=" * 80)
        print(f"Threshold: {best_macro['threshold']:.8f}")
        print_metrics(best_macro["metrics"], "Best Macro-F1 Sweep Result")

        print("\n" + "=" * 80)
        print("Best Threshold by Balanced Accuracy")
        print("=" * 80)
        print(f"Threshold: {best_bal['threshold']:.8f}")
        print_metrics(best_bal["metrics"], "Best Balanced-Acc Sweep Result")

    if timing_result is not None:
        print("\n" + "=" * 80)
        print("Inference Time")
        print("=" * 80)

        print(f"Warmup batches ignored : {timing_result['warmup_batches']}")
        print(f"Timed batches          : {timing_result['timed_batches']}")
        print(f"Timed images           : {timing_result['timed_images']}")
        print(f"Avg batch forward time : {timing_result['avg_batch_time_ms']:.4f} ms")
        print(f"Avg image forward time : {timing_result['avg_image_time_ms']:.4f} ms")
        print(f"Throughput             : {timing_result['throughput']:.2f} images/s")


if __name__ == "__main__":
    main()
