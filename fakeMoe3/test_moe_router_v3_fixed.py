# test_moe_router_v3_fixed.py
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
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from moe_router_model_v3 import build_model_v3


# =========================================================
# Transform
# =========================================================

def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


# =========================================================
# Metrics
# =========================================================

def safe_auc(labels, scores):
    try:
        return roc_auc_score(labels, scores)
    except Exception:
        return float("nan")


def safe_ap(labels, scores):
    try:
        return average_precision_score(labels, scores)
    except Exception:
        return float("nan")


def compute_metrics(labels, preds, positive_scores):
    """
    labels:
      0 = AE
      1 = DMG

    positive_scores:
      score for positive class DMG.
      Higher means more likely DMG.
    """
    labels = np.asarray(labels).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    positive_scores = np.asarray(positive_scores)

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

    roc_auc = safe_auc(labels, positive_scores)
    pr_auc = safe_ap(labels, positive_scores)

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


# =========================================================
# Direction / Threshold
# =========================================================

def get_direction_scores(logits_np, probs_np, direction):
    """
    Return:
      positive_scores: higher means more likely DMG
      margins: threshold score, higher means more likely DMG
      argmax_preds: class predictions under this direction
    """

    # 默认类别：
    # class 0 = AE
    # class 1 = DMG
    logit_ae = logits_np[:, 0]
    logit_dmg = logits_np[:, 1]

    prob_ae = probs_np[:, 0]
    prob_dmg = probs_np[:, 1]

    if direction == "dmg_minus_ae":
        # 正常方向：
        # margin 越大越像 DMG
        margins = logit_dmg - logit_ae
        positive_scores = prob_dmg
        argmax_preds = np.argmax(logits_np, axis=1).astype(np.int64)

    elif direction == "ae_minus_dmg":
        # 反向方向：
        # 原模型越像 AE，实际越像 DMG。
        # 所以定义正类分数为：
        # margin = logit_AE - logit_DMG
        # margin 越大，越判 DMG。
        margins = logit_ae - logit_dmg

        # positive_scores 也要保证越大越像 DMG。
        # 这里不能用 prob_DMG，因为 prob_DMG 在反向下越大反而越像 AE。
        positive_scores = prob_ae

        # 反向 argmax：原来判 AE -> 现在判 DMG；原来判 DMG -> 现在判 AE
        raw_preds = np.argmax(logits_np, axis=1).astype(np.int64)
        argmax_preds = 1 - raw_preds

    else:
        raise ValueError(f"Unknown direction: {direction}")

    return positive_scores, margins, argmax_preds


def sweep_threshold(labels, positive_scores, margins):
    """
    margins:
      higher means more likely DMG.

    pred:
      DMG if margin >= threshold
      AE  if margin < threshold
    """
    labels = np.asarray(labels).astype(np.int64)
    positive_scores = np.asarray(positive_scores)
    margins = np.asarray(margins)

    candidates = np.unique(np.percentile(margins, np.linspace(0, 100, 1001)))

    best_f1 = -1.0
    best_f1_thr = 0.0
    best_f1_metrics = None

    best_bacc = -1.0
    best_bacc_thr = 0.0
    best_bacc_metrics = None

    for thr in candidates:
        preds = (margins >= thr).astype(np.int64)
        m = compute_metrics(labels, preds, positive_scores)

        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_f1_thr = float(thr)
            best_f1_metrics = m

        if m["balanced_acc"] > best_bacc:
            best_bacc = m["balanced_acc"]
            best_bacc_thr = float(thr)
            best_bacc_metrics = m

    return best_f1_thr, best_f1_metrics, best_bacc_thr, best_bacc_metrics


def print_direction_diagnosis(labels, logits_np, probs_np):
    labels = np.asarray(labels).astype(np.int64)

    pos_dmg, margin_dmg, pred_dmg = get_direction_scores(
        logits_np, probs_np, "dmg_minus_ae"
    )

    pos_inv, margin_inv, pred_inv = get_direction_scores(
        logits_np, probs_np, "ae_minus_dmg"
    )

    auc_prob_dmg = safe_auc(labels, probs_np[:, 1])
    auc_neg_prob_dmg = safe_auc(labels, -probs_np[:, 1])
    auc_prob_ae = safe_auc(labels, probs_np[:, 0])

    auc_margin_dmg = safe_auc(labels, margin_dmg)
    auc_margin_inv = safe_auc(labels, margin_inv)

    print("\n" + "=" * 80)
    print("Score Direction Diagnosis")
    print("=" * 80)
    print("Assumption: labels are 0=AE, 1=DMG")
    print(f"ROC-AUC prob_DMG                  : {auc_prob_dmg:.4f}")
    print(f"ROC-AUC -prob_DMG                 : {auc_neg_prob_dmg:.4f}")
    print(f"ROC-AUC prob_AE as DMG-score       : {auc_prob_ae:.4f}")
    print(f"ROC-AUC margin DMG-AE              : {auc_margin_dmg:.4f}")
    print(f"ROC-AUC margin AE-DMG              : {auc_margin_inv:.4f}")

    if auc_margin_inv > auc_margin_dmg:
        print("\n[Diagnosis] Inverted direction is better.")
        print("            当前模型在测试域上很可能出现了 score 方向反转。")
    else:
        print("\n[Diagnosis] Normal direction is better.")
        print("            当前模型分数方向与默认假设一致。")

    return {
        "auc_margin_dmg_minus_ae": auc_margin_dmg,
        "auc_margin_ae_minus_dmg": auc_margin_inv,
        "auc_prob_dmg": auc_prob_dmg,
        "auc_prob_ae": auc_prob_ae,
    }


def choose_auto_direction(labels, logits_np, probs_np):
    labels = np.asarray(labels).astype(np.int64)

    pos_dmg, margin_dmg, _ = get_direction_scores(
        logits_np, probs_np, "dmg_minus_ae"
    )

    pos_inv, margin_inv, _ = get_direction_scores(
        logits_np, probs_np, "ae_minus_dmg"
    )

    auc_dmg = safe_auc(labels, margin_dmg)
    auc_inv = safe_auc(labels, margin_inv)

    if np.isnan(auc_dmg) and np.isnan(auc_inv):
        return "dmg_minus_ae"

    if np.isnan(auc_inv):
        return "dmg_minus_ae"

    if np.isnan(auc_dmg):
        return "ae_minus_dmg"

    if auc_inv > auc_dmg:
        return "ae_minus_dmg"
    else:
        return "dmg_minus_ae"


# =========================================================
# CUDA Event Timing
# =========================================================

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
        for imgs, _ in loader:
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
    throughput = 1000.0 / avg_img_ms if avg_img_ms > 0 else 0.0

    print("\n" + "=" * 80)
    print("CUDA Event Inference Timing")
    print("=" * 80)
    print(f"Warmup batches ignored : {warmup_batches}")
    print(f"Timed batches          : {timed_batches}")
    print(f"Timed images           : {total_images}")
    print(f"Avg batch forward time : {avg_batch_ms:.4f} ms")
    print(f"Avg image forward time : {avg_img_ms:.4f} ms")
    print(f"Throughput             : {throughput:.2f} images/s")
    print("Note: timing only counts model forward, not dataloader or CPU->GPU copy.")


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--score_direction",
        type=str,
        default="auto",
        choices=["auto", "dmg_minus_ae", "ae_minus_dmg"],
        help=(
            "dmg_minus_ae: normal score, margin=logit_DMG-logit_AE. "
            "ae_minus_dmg: inverted score, margin=logit_AE-logit_DMG, treated as DMG score. "
            "auto: choose direction with higher ROC-AUC on this evaluation set."
        )
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Threshold on selected margin. "
            "Since selected margin is always defined as higher=more likely DMG, "
            "prediction is DMG if margin >= threshold."
        )
    )

    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep_both", action="store_true",
                        help="Sweep both normal and inverted score directions.")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--timing_batches", type=int, default=None)
    parser.add_argument("--warmup_batches", type=int, default=5)

    args = parser.parse_args()

    # -----------------------------------------------------
    # Device / checkpoint
    # -----------------------------------------------------
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

    print("\n" + "=" * 80)
    print("Checkpoint Info")
    print("=" * 80)
    print("Loaded checkpoint        :", args.checkpoint)
    print("Device                   :", device)
    print("img_size                 :", img_size)
    print("feature_dim              :", feature_dim)
    print("use_rgb_expert           :", use_rgb_expert)
    print("dct_num_coeff            :", dct_num_coeff)
    print("expert_width             :", expert_width)
    print("dropout                  :", dropout)
    print("Experts                  :", model.expert_names)
    print("Checkpoint class_to_idx  :", ckpt.get("class_to_idx", None))
    print("Checkpoint best_threshold:", ckpt.get("best_threshold", None))

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------
    ds = datasets.ImageFolder(args.test_dir, transform=build_transform(img_size))

    print("\n" + "=" * 80)
    print("Dataset Info")
    print("=" * 80)
    print("test_dir        :", args.test_dir)
    print("test class_to_idx:", ds.class_to_idx)
    print("Total images    :", len(ds))

    if ckpt.get("class_to_idx", None) is not None:
        if ckpt.get("class_to_idx") != ds.class_to_idx:
            print("\n[WARNING] checkpoint class_to_idx != test class_to_idx")
            print("          这可能导致标签方向或类别含义错误。")
            print("          请确认 AE=0, DMG=1 是否一致。")
        else:
            print("[OK] checkpoint/test class_to_idx match.")

    if "AE" in ds.class_to_idx and "DMG" in ds.class_to_idx:
        if ds.class_to_idx["AE"] != 0 or ds.class_to_idx["DMG"] != 1:
            print("\n[WARNING] 当前脚本默认 labels: AE=0, DMG=1。")
            print("          但 test class_to_idx 不是 {'AE':0, 'DMG':1}。")
            print("          当前指标可能需要按实际类别重映射。")
    else:
        print("\n[WARNING] 未检测到标准类别名 AE / DMG。")
        print("          当前脚本默认 class 0=AE, class 1=DMG。")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------
    labels_all = []
    logits_all = []
    probs_all = []
    gates_all = []

    t0 = time.time()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)

            logits, aux_logits, gates = model(imgs)
            probs = torch.softmax(logits, dim=1)

            labels_all.extend(labels.numpy().tolist())
            logits_all.append(logits.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            gates_all.append(gates.cpu().numpy())

    elapsed = time.time() - t0

    labels_all = np.asarray(labels_all).astype(np.int64)
    logits_all = np.concatenate(logits_all, axis=0)
    probs_all = np.concatenate(probs_all, axis=0)
    gates_all = np.concatenate(gates_all, axis=0)

    # -----------------------------------------------------
    # Direction diagnosis
    # -----------------------------------------------------
    diag = print_direction_diagnosis(labels_all, logits_all, probs_all)

    if args.score_direction == "auto":
        selected_direction = choose_auto_direction(labels_all, logits_all, probs_all)
        print("\n" + "=" * 80)
        print("Auto Direction Selection")
        print("=" * 80)
        print(f"Selected score direction: {selected_direction}")
    else:
        selected_direction = args.score_direction
        print("\n" + "=" * 80)
        print("Manual Direction Selection")
        print("=" * 80)
        print(f"Selected score direction: {selected_direction}")

    positive_scores, margins, argmax_preds = get_direction_scores(
        logits_all, probs_all, selected_direction
    )

    # -----------------------------------------------------
    # Main result: argmax or fixed threshold
    # -----------------------------------------------------
    if args.threshold is None:
        preds = argmax_preds
        title = f"Test Result | direction={selected_direction} | argmax"
    else:
        preds = (margins >= args.threshold).astype(np.int64)
        title = f"Test Result | direction={selected_direction} | threshold={args.threshold:.8f}"

    metrics = compute_metrics(labels_all, preds, positive_scores)
    print_report(title, metrics)

    # -----------------------------------------------------
    # Gate mean
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Sweep selected direction
    # -----------------------------------------------------
    if args.sweep:
        best_f1_thr, best_f1_metrics, best_bacc_thr, best_bacc_metrics = sweep_threshold(
            labels_all, positive_scores, margins
        )

        print("\n" + "=" * 80)
        print(f"Selected Direction Sweep | {selected_direction}")
        print("=" * 80)

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

    # -----------------------------------------------------
    # Sweep both directions
    # -----------------------------------------------------
    if args.sweep_both:
        for direction in ["dmg_minus_ae", "ae_minus_dmg"]:
            pos_s, mar_s, arg_s = get_direction_scores(logits_all, probs_all, direction)

            best_f1_thr, best_f1_metrics, best_bacc_thr, best_bacc_metrics = sweep_threshold(
                labels_all, pos_s, mar_s
            )

            print("\n" + "=" * 80)
            print(f"Sweep Both | Direction = {direction}")
            print("=" * 80)

            print("\n" + "=" * 80)
            print("Best Threshold by Macro-F1")
            print("=" * 80)
            print(f"Threshold: {best_f1_thr:.8f}")
            print_report(f"{direction} | Best Macro-F1 Sweep Result", best_f1_metrics)

            print("\n" + "=" * 80)
            print("Best Threshold by Balanced Accuracy")
            print("=" * 80)
            print(f"Threshold: {best_bacc_thr:.8f}")
            print_report(f"{direction} | Best Balanced-Acc Sweep Result", best_bacc_metrics)

    # -----------------------------------------------------
    # CUDA event timing
    # -----------------------------------------------------
    if args.timing:
        cuda_event_timing(
            model=model,
            loader=loader,
            device=device,
            warmup_batches=args.warmup_batches,
            max_timing_batches=args.timing_batches,
        )


if __name__ == "__main__":
    main()
