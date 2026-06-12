import argparse
import time

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from moe_router_model import (
    FeatureMoERouter,
    build_transforms
)


@torch.no_grad()
def test(model, loader, device, idx_to_class, warmup_batches=10):
    model.eval()

    total_correct = 0
    total_num = 0

    num_classes = len(idx_to_class)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    all_gate_weights = []

    forward_times_ms = []
    total_images_timed = 0

    pbar = tqdm(loader, desc="Test", ncols=120)

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if device.type == "cuda":
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

            if batch_idx >= warmup_batches:
                forward_times_ms.append(elapsed_ms)
                total_images_timed += images.size(0)

        else:
            start_time = time.time()

            logits, gate_weights = model(
                images,
                return_aux=False
            )

            elapsed_ms = (time.time() - start_time) * 1000.0

            if batch_idx >= warmup_batches:
                forward_times_ms.append(elapsed_ms)
                total_images_timed += images.size(0)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        total_correct += (preds == labels).sum().item()
        total_num += images.size(0)

        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion[t.long().cpu(), p.long().cpu()] += 1

        all_gate_weights.append(gate_weights.detach().cpu())

        pbar.set_postfix({
            "acc": f"{total_correct / total_num:.4f}"
        })

    acc = total_correct / total_num

    all_gate_weights = torch.cat(all_gate_weights, dim=0)
    avg_gate = all_gate_weights.mean(dim=0)

    print("\n" + "=" * 70)
    print("MoE Router V2 Test Result")
    print("=" * 70)

    print(f"Overall Accuracy: {acc:.4f}")

    print("\nConfusion Matrix:")
    print("Rows = GT, Cols = Pred")
    print(confusion.numpy())

    print("\nPer-class Accuracy:")
    for i in range(num_classes):
        correct = confusion[i, i].item()
        total = confusion[i, :].sum().item()

        class_acc = correct / total if total > 0 else 0.0

        print(f"{idx_to_class[i]} Acc: {class_acc:.4f}")

    print("\nAverage Gate Weights:")
    print(f"RGB     : {avg_gate[0]:.4f}")
    print(f"NOISE   : {avg_gate[1]:.4f}")
    print(f"FFT     : {avg_gate[2]:.4f}")
    print(f"DCT     : {avg_gate[3]:.4f}")
    print(f"BOUNDARY: {avg_gate[4]:.4f}")

    print("\n" + "=" * 70)
    print("Inference Time")
    print("=" * 70)

    if len(forward_times_ms) > 0 and total_images_timed > 0:
        total_time_ms = sum(forward_times_ms)

        avg_batch_time_ms = total_time_ms / len(forward_times_ms)
        avg_image_time_ms = total_time_ms / total_images_timed
        throughput = 1000.0 / avg_image_time_ms

        print(f"Warmup batches ignored : {warmup_batches}")
        print(f"Timed batches          : {len(forward_times_ms)}")
        print(f"Timed images           : {total_images_timed}")
        print(f"Avg batch forward time : {avg_batch_time_ms:.4f} ms")
        print(f"Avg image forward time : {avg_image_time_ms:.4f} ms")
        print(f"Throughput             : {throughput:.2f} images/s")

        if device.type == "cuda":
            print("\nTiming method:")
            print("torch.cuda.Event(enable_timing=True)")
            print("Only model forward is measured.")
            print("Data loading, CPU transforms, and CPU-to-GPU copy are excluded.")
        else:
            print("\nTiming method:")
            print("time.time() on CPU.")
            print("Only model forward is measured approximately.")
    else:
        print("No valid timing recorded.")
        print("Test set may be too small or warmup_batches may be too large.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--warmup_batches", type=int, default=10)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("MoE Router V2 Testing")
    print("=" * 70)
    print("Using device:", device)
    print("Test dir    :", args.test_dir)
    print("Checkpoint  :", args.checkpoint)
    print("=" * 70)

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
            print("\nWarning:")
            print("Checkpoint class_to_idx and test_dataset class_to_idx are different.")
            print("Please make sure AE/DMG label order is correct.")

    idx_to_class = {
        v: k for k, v in test_dataset.class_to_idx.items()
    }

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
    print("Checkpoint val acc:", checkpoint.get("val_acc", "unknown"))
    print("Model version    :", checkpoint.get("model_version", "unknown"))

    test(
        model=model,
        loader=test_loader,
        device=device,
        idx_to_class=idx_to_class,
        warmup_batches=args.warmup_batches
    )


if __name__ == "__main__":
    main()
