import os
import argparse
import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from moe_router_model import (
    FeatureMoERouter,
    build_transforms
)


@torch.no_grad()
def extract_features(model, loader, device, save_dir):
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    all_f_rgb = []
    all_f_noise = []
    all_f_fft = []
    all_f_dct = []
    all_f_boundary = []
    all_f_fused = []
    all_gate = []
    all_logits = []
    all_logit_margin = []
    all_probs = []
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Extract Features", ncols=120)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)

        feature_dict = model(
            images,
            return_aux=False,
            return_features=True
        )

        logits = feature_dict["logits"]
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        logit_margin = logits[:, 1] - logits[:, 0]

        all_f_rgb.append(feature_dict["f_rgb"].detach().cpu())
        all_f_noise.append(feature_dict["f_noise"].detach().cpu())
        all_f_fft.append(feature_dict["f_fft"].detach().cpu())
        all_f_dct.append(feature_dict["f_dct"].detach().cpu())
        all_f_boundary.append(feature_dict["f_boundary"].detach().cpu())
        all_f_fused.append(feature_dict["f_fused"].detach().cpu())
        all_gate.append(feature_dict["gate_weights"].detach().cpu())
        all_logits.append(logits.detach().cpu())
        all_logit_margin.append(logit_margin.detach().cpu())
        all_probs.append(probs.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.cpu())

    all_f_rgb = torch.cat(all_f_rgb, dim=0).numpy()
    all_f_noise = torch.cat(all_f_noise, dim=0).numpy()
    all_f_fft = torch.cat(all_f_fft, dim=0).numpy()
    all_f_dct = torch.cat(all_f_dct, dim=0).numpy()
    all_f_boundary = torch.cat(all_f_boundary, dim=0).numpy()
    all_f_fused = torch.cat(all_f_fused, dim=0).numpy()
    all_gate = torch.cat(all_gate, dim=0).numpy()
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_logit_margin = torch.cat(all_logit_margin, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    np.save(os.path.join(save_dir, "f_rgb.npy"), all_f_rgb)
    np.save(os.path.join(save_dir, "f_noise.npy"), all_f_noise)
    np.save(os.path.join(save_dir, "f_fft.npy"), all_f_fft)
    np.save(os.path.join(save_dir, "f_dct.npy"), all_f_dct)
    np.save(os.path.join(save_dir, "f_boundary.npy"), all_f_boundary)
    np.save(os.path.join(save_dir, "f_fused.npy"), all_f_fused)

    np.save(os.path.join(save_dir, "gate_weights.npy"), all_gate)
    np.save(os.path.join(save_dir, "logits.npy"), all_logits)
    np.save(os.path.join(save_dir, "logit_margin.npy"), all_logit_margin)
    np.save(os.path.join(save_dir, "probs.npy"), all_probs)
    np.save(os.path.join(save_dir, "preds.npy"), all_preds)
    np.save(os.path.join(save_dir, "labels.npy"), all_labels)

    print("\nFeature extraction finished.")
    print("Saved to:", save_dir)

    print("\nFeature shapes:")
    print("f_rgb        :", all_f_rgb.shape)
    print("f_noise      :", all_f_noise.shape)
    print("f_fft        :", all_f_fft.shape)
    print("f_dct        :", all_f_dct.shape)
    print("f_boundary   :", all_f_boundary.shape)
    print("f_fused      :", all_f_fused.shape)
    print("gate         :", all_gate.shape)
    print("logits       :", all_logits.shape)
    print("logit_margin :", all_logit_margin.shape)
    print("probs        :", all_probs.shape)
    print("preds        :", all_preds.shape)
    print("labels       :", all_labels.shape)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./features_moe_v2_domain")

    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("MoE Router V2 Feature Extraction")
    print("=" * 70)
    print("Using device:", device)
    print("Data dir    :", args.data_dir)
    print("Checkpoint  :", args.checkpoint)
    print("Save dir    :", args.save_dir)
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

    dataset = datasets.ImageFolder(
        root=args.data_dir,
        transform=test_transform
    )

    print("Class mapping:", dataset.class_to_idx)
    print("Total images :", len(dataset))
    print("Checkpoint class_to_idx:", checkpoint.get("class_to_idx", "unknown"))

    loader = DataLoader(
        dataset,
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
    print("Val macro f1     :", checkpoint.get("val_argmax_macro_f1", "unknown"))
    print("Saved threshold  :", checkpoint.get("best_logit_margin_threshold", "unknown"))

    extract_features(
        model=model,
        loader=loader,
        device=device,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
