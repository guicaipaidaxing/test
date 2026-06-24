# extract_features_moe_router_v3.py
import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from moe_router_model_v3 import build_model_v3


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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

    print("Loaded:", args.checkpoint)
    print("Experts:", model.expert_names)

    ds = datasets.ImageFolder(args.data_dir, transform=build_transform(img_size))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    labels_all = []
    logits_all = []
    probs_all = []
    margins_all = []
    preds_all = []
    gates_all = []
    fused_all = []
    expert_features = {name: [] for name in model.expert_names}
    paths_all = [p for p, _ in ds.samples]

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)

            out = model(imgs, return_features=True)
            logits = out["logits"]
            probs = torch.softmax(logits, dim=1)
            margins = logits[:, 1] - logits[:, 0]
            preds = torch.argmax(logits, dim=1)

            labels_all.extend(labels.numpy().tolist())
            logits_all.append(logits.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            margins_all.append(margins.cpu().numpy())
            preds_all.append(preds.cpu().numpy())
            gates_all.append(out["gates"].cpu().numpy())
            fused_all.append(out["fused"].cpu().numpy())

            for name in model.expert_names:
                expert_features[name].append(out["features"][name].cpu().numpy())

    np.save(os.path.join(args.out_dir, "labels.npy"), np.array(labels_all))
    np.save(os.path.join(args.out_dir, "logits.npy"), np.concatenate(logits_all, axis=0))
    np.save(os.path.join(args.out_dir, "probs.npy"), np.concatenate(probs_all, axis=0))
    np.save(os.path.join(args.out_dir, "logit_margin.npy"), np.concatenate(margins_all, axis=0))
    np.save(os.path.join(args.out_dir, "preds.npy"), np.concatenate(preds_all, axis=0))
    np.save(os.path.join(args.out_dir, "gate_weights.npy"), np.concatenate(gates_all, axis=0))
    np.save(os.path.join(args.out_dir, "fused_features.npy"), np.concatenate(fused_all, axis=0))

    for name in model.expert_names:
        np.save(
            os.path.join(args.out_dir, f"features_{name}.npy"),
            np.concatenate(expert_features[name], axis=0),
        )

    with open(os.path.join(args.out_dir, "paths.txt"), "w", encoding="utf-8") as f:
        for p in paths_all:
            f.write(p + "\n")

    with open(os.path.join(args.out_dir, "expert_names.txt"), "w", encoding="utf-8") as f:
        for name in model.expert_names:
            f.write(name + "\n")

    print("Saved features to:", args.out_dir)


if __name__ == "__main__":
    main()
