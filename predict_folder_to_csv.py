import argparse
import os
import csv
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm


def build_eval_dataset(data_root: str, image_size: int):
    val_transforms = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=data_root, transform=val_transforms)
    return dataset


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt.get("model_name")
    class_names = ckpt.get("class_names")
    if model_name is None or class_names is None:
        raise ValueError("Checkpoint missing 'model_name' or 'class_names'.")
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, class_names


def main():
    parser = argparse.ArgumentParser(description="Run inference over a folder and export predictions to CSV")
    parser.add_argument("--data-dir", type=str, required=True, help="Folder with class subfolders (ImageFolder)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint saved by simple trainers")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to write predictions CSV; default under experiments/predictions/")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--subset", type=str, default="all", choices=["all", "train", "val"], help="Which subset to evaluate (reconstructed by seed/val-ratio)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio to reconstruct split if subset != all")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for the split")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    # Build dataset and model
    dataset = build_eval_dataset(args.data_dir, args.img_size)
    ds_classes = dataset.classes
    model, ckpt_classes = load_model_from_checkpoint(args.checkpoint, device)

    # Verify/align classes by name
    name_to_idx_ckpt = {name: i for i, name in enumerate(ckpt_classes)}
    name_to_idx_ds = {name: i for i, name in enumerate(ds_classes)}
    if set(name_to_idx_ckpt.keys()) != set(name_to_idx_ds.keys()):
        missing_in_ds = set(name_to_idx_ckpt.keys()) - set(name_to_idx_ds.keys())
        missing_in_ckpt = set(name_to_idx_ds.keys()) - set(name_to_idx_ckpt.keys())
        raise ValueError(
            f"Class mismatch between checkpoint and dataset. Missing in dataset: {missing_in_ds}; Missing in checkpoint: {missing_in_ckpt}"
        )
    # Map dataset label index -> checkpoint label index by name
    dsidx_to_ckptidx = [name_to_idx_ckpt[name] for name in ds_classes]

    # Reconstruct subset indices if needed
    all_indices: List[int] = list(range(len(dataset.samples)))
    if args.subset == "all":
        selected_indices = all_indices
    else:
        torch.manual_seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(len(all_indices), generator=generator).tolist()
        num_val = int(len(all_indices) * args.val_ratio)
        val_indices = perm[:num_val]
        train_indices = perm[num_val:]
        selected_indices = train_indices if args.subset == "train" else val_indices

    selected_samples = [dataset.samples[i] for i in selected_indices]
    subset_ds = Subset(dataset, selected_indices)
    loader = DataLoader(subset_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    # Prepare output path
    if args.output_csv is None:
        base = os.path.splitext(os.path.basename(args.checkpoint))[0]
        out_dir = os.path.join("experiments", "predictions")
        os.makedirs(out_dir, exist_ok=True)
        suffix = "" if args.subset == "all" else f"_{args.subset}"
        output_csv = os.path.join(out_dir, f"preds_{base}{suffix}.csv")
    else:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        output_csv = args.output_csv

    # Iterate and predict
    total = 0
    correct = 0
    model.eval()

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label", "pred_label", "pred_confidence", "correct"])

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                targets = targets.to(device)

                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                pred_idx = probs.argmax(dim=1)
                pred_conf = probs.gather(1, pred_idx.view(-1, 1)).squeeze(1)

                # Convert dataset label indices to checkpoint label indices for correctness
                mapped_targets = torch.tensor([dsidx_to_ckptidx[int(t)] for t in targets.tolist()], device=device)
                is_correct = (pred_idx == mapped_targets)

                for i in range(images.size(0)):
                    img_path, ds_label_idx = selected_samples[total + i]
                    true_name = ds_classes[int(ds_label_idx)]
                    pred_name = ckpt_classes[int(pred_idx[i].item())]
                    conf = float(pred_conf[i].item())
                    corr = int(is_correct[i].item())
                    writer.writerow([img_path, true_name, pred_name, f"{conf:.6f}", corr])

                correct += is_correct.sum().item()
                total += images.size(0)

    acc = (correct / total) if total > 0 else 0.0
    print(f"Wrote predictions to {output_csv}")
    print(f"Overall accuracy: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
