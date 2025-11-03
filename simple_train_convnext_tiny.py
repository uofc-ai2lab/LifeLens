import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize



def build_dataloaders_from_folder(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    validation_ratio: float = 0.2,
    split_seed: int | None = None,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, list[str]]:
    """Create train/val loaders. If split_seed is None, a different split is used each run.
    Train loader shuffles each epoch; val is deterministic.
    """
    train_transforms = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # optional; disable if vertical flips don't make sense
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomAutocontrast(p=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), value=0),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root=data_root, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=data_root, transform=val_transforms)

    import time
    num_samples = len(train_dataset.samples)
    # Pick a run-specific seed when not provided so each run can have a different split
    _split_seed = split_seed if split_seed is not None else int(time.time() * 1000) % (2**32)
    generator = torch.Generator().manual_seed(_split_seed)
    permuted_indices = torch.randperm(num_samples, generator=generator).tolist()

    num_val = int(num_samples * validation_ratio)
    val_indices = permuted_indices[:num_val]
    train_indices = permuted_indices[num_val:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,  # new shuffle every epoch
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, train_dataset.classes


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=num_classes)
    return model


def train_one_epoch(model, data_loader, device, optimizer, criterion) -> float:
    model.train()
    running_loss = 0.0
    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(1, len(data_loader))


def validate(model, data_loader, device, criterion, num_classes: int) -> Tuple[float, float, dict]:
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    accuracy = correct / max(1, total)
    # Compute additional metrics
    metrics = {
        "roc_auc": None,
        "pr_auc": None,
        "precision_macro": None,
        "recall_macro": None,
    }
    try:
        y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, num_classes))
        y_true = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0,), dtype=int)
        y_pred = y_prob.argmax(axis=1) if y_prob.size else np.array([], dtype=int)

        # Macro precision/recall from predictions
        if y_true.size > 0:
            prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
            metrics["precision_macro"] = float(prec)
            metrics["recall_macro"] = float(rec)

        # ROC-AUC and PR-AUC (Average Precision)
        if num_classes == 2 and y_true.size > 0 and y_prob.size:
            y_score = y_prob[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        elif num_classes > 2 and y_true.size > 0 and y_prob.size:
            y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
            # Some splits may miss classes; guard against shape mismatch
            if y_true_bin.shape[1] == y_prob.shape[1]:
                metrics["roc_auc"] = float(roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr"))
                metrics["pr_auc"] = float(average_precision_score(y_true_bin, y_prob, average="macro"))
    except Exception:
        # Leave metrics as None if computation fails (e.g., single-class val split)
        pass

    return running_loss / max(1, len(data_loader)), accuracy, metrics


def main():
    parser = argparse.ArgumentParser(description="Simple training using folder structure (single-label) - ConvNeXt-Tiny")
    parser.add_argument("--data-dir", type=str, default="data/images/Wound_dataset", help="Folder with class subfolders (e.g., data/images/Wound_dataset)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--split-seed", type=int, default=None, help="Seed for train/val split. Default: random each run")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze all but the final classifier (if supported)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pin_memory = (device.type == "cuda")

    train_loader, val_loader, class_names = build_dataloaders_from_folder(
        data_root=args.data_dir,
        image_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_ratio=args.val_ratio,
        split_seed=args.split_seed,
        pin_memory=pin_memory,
    )

    num_classes = len(class_names)
    model = create_model(num_classes=num_classes, pretrained=True).to(device)

    if args.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "get_classifier"):
            classifier = model.get_classifier()
            try:
                for param in classifier.parameters():
                    param.requires_grad = True
            except Exception:
                pass

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    best_val_accuracy = -1.0
    os.makedirs("experiments/checkpoints/simple", exist_ok=True)
    checkpoint_path = os.path.join("experiments/checkpoints/simple", f"best_convnext_tiny.pt")

    best_metrics = None
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_accuracy, metrics = validate(model, val_loader, device, criterion, num_classes)
        extra = []
        if metrics.get("roc_auc") is not None:
            extra.append(f"auc {metrics['roc_auc']:.4f}")
        if metrics.get("pr_auc") is not None:
            extra.append(f"pr_auc {metrics['pr_auc']:.4f}")
        if metrics.get("precision_macro") is not None and metrics.get("recall_macro") is not None:
            extra.append(f"prec {metrics['precision_macro']:.4f} recall {metrics['recall_macro']:.4f}")
        extra_str = (" - " + " - ".join(extra)) if extra else ""
        print(f"Epoch {epoch+1}/{args.epochs} - train_loss {train_loss:.4f} - val_loss {val_loss:.4f} - val_acc {val_accuracy:.4f}{extra_str}")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_metrics = metrics
            torch.save({
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "model_name": "convnext_tiny",
            }, checkpoint_path)
    # Optionally, persist metrics alongside checkpoint
    try:
        import json
        metrics_out = {
            "model_name": "convnext_tiny",
            "best_val_accuracy": float(best_val_accuracy),
            "roc_auc": (None if best_metrics is None else best_metrics.get("roc_auc")),
            "pr_auc": (None if best_metrics is None else best_metrics.get("pr_auc")),
            "precision_macro": (None if best_metrics is None else best_metrics.get("precision_macro")),
            "recall_macro": (None if best_metrics is None else best_metrics.get("recall_macro")),
        }
        metrics_path = os.path.join("experiments", "checkpoints", "simple", "metrics_convnext_tiny.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_out, f, indent=2)
    except Exception:
        pass
    print(f"Done. Best val acc: {best_val_accuracy:.4f}. Saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
