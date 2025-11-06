import argparse
import os
from typing import Tuple, Optional

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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



def build_dataloaders_from_folder(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.0,
    split_seed: int | None = None,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], list[str]]:
    train_transforms = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
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
    _split_seed = split_seed if split_seed is not None else int(time.time() * 1000) % (2**32)
    generator = torch.Generator().manual_seed(_split_seed)
    permuted_indices = torch.randperm(num_samples, generator=generator).tolist()

    if validation_ratio + test_ratio >= 1.0:
        raise ValueError("validation_ratio + test_ratio must be < 1.0")

    num_val = int(num_samples * validation_ratio)
    num_test = int(num_samples * test_ratio)
    val_indices = permuted_indices[:num_val]
    test_indices = permuted_indices[num_val:num_val + num_test]
    train_indices = permuted_indices[num_val + num_test:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(val_dataset, test_indices) if num_test > 0 else None

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
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
    test_loader = None
    if test_subset is not None:
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return train_loader, val_loader, test_loader, train_dataset.classes


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=num_classes)
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

        if y_true.size > 0:
            prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
            metrics["precision_macro"] = float(prec)
            metrics["recall_macro"] = float(rec)

        if num_classes == 2 and y_true.size > 0 and y_prob.size:
            y_score = y_prob[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        elif num_classes > 2 and y_true.size > 0 and y_prob.size:
            y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
            if y_true_bin.shape[1] == y_prob.shape[1]:
                metrics["roc_auc"] = float(roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr"))
                metrics["pr_auc"] = float(average_precision_score(y_true_bin, y_prob, average="macro"))
    except Exception:
        pass

    return running_loss / max(1, len(data_loader)), accuracy, metrics


def plot_confusion_matrix_image(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    save_path: str,
) -> None:
    """Generate and save a confusion matrix PNG on the given dataloader.

    Args:
        model: Trained classification model.
        data_loader: Dataloader to evaluate (e.g., validation set).
        device: Torch device to run inference on.
        class_names: Ordered class names corresponding to indices.
        save_path: Output path for the PNG image.
    """
    model.eval()
    num_classes = len(class_names)
    all_true, all_pred = [], []
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_pred.append(preds)
            all_true.append(targets.numpy())

    y_true = np.concatenate(all_true, axis=0) if all_true else np.array([], dtype=int)
    y_pred = np.concatenate(all_pred, axis=0) if all_pred else np.array([], dtype=int)

    if y_true.size == 0 or y_pred.size == 0:
        print("Confusion matrix skipped: no predictions/labels available.")
        return

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix (val)')
    tick_marks = range(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    # Label each cell with count
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix image to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple training using folder structure (single-label) - Swin-Tiny")
    parser.add_argument("--data-dir", type=str, default="ImageData/images/Wound_dataset", help="Folder with class subfolders (e.g., ImageData/images/Wound_dataset)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of data for validation (default 0.2)")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Fraction of data for test (default 0.0 - disabled)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--split-seed", type=int, default=None, help="Seed for train/val split. Default: random each run")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze all but the final classifier (if supported)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pin_memory = (device.type == "cuda")

    train_loader, val_loader, test_loader, class_names = build_dataloaders_from_folder(
        data_root=args.data_dir,
        image_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
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
    optimizer = AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr)

    best_val_accuracy = -1.0
    os.makedirs("experiments/checkpoints/simple", exist_ok=True)
    checkpoint_path = os.path.join("experiments/checkpoints/simple", f"best_swin_tiny_patch4_window7_224.pt")

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
                "model_name": "swin_tiny_patch4_window7_224",
            }, checkpoint_path)
    # Optionally persist metrics (validation best + test if available)
    try:
        import json
        metrics_out = {
            "model_name": "swin_tiny_patch4_window7_224",
            "best_val_accuracy": float(best_val_accuracy),
            "roc_auc": (None if best_metrics is None else best_metrics.get("roc_auc")),
            "pr_auc": (None if best_metrics is None else best_metrics.get("pr_auc")),
            "precision_macro": (None if best_metrics is None else best_metrics.get("precision_macro")),
            "recall_macro": (None if best_metrics is None else best_metrics.get("recall_macro")),
        }
        # Evaluate on test split if provided
        if test_loader is not None:
            test_loss, test_acc, test_metrics = validate(model, test_loader, device, criterion, num_classes)
            metrics_out.update({
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
                "test_roc_auc": test_metrics.get("roc_auc"),
                "test_pr_auc": test_metrics.get("pr_auc"),
                "test_precision_macro": test_metrics.get("precision_macro"),
                "test_recall_macro": test_metrics.get("recall_macro"),
            })
        metrics_path = os.path.join("experiments", "checkpoints", "simple", "metrics_swin_tiny_patch4_window7_224.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_out, f, indent=2)
    except Exception:
        pass
    print(f"Done. Best val acc: {best_val_accuracy:.4f}. Saved to {checkpoint_path}")

    # Simple confusion matrix plot on the validation set (end of training)
    try:
        out_png = os.path.join('experiments', 'checkpoints', 'simple', 'confusion_matrix_swin_tiny.png')
        plot_confusion_matrix_image(model, val_loader, device, class_names, out_png)
    except Exception as _e:
        print(f"Could not create confusion matrix plot: {_e}")

    # Optional test confusion matrix
    if test_loader is not None:
        try:
            out_test_png = os.path.join('experiments', 'checkpoints', 'simple', 'confusion_matrix_swin_tiny_test.png')
            plot_confusion_matrix_image(model, test_loader, device, class_names, out_test_png)
        except Exception as _e:
            print(f"Could not create test confusion matrix plot: {_e}")


if __name__ == "__main__":
    main()
