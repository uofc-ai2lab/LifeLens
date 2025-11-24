"""Swin-Tiny trainer for ImageFolder classification.
Handles split, training loop, metrics, checkpoint and confusion matrix.
"""

import argparse
import os
from typing import Tuple, Optional, Dict, Any, List

import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import time
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
    """Create train/val(/test) loaders using a shuffled index split."""
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
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = pin_memory,
    )
    # basically if we dont include a test flag in the cmd line, we return None here
    test_loader = None
    if test_subset is not None:
        test_loader = DataLoader(
            test_subset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = pin_memory,
        )
    return train_loader, val_loader, test_loader, train_dataset.classes


def build_dataloaders_from_detection_crops(
    crops_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.0,
    split_seed: int | None = None,
    pin_memory: bool = False,
    filename_delimiter: str = "_",
    label_position: int = -2,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], List[str]]:
    """Create dataloaders from detection crop directory structure.

    Expects files named like: <origstem>_<part>_<idx>.jpg (default pattern)
    label_position chooses which token from split(filename_without_ext) is treated as class label.
    label_position = -2 matches the part token in pattern above.

    Args:
        crops_root: Path to detection output 'crops' directory.
        image_size: Target resize/crop size.
        batch_size: Batch size.
        validation_ratio: Fraction for validation.
        test_ratio: Fraction for optional test set.
        split_seed: Random seed for splitting.
        filename_delimiter: Delimiter used in crop filenames.
        label_position: Index of token to extract class label.
    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
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

    exts = {".jpg", ".jpeg", ".png"}
    all_files: List[str] = []
    for root, _, files in os.walk(crops_root):
        for f in files:
            p = os.path.join(root, f)
            if os.path.splitext(f)[1].lower() in exts:
                all_files.append(p)
    if not all_files:
        raise RuntimeError(f"No crop images found under {crops_root}")
    labels: List[str] = []
    for f in all_files:
        name = os.path.splitext(os.path.basename(f))[0]
        tokens = name.split(filename_delimiter)
        if len(tokens) < abs(label_position):
            continue
        label = tokens[label_position]
        labels.append(label)
    class_names = sorted(set(labels))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    samples = [(fp, class_to_idx[os.path.splitext(os.path.basename(fp))[0].split(filename_delimiter)[label_position]]) for fp in all_files if os.path.splitext(os.path.basename(fp))[0].split(filename_delimiter)[label_position] in class_to_idx]
    if not samples:
        raise RuntimeError("No labeled samples extracted from crop filenames")

    _split_seed = split_seed if split_seed is not None else int(time.time() * 1000) % (2**32)
    g = torch.Generator().manual_seed(_split_seed)
    perm = torch.randperm(len(samples), generator=g).tolist()
    num_val = int(len(samples) * validation_ratio)
    num_test = int(len(samples) * test_ratio)
    val_idx = perm[:num_val]
    test_idx = perm[num_val:num_val+num_test]
    train_idx = perm[num_val+num_test:]

    class CropDataset(torch.utils.data.Dataset):
        def __init__(self, items: List[Tuple[str,int]], transform):
            self.items = items
            self.transform = transform
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            path, label = self.items[i]
            from PIL import Image
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label

    train_ds = CropDataset([samples[i] for i in train_idx], train_transforms)
    val_ds = CropDataset([samples[i] for i in val_idx], val_transforms)
    test_ds = CropDataset([samples[i] for i in test_idx], val_transforms) if num_test > 0 else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader, class_names


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build Swin-Tiny model for given class count."""
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=num_classes)
    return model


def train_one_epoch(model, data_loader, device, optimizer, criterion) -> float:
    """One epoch of supervised training."""
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
    """Evaluate and return loss, accuracy and a metrics dict (ROC/PR if possible)."""
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
    except (ValueError, RuntimeError, TypeError):
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


def train_swin_tiny(
    data_dir: Optional[str] = None,
    epochs: int = 5,
    batch_size: int = 32,
    img_size: int = 224,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    lr: float = 3e-4,
    split_seed: int | None = None,
    freeze_backbone: bool = False,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
    save_root: str = "experiments/checkpoints/simple",
    make_confusion_matrices: bool = True,
    from_detection_crops: bool = False,
    detection_crops_root: Optional[str] = None,
    detection_label_position: int = -2,
) -> Dict[str, Any]:
    """Train Swin Tiny on an ImageFolder dataset and return metrics.

    Minimal callable entrypoint so other scripts (e.g. detection pipeline) can import and invoke
    without depending on CLI argument parsing.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = (device.type == "cuda")
    if from_detection_crops:
        if not detection_crops_root:
            raise ValueError("detection_crops_root must be provided when from_detection_crops=True")
        train_loader, val_loader, test_loader, class_names = build_dataloaders_from_detection_crops(
            crops_root=detection_crops_root,
            image_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            validation_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed,
            pin_memory=pin_memory,
            label_position=detection_label_position,
        )
    else:
        if not data_dir:
            raise ValueError("data_dir must be provided when from_detection_crops=False")
        train_loader, val_loader, test_loader, class_names = build_dataloaders_from_folder(
            data_root=data_dir,
            image_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            validation_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed,
            pin_memory=pin_memory,
        )
    num_classes = len(class_names)
    model = create_model(num_classes=num_classes, pretrained=True).to(device)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        if hasattr(model, "get_classifier"):
            try:
                for param in classifier.parameters():
                    param.requires_grad = True
            except (AttributeError, TypeError):
                pass
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = -1.0
    os.makedirs("experiments/checkpoints/simple", exist_ok=True)
    checkpoint_path = os.path.join("experiments/checkpoints/simple", "best_swin_tiny_patch4_window7_224.pt")
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    best_metrics = None
    os.makedirs(save_root, exist_ok=True)
    ckpt_path = os.path.join(save_root, "best_swin_tiny_patch4_window7_224.pt")
    for epoch in range(epochs):
        tr_loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_acc, metrics = validate(model, val_loader, device, criterion, num_classes)
        extra = []
        if metrics.get("roc_auc") is not None:
            extra.append(f"auc {metrics['roc_auc']:.4f}")
        if metrics.get("pr_auc") is not None:
            extra.append(f"pr_auc {metrics['pr_auc']:.4f}")
        if metrics.get("precision_macro") is not None and metrics.get("recall_macro") is not None:
            extra.append(f"prec {metrics['precision_macro']:.4f} recall {metrics['recall_macro']:.4f}")
        extra_str = (" - " + " - ".join(extra)) if extra else ""
        print(f"[cls] Epoch {epoch+1}/{epochs} - train_loss {tr_loss:.4f} - val_loss {val_loss:.4f} - val_acc {val_acc:.4f}{extra_str}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
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
    except (ImportError, OSError, TypeError, ValueError, RuntimeError):
        pass
    print(f"Done. Best val acc: {best_val_accuracy:.4f}. Saved to {checkpoint_path}")

    # Simple confusion matrix plot on the validation set (end of training)
    try:
        out_png = os.path.join('experiments', 'checkpoints', 'simple', 'confusion_matrix_swin_tiny.png')
        plot_confusion_matrix_image(model, val_loader, device, class_names, out_png)
    except (OSError, ValueError, RuntimeError) as _e:
        print(f"Could not create confusion matrix plot: {_e}")

    # Optional test confusion matrix
    if test_loader is not None:
        try:
            out_test_png = os.path.join('experiments', 'checkpoints', 'simple', 'confusion_matrix_swin_tiny_test.png')
            plot_confusion_matrix_image(model, test_loader, device, class_names, out_test_png)
        except (OSError, ValueError, RuntimeError) as _e:
            print(f"Could not create test confusion matrix plot: {_e}")


if __name__ == "__main__":
    train_swin_tiny("ImageData/images/Wound_dataset", epochs=1, batch_size=8, img_size=224, val_ratio=0.2)
