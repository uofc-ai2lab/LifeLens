import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import timm  # pretrained model zoo



def build_dataloaders_from_folder(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    validation_ratio: float = 0.2,
    seed: int = 42,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, list[str]]:
    # Two datasets with different transforms for train/val but pointing to the same folder
    train_transforms = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    indices = list(range(num_samples))
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    permuted_indices = torch.randperm(num_samples, generator=generator).tolist()

    num_val = int(num_samples * validation_ratio)
    val_indices = permuted_indices[:num_val]
    train_indices = permuted_indices[num_val:]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, train_dataset.classes


def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
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


def validate(model, data_loader, device, criterion) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
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
    accuracy = correct / max(1, total)
    return running_loss / max(1, len(data_loader)), accuracy


def main():
    parser = argparse.ArgumentParser(description="Simple training using folder structure (single-label)")
    parser.add_argument("--data-dir", type=str, default="data/images/Wound_dataset", help="Folder with class subfolders (e.g., data/images/Wound_dataset)")
    parser.add_argument("--model", type=str, default="efficientnet_b3", help="Model name (e.g., efficientnet_b3, convnext_tiny, swin_tiny_patch4_window7_224)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
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
        pin_memory=pin_memory,
    )

    num_classes = len(class_names)
    model = create_model(args.model, num_classes=num_classes, pretrained=True).to(device)

    if args.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Try to unfreeze classifier head if the model exposes it
        if hasattr(model, "get_classifier"):
            classifier = model.get_classifier()
            # Some models return a Layer, others a name/string
            try:
                for param in classifier.parameters():
                    param.requires_grad = True
            except Exception:
                pass

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    best_val_accuracy = -1.0
    os.makedirs("experiments/checkpoints/simple", exist_ok=True)
    checkpoint_path = os.path.join("experiments/checkpoints/simple", f"best_{args.model}.pt")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_loss, val_accuracy = validate(model, val_loader, device, criterion)
        print(f"Epoch {epoch+1}/{args.epochs} - train_loss {train_loss:.4f} - val_loss {val_loss:.4f} - val_acc {val_accuracy:.4f}")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "model_name": args.model,
            }, checkpoint_path)
    print(f"Done. Best val acc: {best_val_accuracy:.4f}. Saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
