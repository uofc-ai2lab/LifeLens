"""Inference helper: run an injury classifier on detection crops.
Basically here we are going to load the checkpoints from the model we trained under visualprocessing,
from there we use that model to infer on the crops that were generated from the object detection step.

This is intended for the main pipeline:
- Object detection produces crops named like: <origstem>_<part>_<idx>.jpg
- We run an injury classifier checkpoint (trained on wound dataset classes)
  on each crop and save a JSON + CSV report.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from PIL import Image


@dataclass(frozen=True)
class CropPrediction:
    crop_path: str
    image_id: str
    body_part: str
    injury_pred: str
    injury_prob: float


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, List[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name")
    class_names = checkpoint.get("class_names")
    if not model_name or not class_names:
        raise ValueError("Checkpoint missing 'model_name' or 'class_names'.")

    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, list(class_names)


def _default_val_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class _CropsDataset(Dataset):
    def __init__(self, crop_paths: List[Path], image_size: int):
        self.crop_paths = crop_paths
        self.transform = _default_val_transforms(image_size)

    def __len__(self) -> int:
        return len(self.crop_paths)

    def __getitem__(self, index: int):
        image_path = self.crop_paths[index]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor, str(image_path)


def _infer_body_part_from_filename(path: Path, delimiter: str = "_", label_position: int = -2) -> str:
    stem = path.stem
    tokens = stem.split(delimiter)
    if len(tokens) < abs(label_position):
        return "unknown"
    return tokens[label_position]


def predict_injuries_on_detection_crops(
    crops_root: str,
    checkpoint_path: str,
    out_json_path: str,
    out_csv_path: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
    filename_delimiter: str = "_",
    body_part_label_position: int = -2,
) -> Dict[str, Any]:
    """Run injury classifier on detection crops and write reports."""

    crops_root_path = Path(crops_root)
    if not crops_root_path.exists():
        raise FileNotFoundError(f"Crops root not found: {crops_root}")

    crop_paths = sorted(
        [
            path
            for path in crops_root_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    if not crop_paths:
        raise RuntimeError(f"No crop images found under {crops_root}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, injury_class_names = load_model_from_checkpoint(checkpoint_path, device)

    dataset = _CropsDataset(crop_paths, image_size=image_size)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    predictions: List[CropPrediction] = []

    with torch.no_grad():
        for images, image_paths in data_loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predicted_index = probabilities.argmax(dim=1)
            predicted_probability = probabilities.gather(1, predicted_index.view(-1, 1)).squeeze(1)

            for batch_index in range(len(image_paths)):
                crop_path = Path(image_paths[batch_index])
                image_id = crop_path.parent.name  # crops/<image_id>/<file>.jpg
                body_part = _infer_body_part_from_filename(
                    crop_path,
                    delimiter=filename_delimiter,
                    label_position=body_part_label_position,
                )
                injury_name = injury_class_names[int(predicted_index[batch_index].item())]
                injury_probability = float(predicted_probability[batch_index].item())
                predictions.append(
                    CropPrediction(
                        crop_path=str(crop_path),
                        image_id=image_id,
                        body_part=body_part,
                        injury_pred=injury_name,
                        injury_prob=injury_probability,
                    )
                )

    # Aggregate: per (image_id, body_part) pick most common top-1, tie-break by avg prob
    aggregation_by_image_and_part: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for pred in predictions:
        key = (pred.image_id, pred.body_part)
        entry = aggregation_by_image_and_part.setdefault(key, {"counts": {}, "probabilities": {}, "count": 0})
        entry["count"] += 1
        entry["counts"][pred.injury_pred] = entry["counts"].get(pred.injury_pred, 0) + 1
        entry["probabilities"].setdefault(pred.injury_pred, []).append(pred.injury_prob)

    per_part_summary: List[Dict[str, Any]] = []
    for (image_id, body_part), entry in sorted(aggregation_by_image_and_part.items()):
        counts: Dict[str, int] = entry["counts"]
        probabilities: Dict[str, List[float]] = entry["probabilities"]
        best_injury = sorted(
            counts.keys(),
            key=lambda name: (
                counts[name],
                sum(probabilities[name]) / max(1, len(probabilities[name])),
            ),
            reverse=True,
        )[0]
        per_part_summary.append(
            {
                "image_id": image_id,
                "body_part": body_part,
                "n_crops": int(entry["count"]),
                "injury_pred": best_injury,
                "vote_count": int(counts[best_injury]),
                "avg_prob": float(sum(probabilities[best_injury]) / max(1, len(probabilities[best_injury]))),
            }
        )

    out_json = {
        "checkpoint": checkpoint_path,
        "crops_root": crops_root,
        "injury_classes": injury_class_names,
        "num_crops": len(predictions),
        "predictions": [pred.__dict__ for pred in predictions],
        "per_part_summary": per_part_summary,
    }

    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["image_id", "body_part", "n_crops", "injury_pred", "vote_count", "avg_prob"],
        )
        w.writeheader()
        for row in per_part_summary:
            w.writerow(row)

    return {
        "out_json": out_json_path,
        "out_csv": out_csv_path,
        "num_crops": len(predictions),
        "num_images": len(set(p.image_id for p in predictions)),
    }
