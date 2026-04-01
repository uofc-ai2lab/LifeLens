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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
from torchvision import transforms
import timm
from PIL import Image

from config.logger import Logger

log = Logger("[video][classification]")


def _is_excluded_injury_label(label: str) -> bool:
    return label.strip().lower() == "ingrown nails"


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


def _infer_body_part_from_filename(path: Path, delimiter: str = "_", label_position: int = -2) -> str:
    stem = path.stem
    tokens = stem.split(delimiter)
    if len(tokens) < abs(label_position):
        return "unknown"
    return tokens[label_position]


def _iter_crop_paths(crops_root: str) -> Iterator[Path]:
    """Yield crop image paths recursively without materializing full path lists."""
    crops_root_path = Path(crops_root)
    if not crops_root_path.exists():
        raise FileNotFoundError(f"Crops root not found: {crops_root}")
    yielded = 0
    for path in sorted(crops_root_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        # Ignore alpha-masked helper exports; they are not classifier inputs and
        # their filename suffix breaks body_part parsing (e.g. *_arm1_0_alpha.png).
        if path.stem.lower().endswith("_alpha"):
            continue
        yielded += 1
        yield path
    if yielded == 0:
        raise RuntimeError(f"No crop images found under {crops_root}")


def _update_aggregation(
    aggregation_by_image_and_part: Dict[Tuple[str, str], Dict[str, Any]],
    *,
    image_id: str,
    body_part: str,
    injury_name: str,
    injury_probability: float,
) -> None:
    key = (image_id, body_part)
    entry = aggregation_by_image_and_part.setdefault(
        key,
        {
            "count": 0,
            "counts": {},
            "prob_sums": {},
            "prob_max": {},
        },
    )
    entry["count"] += 1

    counts: Dict[str, int] = entry["counts"]
    prob_sums: Dict[str, float] = entry["prob_sums"]
    prob_max: Dict[str, float] = entry["prob_max"]

    counts[injury_name] = counts.get(injury_name, 0) + 1
    prob_sums[injury_name] = prob_sums.get(injury_name, 0.0) + injury_probability
    prob_max[injury_name] = max(prob_max.get(injury_name, float("-inf")), injury_probability)


def _aggregate_predictions_by_part(
    aggregation_by_image_and_part: Dict[Tuple[str, str], Dict[str, Any]],
    use_max_non_no_injury: bool = False,
    no_injury_label: str = "no_injury",
) -> List[Dict[str, Any]]:
    """Aggregate predictions by (image_id, body_part) using compact running stats."""
    per_part_summary: List[Dict[str, Any]] = []
    for (image_id, body_part), entry in sorted(aggregation_by_image_and_part.items()):
        counts: Dict[str, int] = {
            name: value for name, value in entry["counts"].items() if not _is_excluded_injury_label(name)
        }
        prob_sums: Dict[str, float] = {
            name: value for name, value in entry["prob_sums"].items() if name in counts
        }
        prob_max: Dict[str, float] = {
            name: value for name, value in entry["prob_max"].items() if name in counts
        }

        if not counts:
            continue

        best_injury = ""
        if use_max_non_no_injury:
            non_no_injury_classes = [
                name for name in counts.keys() if name.lower() != no_injury_label.lower()
            ]
            if non_no_injury_classes:
                best_injury = max(
                    non_no_injury_classes,
                    key=lambda name: prob_max.get(name, 0.0),
                )

        if not best_injury:
            best_injury = sorted(
                counts.keys(),
                key=lambda name: (
                    counts[name],
                    prob_sums.get(name, 0.0) / max(1, counts[name]),
                ),
                reverse=True,
            )[0]

        avg_prob = float(prob_sums.get(best_injury, 0.0) / max(1, counts[best_injury]))

        per_part_summary.append(
            {
                "image_id": image_id,
                "body_part": body_part,
                "n_crops": int(entry["count"]),
                "injury_pred": best_injury,
                "vote_count": int(counts[best_injury]),
                "avg_prob": avg_prob,
            }
        )
    return per_part_summary


def _save_reports(
    out_json_path: str,
    out_csv_path: str,
    checkpoint_path: str,
    crops_root: str,
    injury_class_names: List[str],
    num_crops: int,
    per_part_summary: List[Dict[str, Any]],
) -> None:
    """Save JSON and CSV reports."""
    injury_class_names = [name for name in injury_class_names if not _is_excluded_injury_label(name)]
    out_json = {
        "checkpoint": checkpoint_path,
        "crops_root": crops_root,
        "injury_classes": injury_class_names,
        "num_crops": int(num_crops),
        # Intentionally omitted for memory safety on large runs.
        "predictions": [],
        "predictions_omitted": True,
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
    use_max_non_no_injury_aggregation: bool = False,
    no_injury_label: str = "no_injury",
) -> Dict[str, Any]:
    """Run injury classifier on detection crops and write reports."""
    log.header("Starting Injury Classification...")

    checkpoint_file = Path(checkpoint_path).expanduser()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Injury checkpoint not found: {checkpoint_file.resolve()}")

    checkpoint_resolved = str(checkpoint_file.resolve())
    checkpoint_stats = checkpoint_file.stat()
    checkpoint_size_mb = checkpoint_stats.st_size / (1024 * 1024)
    checkpoint_mtime = datetime.fromtimestamp(checkpoint_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    log.info(
        f"Using injury checkpoint: {checkpoint_resolved} "
        f"({checkpoint_size_mb:.2f} MB, mtime={checkpoint_mtime})"
    )
    
    if num_workers:
        log.info("num_workers is ignored in streaming mode")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, injury_class_names = load_model_from_checkpoint(checkpoint_resolved, device)

    transform = _default_val_transforms(image_size)
    aggregation_by_image_and_part: Dict[Tuple[str, str], Dict[str, Any]] = {}
    unique_image_ids: set[str] = set()

    batch_tensors: List[torch.Tensor] = []
    batch_paths: List[Path] = []
    processed_crops = 0

    def _flush_batch() -> int:
        if not batch_tensors:
            return 0
        images = torch.stack(batch_tensors, dim=0).to(
            device,
            non_blocking=(device.type == "cuda"),
        )
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = probabilities.argmax(dim=1)
        predicted_probability = probabilities.gather(1, predicted_index.view(-1, 1)).squeeze(1)

        for i, crop_path in enumerate(batch_paths):
            image_id = crop_path.parent.name  # crops/<session>/<image_id>/<file>.jpg
            unique_image_ids.add(image_id)

            body_part = _infer_body_part_from_filename(
                crop_path,
                delimiter=filename_delimiter,
                label_position=body_part_label_position,
            )
            injury_name = injury_class_names[int(predicted_index[i].item())]
            injury_probability = float(predicted_probability[i].item())

            _update_aggregation(
                aggregation_by_image_and_part,
                image_id=image_id,
                body_part=body_part,
                injury_name=injury_name,
                injury_probability=injury_probability,
            )

        count = len(batch_paths)
        batch_tensors.clear()
        batch_paths.clear()
        return count

    with torch.no_grad():
        for crop_path in _iter_crop_paths(crops_root):
            image = Image.open(crop_path).convert("RGB")
            image_tensor = transform(image)
            batch_tensors.append(image_tensor)
            batch_paths.append(crop_path)

            if len(batch_tensors) >= max(1, int(batch_size)):
                processed_crops += _flush_batch()

        processed_crops += _flush_batch()

    log.info(f"Collected {processed_crops} crops")

    per_part_summary = _aggregate_predictions_by_part(
        aggregation_by_image_and_part,
        use_max_non_no_injury=use_max_non_no_injury_aggregation,
        no_injury_label=no_injury_label,
    )

    _save_reports(
        out_json_path=out_json_path,
        out_csv_path=out_csv_path,
        checkpoint_path=checkpoint_resolved,
        crops_root=crops_root,
        injury_class_names=injury_class_names,
        num_crops=processed_crops,
        per_part_summary=per_part_summary,
    )

    log.success(f"Classification complete: {processed_crops} crops")

    return {
        "out_json": out_json_path,
        "out_csv": out_csv_path,
        "num_crops": processed_crops,
        "num_images": len(unique_image_ids),
    }
