"""Train the injury classifier used by the video pipeline.

The video pipeline runs injury inference on detection crops using the checkpoint
configured by PIPELINE_INJURY_CHECKPOINT (see config/.env.template).

This script produces a compatible checkpoint containing:
- model_name
- class_names
- state_dict

Dataset format:
- ImageFolder directory: <data_dir>/<class_name>/*.jpg

Usage:
    python scripts/train_video_injury_classifier.py

This default assumes the Kaggle wound dataset has been extracted to:
    data/video/source_files/images/Wound_dataset/

You can override the dataset location via --data-dir.

Example:
    python scripts/train_video_injury_classifier.py --data-dir data/injury_dataset \
        --save-root checkpoints/classificationModel/injury
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = (
    PROJECT_ROOT
    / "src_video"
    / "services"
    / "classification_service"
    / "simple_train_swin_tiny.py"
)

# Keep runnable from anywhere.
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = "data/video/source_files/images/Wound_dataset"


def _load_train_swin_tiny():
    spec = importlib.util.spec_from_file_location("simple_train_swin_tiny", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load trainer module from {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "train_swin_tiny"):
        raise AttributeError("Trainer module missing train_swin_tiny")
    return module.train_swin_tiny


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the video injury classifier (ImageFolder)")
    p.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"ImageFolder root: <dir>/<class_name>/*.jpg (default: {DEFAULT_DATA_DIR})",
    )
    p.add_argument(
        "--save-root",
        default="checkpoints/classificationModel/injury",
        help="Directory to save checkpoint + metrics (default: checkpoints/classificationModel/injury)",
    )

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--test-ratio", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--split-seed", type=int, default=42)

    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--freeze-backbone-epochs", type=int, default=0)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-confusion-matrix", action="store_true")

    return p.parse_args()


def main() -> int:
    args = _parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    train_swin_tiny = _load_train_swin_tiny()

    print(f"Training injury classifier from ImageFolder: {data_dir}")
    print(f"Saving to: {save_root}")

    train_swin_tiny(
        data_dir=str(data_dir),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        img_size=int(args.img_size),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        lr=float(args.lr),
        split_seed=int(args.split_seed),
        freeze_backbone=bool(args.freeze_backbone),
        freeze_backbone_epochs=int(args.freeze_backbone_epochs),
        backbone_lr_mult=float(args.backbone_lr_mult),
        num_workers=int(args.num_workers),
        save_root=str(save_root),
        make_confusion_matrices=(not bool(args.no_confusion_matrix)),
        from_detection_crops=False,
        detection_crops_root=None,
        detection_label_position=-2,
        device=None,
    )

    print("Done. If you want the pipeline to use this checkpoint, set PIPELINE_INJURY_CHECKPOINT to:")
    print(str(save_root / "best_swin_tiny_patch4_window7_224.pt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
