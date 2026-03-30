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
    data/video/source_files/Images/Wound_dataset/

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
import time


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

DEFAULT_DATA_DIR = "data/video/source_files/Images/Wound_dataset"
DEFAULT_SAVE_ROOT = "checkpoints/classificationModel/injury"


def _timestamp_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _pick_non_overwriting_save_root(requested: Path) -> Path:
    """Return a save root that will not overwrite an existing best checkpoint.

    If the requested directory already contains a best checkpoint (or metrics), we
    create a timestamped run subdirectory under it.
    """
    requested.mkdir(parents=True, exist_ok=True)

    existing_markers = [
        requested / "best_swin_tiny_patch4_window7_224.pt",
        requested / "metrics_swin_tiny_patch4_window7_224.json",
        requested / "training_history_swin_tiny_patch4_window7_224.json",
    ]
    if any(p.exists() for p in existing_markers):
        run_dir = requested / f"run_{_timestamp_tag()}_balanced"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    return requested


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
        default=DEFAULT_SAVE_ROOT,
        help=(
            "Directory to save checkpoint + metrics. "
            "If this directory already contains a best checkpoint, we will create a timestamped run subfolder to avoid overwriting. "
            f"(default: {DEFAULT_SAVE_ROOT})"
        ),
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
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to existing checkpoint to fine-tune from instead of starting from ImageNet weights",
    )
    p.add_argument(
        "--balanced-sampling",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Use class-balanced sampling for the training DataLoader (WeightedRandomSampler). "
            "Helps reduce bias when classes are imbalanced. (default: true)"
        ),
    )

    return p.parse_args()


def main() -> int:
    args = _parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    requested_save_root = Path(args.save_root)
    save_root = _pick_non_overwriting_save_root(requested_save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    train_swin_tiny = _load_train_swin_tiny()

    print(f"Training injury classifier from ImageFolder: {data_dir}")
    if save_root != requested_save_root:
        print(f"Saving to: {save_root} (auto-created run dir to avoid overwriting {requested_save_root})")
    else:
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
        finetune_checkpoint=args.checkpoint,
        from_detection_crops=False,
        detection_crops_root=None,
        detection_label_position=-2,
        device=None,
        balanced_sampling=bool(args.balanced_sampling),
    )

    try:
        latest_ptr = requested_save_root / "LATEST_CHECKPOINT.txt"
        ckpt_path = save_root / "best_swin_tiny_patch4_window7_224.pt"
        latest_ptr.parent.mkdir(parents=True, exist_ok=True)
        with latest_ptr.open("w", encoding="utf-8") as f:
            f.write(str(ckpt_path.resolve()))
        print(f"Wrote latest checkpoint pointer: {latest_ptr} -> {ckpt_path}")
    except Exception as e:
        print(f"Warning: could not write LATEST_CHECKPOINT.txt pointer: {e}")

    print("Done. If you want the pipeline to use this checkpoint, set PIPELINE_INJURY_CHECKPOINT to:")
    print(str(save_root / "best_swin_tiny_patch4_window7_224.pt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
