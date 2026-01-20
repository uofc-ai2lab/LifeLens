"""Create a no_injury dataset by sampling detection crops.

This script randomly samples detection crops (which have no visible injuries)
and copies them to a new no_injury class folder for training.

The sampling is stratified by source image to ensure diversity and avoid
overfitting to specific scenes or individuals.

By default, uses an 80/20 train/test split to avoid data leakage when testing
the pipeline on the same crops later.

Usage:
    python scripts/create_no_injury_dataset.py (default run)

    python scripts/create_no_injury_dataset.py --num-samples 500 \
        --crops-root data/video/output_files/DetectionOutput/crops \
        --output-dir data/video/source_files/Images/Wound_dataset/no_injury

    # Use all samples for training (no held-out test set):
    python scripts/create_no_injury_dataset.py --train-ratio 1.0
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _collect_crops_by_source(crops_root: str) -> Dict[str, List[Path]]:
    """Group crop files by source image (parent directory)."""
    crops_root_path = Path(crops_root)
    crops_by_source: Dict[str, List[Path]] = defaultdict(list)
    
    for crop_path in crops_root_path.rglob("*.jpg"):
        if crop_path.is_file():
            source_image_id = crop_path.parent.name
            crops_by_source[source_image_id].append(crop_path)
    
    if not crops_by_source:
        raise RuntimeError(f"No crops found in {crops_root}")
    
    return crops_by_source


def _stratified_sample(
    crops_by_source: Dict[str, List[Path]],
    num_samples: int,
    seed: int | None = None,
) -> List[Path]:
    """Sample crops evenly across source images for diversity."""
    if seed is not None:
        random.seed(seed)
    
    source_ids = sorted(crops_by_source.keys())
    num_sources = len(source_ids)
    samples_per_source = max(1, num_samples // num_sources)
    
    selected_crops: List[Path] = []
    
    for source_id in source_ids:
        crops = crops_by_source[source_id]
        # Sample up to samples_per_source from this source
        num_to_sample = min(samples_per_source, len(crops))
        sampled = random.sample(crops, num_to_sample)
        selected_crops.extend(sampled)
    
    # If we don't have enough samples yet, add more randomly
    if len(selected_crops) < num_samples:
        all_crops = [c for crops in crops_by_source.values() for c in crops]
        already_selected = set(selected_crops)
        remaining = [c for c in all_crops if c not in already_selected]
        deficit = num_samples - len(selected_crops)
        additional = random.sample(remaining, min(deficit, len(remaining)))
        selected_crops.extend(additional)
    
    # Trim to exact number if we went over
    if len(selected_crops) > num_samples:
        selected_crops = random.sample(selected_crops, num_samples)
    
    return selected_crops


def _copy_crops(selected_crops: List[Path], output_dir: str) -> None:
    """Copy selected crops to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, crop_path in enumerate(selected_crops, 1):
        # Generate a unique name to avoid collisions
        dst = output_path / f"no_injury_{i:04d}.jpg"
        shutil.copy2(crop_path, dst)
        if i % 50 == 0:
            print(f"  Copied {i}/{len(selected_crops)}")
    
    print(f"  Copied {len(selected_crops)} total")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create no_injury dataset from detection crops")
    p.add_argument(
        "--num-samples",
        type=int,
        default=300,
        help="Number of crops to sample (default: 300)",
    )
    p.add_argument(
        "--crops-root",
        default="data/video/output_files/DetectionOutput/crops",
        help="Path to detection crops directory",
    )
    p.add_argument(
        "--output-dir",
        default="data/video/source_files/Images/Wound_dataset/no_injury",
        help="Output directory for no_injury class (training set)",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of sampled crops to use for training (rest held out for testing). "
             "E.g., 0.8 = 80%% train, 20%% test (default: 0.8)",
    )
    p.add_argument(
        "--test-output-dir",
        default=None,
        help="Optional: save held-out test crops to this directory. "
             "Only used if --train-ratio < 1.0",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    
    crops_root = Path(args.crops_root)
    if not crops_root.exists():
        raise FileNotFoundError(f"Crops root not found: {crops_root}")
    
    if not 0 < args.train_ratio <= 1.0:
        raise ValueError("--train-ratio must be between 0 (exclusive) and 1.0 (inclusive)")
    
    print(f"Collecting crops from: {crops_root}")
    crops_by_source = _collect_crops_by_source(str(crops_root))
    print(f"Found {len(crops_by_source)} source images with {sum(len(c) for c in crops_by_source.values())} total crops")
    
    print(f"\nStratified sampling: {args.num_samples} samples across {len(crops_by_source)} source images")
    selected_crops = _stratified_sample(crops_by_source, args.num_samples, seed=args.seed)
    print(f"Selected {len(selected_crops)} crops")
    
    # Split into train/test if requested
    if args.train_ratio < 1.0:
        num_train = int(len(selected_crops) * args.train_ratio)
        train_crops = selected_crops[:num_train]
        test_crops = selected_crops[num_train:]
        print(f"\nTrain/test split: {len(train_crops)} training, {len(test_crops)} held-out test")
    else:
        train_crops = selected_crops
        test_crops = []
    
    print(f"\nCopying training crops to: {args.output_dir}")
    _copy_crops(train_crops, args.output_dir)
    
    if test_crops:
        if args.test_output_dir:
            print(f"Copying test crops to: {args.test_output_dir}")
            _copy_crops(test_crops, args.test_output_dir)
        else:
            print(f"\nℹ️  {len(test_crops)} test crops were held out but not saved.")
            print(f"   Use --test-output-dir to save them separately if needed.")
    
    print(f"\n✓ Done!")
    if train_crops:
        print(f"  Training set: {len(train_crops)} samples in {args.output_dir}")
    if test_crops:
        print(f"  Test set: {len(test_crops)} samples (held out for pipeline testing)")
    print(f"\nReady to train: python scripts/train_video_injury_classifier.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
