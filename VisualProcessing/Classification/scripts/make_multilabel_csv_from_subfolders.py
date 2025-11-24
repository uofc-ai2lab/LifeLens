import argparse
import csv
import os
from pathlib import Path
from collections import Counter, defaultdict
import random

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# Map folder names to canonical labels
LABEL_MAP = {
    'Abrasions': 'Abrasion',
    'Bruises': 'Bruise',
    'Burns': 'Burn',
    'Cut': 'Cut',
    'Laceration': 'Laceration',
    'Stab_wound': 'Stab_wound',  # Keep as its own class
}


def find_images(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if Path(name).suffix.lower() in SUPPORTED_EXTS:
                yield Path(dirpath) / name


def infer_label_from_path(image_path: Path, class_root: Path) -> str | None:
    try:
        rel = image_path.relative_to(class_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 2:
        return None
    folder = parts[0]
    return LABEL_MAP.get(folder)


def main():
    parser = argparse.ArgumentParser(
        description='Create stratified train/val CSVs from class subfolders with label normalization.'
                    ' Folder names are mapped via LABEL_MAP. Outputs image_path,labels columns.'
    )
    parser.add_argument('image_root', type=str, help='Root folder containing class subfolders (e.g., ImageData/images/Wound_dataset)')
    parser.add_argument('--out_dir', type=str, default='ImageData', help='Directory to write train.csv and val.csv')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio (0-1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic split')
    parser.add_argument('--relative-to', type=str, default='.', help='Write image_path relative to this directory (default: repo root)')
    args = parser.parse_args()

    random.seed(args.seed)

    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    label_counts = Counter()

    for path in find_images(image_root):
        label = infer_label_from_path(path, image_root)
        if label is None:
            continue
        label_counts[label] += 1
        try:
            relative_base = Path(args.relative_to)
            relative_path = path.relative_to(relative_base)
        except Exception:
            relative_path = path
        rows.append({'image_path': str(relative_path).replace('\\', '/'), 'labels': label})

    if not rows:
        raise SystemExit(f'No labeled images found under {image_root}')

    # Stratified shuffle/split by label
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r['labels']].append(r)

    val_rows = []
    train_rows = []
    rng = random.Random(args.seed)
    for lbl, items in by_label.items():
        rng.shuffle(items)
        n_val = int(len(items) * args.val_ratio)
        val_rows.extend(items[:n_val])
        train_rows.extend(items[n_val:])

    # Final shuffle for CSV order consistency
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    for split_name, split_rows in (('train', train_rows), ('val', val_rows)):
        out_csv = out_dir / f'{split_name}.csv'
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image_path', 'labels'])
            writer.writeheader()
            writer.writerows(split_rows)
        print(f'Wrote {len(split_rows)} rows to {out_csv}')

    print('Label counts (total):')
    for k, v in sorted(label_counts.items()):
        print(f'  {k}: {v}')

    # Per-split counts
    def split_counts(rows_list):
        c = Counter()
        for r in rows_list:
            c[r['labels']] += 1
        return c

    trc = split_counts(train_rows)
    vlc = split_counts(val_rows)
    print('Train counts:')
    for k in sorted(label_counts.keys()):
        print(f'  {k}: {trc.get(k, 0)}')
    print('Val counts:')
    for k in sorted(label_counts.keys()):
        print(f'  {k}: {vlc.get(k, 0)}')

    print('Done. Verify CSVs and adjust mapping in LABEL_MAP if needed.')


if __name__ == '__main__':
    main()
