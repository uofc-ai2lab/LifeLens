Image data layout

The Swin-Tiny trainer performs its own train/val (and optional test) split directly from an ImageFolder structure.

Directory structure:
```
VisualProcessing/Classification/ImageData/images/Wound_dataset/
  Abrasion/
    img001.jpg
    ...
  Bruise/
  Burn/
  Cut/
  Laceration/
  Stab_wound/
  Normal skin/
```

Run:
```bash
python VisualProcessing/Classification/ClassificationModels/simple_train_swin_tiny.py \
  --data-dir VisualProcessing/Classification/ImageData/images/Wound_dataset \
  --epochs 5 --val-ratio 0.2 --split-seed 42
```

Flags:
- `--test-ratio` adds a held-out test subset.
- `--split-seed` makes the split reproducible.

Image source:
https://www.kaggle.com/datasets/yasinpratomo/wound-dataset?resource=download

