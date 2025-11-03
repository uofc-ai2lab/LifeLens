# Paramedic Injury Classification (Multi-label)

This repo will focus on simple, practical multi-label classification for external injuries using still images (e.g., a key frame from a video). Optional ideas are captured here so they’re not lost, but we’ll start basic.

## Core classes (multi-label)
Use concise, normalized names:
- Abrasion
- Bruise
- Laceration
- Cut (or Puncture) — pick one; we’ll start with "Cut"
- Burn
- Normal skin

Optional classes to consider later:
- Avulsion
- Bite
- Other/unclear
- Chronic ulcers (Pressure/Diabetic/Venous) – likely not needed for field triage

## Scope and assumptions
- One image may have multiple applicable labels (multi-label).
- Start with still images (first or best frame from a video). Video handling is optional later.

## Simple data spec (CSV)
We’ll keep dataset wiring dead simple with a CSV file:

- File: `data/annotations/labels.csv`
- Columns:
  - `image_path` (relative to project root or a base images dir)
  - `labels` (semicolon-separated class names)

Example:
```
image_path,labels
images/img_0001.jpg,"Bruise;Laceration"
images/img_0002.jpg,"Burn"
images/img_0003.jpg,"Normal skin"
```

## Minimal plan
- Baseline 1 (no training): Zero-shot classifier using OpenCLIP with prompts for each class; apply per-class thresholds.
- Baseline 2 (light training): Linear probe on CLIP features (One-vs-Rest logistic regression).
- Baseline 3 (small fine-tune): EfficientNet-B0 (or ConvNeXt-Tiny) with a 6-logit sigmoid head and BCEWithLogitsLoss.

## Quick prompts (zero-shot)
- "a clinical photo of an abrasion"
- "a clinical photo of a bruise"
- "a clinical photo of a laceration"
- "a clinical photo of a cut"
- "a clinical photo of a burn"
- "a clinical photo of normal skin"

We can add variants like "wound photo showing {class}" and average text embeddings for a slight boost.

## Metrics and thresholds (basic)
- Metrics: macro/micro F1, per-class precision/recall, mAP.
- Start thresholds at 0.5; then tune per class on the validation set (maximize F1 or Youden’s J).

## Handling imbalance (basic)
- Use class weights in BCEWithLogits, or oversample minority labels.
- Keep augmentations moderate: random crop/resize, flips, small color jitter; avoid heavy transforms that distort clinical appearance.

---

## Optional ideas (for later)

### Attributes (additional labels)
Add a small attribute head or additional labels:
- Active bleeding (yes/no; consider severity flag)
- Severe burn (yes/no) or burn degree buckets
- Foreign object present (yes/no)
- Contamination/soiling (yes/no)

### Advanced models
- Transformers: ViT-B/16 or Swin-T (multi-label head)
- Foundation features: DINOv2 or OpenCLIP for stronger linear probes
- Lightweight: MobileNetV3-Large / EfficientNetV2-S for speed/edge

### Better thresholding and calibration
- Optimize per-class thresholds on val set
- Try temperature scaling or Platt scaling for probability calibration

### Video handling (later)
- Start with per-frame predictions + temporal smoothing (majority vote or EMA)
- Optionally pick the sharpest/clearest frame as the single inference frame
- Consider lightweight tracking to avoid label flicker

---

## Next steps
1) Confirm the exact class list (the six above) and finalize names.
2) Prepare a small CSV with a handful of labeled examples in `data/annotations/labels.csv`.
3) We’ll wire up two tiny scripts: zero-shot and linear-probe. If useful, we’ll add one small fine-tune script next.
