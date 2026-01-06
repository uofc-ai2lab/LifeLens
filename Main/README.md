# Main Pipeline (Detection  Classification)

This folder contains the unified *image* pipeline entrypoint used by the project.

At a high level:

1. **Body-part detection** (YOLO segmentation) runs on a source image folder.
2. **Crops** are written to a structured output directory.
3. **Injury classification inference** runs a trained checkpoint on the produced crops.

The pipeline code lives in:

- `Main/main_pipeline.py` (orchestration)
- `Main/detect_body_parts.py` (detection + crop export)
- `Main/infer_injuries_on_crops.py` (injury classifier inference over crops)
- `Main/pipeline_config.py` (env-driven configuration)

## Quickstart

1. Create / activate your Python environment and install deps (see repo-level README).
2. Configure environment variables by copying or editing the repo root `.env.template`.
3. Run:

```bash
python Main/main_pipeline.py
```

If you run via VS Code, ensure your Python extension is pointing at `.env.template` (or a local `.env`) so the `PIPELINE_*` values are available.

## Configuration

The pipeline reads config from environment variables (see `Main/pipeline_config.py`).
Defaults are defined in code, and the repo root `.env.template` documents the intended values.

### Most important variables

- `PIPELINE_DETECTION_SOURCE`
  - Default: `VisualProcessing/ObjectDetection/Priv_personpart/ImageSamples`
  - Folder of input images (jpg/png/etc)
- `PIPELINE_DETECTION_MODEL`
  - Default: `MnLgt/yolo-human-parse`
  - Can be a local `.pt` file or a Hugging Face repo id
- `PIPELINE_ROOT`
  - Default: `Main/PipelineOutputs`
  - Parent output directory
- `PIPELINE_DETECTION_OUTPUT`
  - Default: `Main/PipelineOutputs/DetectionOutput`
- `PIPELINE_CLASSIFICATION_OUTPUT`
  - Default: `Main/PipelineOutputs/ClassificationOutput`

### Detection tuning

- `PIPELINE_MAX_IMAGES` (limits how many images are processed)
- `PIPELINE_CLASSES` (comma-separated list of parts)
- `PIPELINE_MIN_AREA` (filters tiny masks)
- `PIPELINE_MARGIN` (bbox expansion around each mask)
- `PIPELINE_ADD_HEAD` (creates a composite `head` crop from face/hair/neck)
- `PIPELINE_ALPHA_PNG` (writes extra `_alpha.png` crops with transparent background)
- `PIPELINE_DEVICE` (optional, e.g. `cuda:0` or `cpu`)
- `PIPELINE_DEBUG` (extra prints)

### Injury classifier inference

- `PIPELINE_INJURY_CHECKPOINT`
  - Default: `experiments/checkpoints/simple/best_swin_tiny_patch4_window7_224.pt`
- `PIPELINE_INJURY_REPORT_JSON`
- `PIPELINE_INJURY_REPORT_CSV`

Notes:

- The `.env.template` currently contains additional `PIPELINE_CLS_*` variables for training; the default pipeline entrypoint (`Main/main_pipeline.py`) **does not train** a classifier today, it only runs injury inference.
- The detector supports an optional ImageFolder-style export (`PIPELINE_CLASSIFICATION_EXPORT`) when `PIPELINE_USE_DETECTION_CROPS_FOR_TRAINING=false`.

## Output layout

By default outputs land under `Main/PipelineOutputs/`.

### Detection output (`DetectionOutput/`)

- `crops/<image_id>/*.jpg`
  - Crops are named like `<origstem>_<part>_<idx>.jpg`
- `annotated/*.jpg`
  - Model visualizations of detections
- `vis/*.jpg`
  - Simple bbox visualizations

### Classification output (`ClassificationOutput/`)

- `injury_predictions.json`
  - Per-crop predictions + per-part aggregation
- `injury_predictions_summary.csv`
  - Aggregated summary per `(image_id, body_part)`

These output folders are intended to be generated artifacts and are ignored by git.

## Troubleshooting

- **No crops were created**: confirm `PIPELINE_DETECTION_SOURCE` points to a folder with images and that the model is loading successfully.
- **Slow first run**: if using a Hugging Face repo id for the detection model, weights may download on first run.
- **CUDA issues**: set `PIPELINE_DEVICE=cpu` to force CPU inference.

## Related folders

- `VisualProcessing/` contains older standalone scripts and datasets used during development.
  The unified pipeline in `Main/` is the recommended entrypoint for running detection + injury inference.