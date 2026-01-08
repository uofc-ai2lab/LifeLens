# LifeLens

LifeLens is a capstone project that combines:

- **Audio transcription + diarization** (WhisperX / faster-whisper / WhisperTRT)
- **NLP extraction** of interventions and medications from transcripts
- **Image pipeline** for body-part detection (YOLO segmentation) and injury classification on the resulting crops

## Table of contents

- [Quickstart](#quickstart)
- [Configuration (.env)](#configuration-env)
- [Run services (CLI)](#run-services-cli)
- [Main image pipeline (detection → classification)](#main-image-pipeline-detection--classification)
- [NLP pipeline](#nlp-pipeline)
- [Injury classification notes (multi-label)](#injury-classification-notes-multi-label)

## Quickstart

### 1) Create and activate a Python virtual environment

```sh
python3.11 -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Mac/Linux
source .venv/bin/activate
```

### 2) Install dependencies

This repo contains multiple components, some with their own dependency sets.

- macOS dev env: `pip install -r requirements-mac.txt`
- Additional component-specific requirements exist under:
  - `WhisperTRT/requirements.txt`
  - `encryption/requirements.txt`
  - `VisualProcessing/**/requirements.txt`

### 3) Configuration (.env)

Copy `.env.template` to `.env` and fill in required values.

Key variables:

- `HUGGING_FACE_TOKEN` (needed if pulling models from Hugging Face)
- `DEVICE` (`cpu` or `cuda`)
- `AUDIO_FILE_PATH` (path to the audio to transcribe)
- `OUTPUT_DIR` (default `./output`)

The file also contains `PIPELINE_*` variables used by the image pipeline.

## Run services (CLI)

`main.py` is the command-line entrypoint for the project’s microservices:

```sh
python main.py whisperx
python main.py faster_whisper
python main.py whispertrt
python main.py nlp
```

Outputs are generally written under `./output/` and/or `Main/PipelineOutputs/` depending on the service.

## Main image pipeline (detection → classification)

The unified image pipeline lives in the `Main/` folder:

1. **Body-part detection** (YOLO segmentation) runs on a source image folder.
2. **Crops** are written to a structured output directory.
3. **Injury classification inference** runs a trained checkpoint on the produced crops.

Key files:

- `Main/main_pipeline.py` (orchestration)
- `Main/detect_body_parts.py` (detection + crop export)
- `Main/infer_injuries_on_crops.py` (injury classifier inference over crops)
- `Main/pipeline_config.py` (env-driven configuration)

### Run

```sh
python Main/main_pipeline.py
```

### Most important `PIPELINE_*` variables

- `PIPELINE_DETECTION_SOURCE` (input image folder)
- `PIPELINE_DETECTION_MODEL` (local `.pt` or Hugging Face repo id)
- `PIPELINE_ROOT` / `PIPELINE_DETECTION_OUTPUT` / `PIPELINE_CLASSIFICATION_OUTPUT`
- `PIPELINE_INJURY_CHECKPOINT` (injury classifier checkpoint)
- `PIPELINE_DEVICE` (optional override, e.g. `cuda:0` or `cpu`)

Defaults live in `Main/pipeline_config.py`, and intended values are documented in `.env.template`.

### Output layout

By default outputs land under `Main/PipelineOutputs/`:

- `DetectionOutput/`
  - `crops/<image_id>/*.jpg`
  - `annotated/*.jpg`
  - `vis/*.jpg`
- `ClassificationOutput/`
  - `injury_predictions.json`
  - `injury_predictions_summary.csv`

## NLP pipeline

The NLP pipeline reads a Whisper transcript CSV and writes extracted entities:

1. Load transcript from `./output/transcript.csv`
2. Extract **interventions** (MedCAT)
3. Extract **medications** (BioNER)
4. Save results to `./output/nlp_extracted.csv`

### Setup (NLP-only)

Some NLP dependencies may be installed separately until the repo uses a single pinned requirements set:

```sh
pip install medcat~=1.16.0
python -m spacy download en_core_web_sm
```

### Download MedCAT model + example data (optional)

```sh
cd nlpPipeline
mkdir -p ../data_p3.2

# Download MedCAT model and example data
wget -N https://cogstack-medcat-example-models.s3.eu-west-2.amazonaws.com/medcat-example-models/medmen_wstatus_2021_oct.zip -P ../data_p3.2
wget -N https://raw.githubusercontent.com/CogStack/MedCATtutorials/main/notebooks/introductory/data/pt_notes.csv -P ../data_p3.2
```

Notes:

- Keep `data_p3.2/` local only (do not commit).

### Run

```sh
python main.py nlp
```

## Injury classification notes (multi-label)

### Core classes (multi-label)

- Abrasion
- Bruise
- Laceration
- Cut
- Burn
- Normal skin

### Simple data spec (CSV)

- File: `data/annotations/labels.csv`
- Columns:
  - `image_path` (relative to project root)
  - `labels` (semicolon-separated class names)

Example:

```csv
image_path,labels
images/img_0001.jpg,"Bruise;Laceration"
images/img_0002.jpg,"Burn"
images/img_0003.jpg,"Normal skin"
```

### Baselines

- Zero-shot with CLIP prompts + per-class thresholds
- Linear probe on CLIP features (one-vs-rest logistic regression)
- Small fine-tune (e.g., EfficientNet / Swin) with sigmoid + BCEWithLogitsLoss
