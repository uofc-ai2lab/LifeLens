# LifeLens Codebase
A complete audio-to-transcript and transcript-to-meaning pipeline with services for transcription, intervention extraction, medication extraction, and semantic filtering.

**Note: Ensure you are running all of the following commands from the project root unless otherwise specified.**

## Environment Setup
### Step 1: Verify Python Environment
First, check which Python executable is currently active

**For Mac:**

```bash
which python
``` 
or, check Python3: 
```bash
which python3
``` 

**For Windows:**

```bash
where python
``` 
or, check Python3: 
```bash
where python3
``` 

If the output points to Anaconda (not desired):
```
.../anaconda3/bin/python
```

Deactivate Conda:
```
conda deactivate
```

Then verify again (commands above)

At this point, one of the following is expected and OK:
- No output (Python not currently on PATH)
- A system Python location (e.g. `/usr/bin/python3`)

You are now ready to create and activate a virtual environment.

### Step 2: Setup Virtual Environment
Once you have the expected output, create a **dedicated virtual environment**:

Create venv:
```bash
python3.11 -m venv .venv
```

Activate on Windows:
```bash
.venv\Scripts\activate
```

Activate on Mac/Linux:
```bash
source .venv/bin/activate
```

### Step 3: Upgrade pip
```sh
python -m pip install --upgrade pip
```

## Install Dependencies
***Note: run all of the following commands INSIDE your virtual environment***

### Install Required Dependencies
For all operating systems, run the following:
```sh
pip install -r requirements.txt
```

If operating on a jetson-nano, also install jetson-specific dependencies:
```sh
pip install -r requirements-jetson.txt
```

###  Download MedCAT Model (One-Time Setup)
To keep setup simple and cross-platform, we provide a Python script that:
- Creates the required data directory
- Downloads the MedCAT model
- Downloads example clinical notes

From the project root, run the setup script:
```bash
python -m scripts.setup_medcat
```

## Depenedencies for video pipeline:
Dataset for training classifier:
Download the `wounds_dataset` (Classification)

Download the dataset from Kaggle:
https://www.kaggle.com/datasets/yasinpratomo/wound-dataset?resource=download

Extract the downloaded folder so the images end up at:

```
data/video/source_files/Images/Wound_dataset/
  Abrasion/
  Bruise/
  Burn/
  Cut/
  Laceration/
  Stab_wound/
  Normal skin/
```
Note that imageSamples is a subset of this dataset


Dataset for object detection:
Image set link: https://github.com/xiaojie1017/Human-Parts Object detection link: https://huggingface.co/MnLgt/yolo-human-parse/tree/main Dataset Assets (Priv_personpart):

## .ENV Setup

1. **Hugging Face Token**
   - Register at [Hugging Face](https://huggingface.co)
   - Get your access token with READ permissions from [Settings > Access Tokens](https://huggingface.co/settings/tokens)

2. **OpenAI API Key**
   - Sign up at [OpenAI Platform](https://platform.openai.com)
   - Navigate to [API Keys](https://platform.openai.com/api-keys)
   - Create a new secret key

3. **Google API Key**
   - Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create a new API key
   - (Alternative: Use [Google Cloud Console](https://console.cloud.google.com/apis/credentials) for Gemini API access)

4. **Create .env file**
   - Copy `env.template` from `/config` to `.env` in the root folder
   - Add your API keys:
     ```dotenv
     HUGGINGFACE_TOKEN=your_hf_token_here
     OPENAI_API_KEY=your_openai_key_here
     GOOGLE_API_KEY=your_google_key_here
     ```

**Keep env.template updated with any new variables your services require.**

# Audio Processing Pipeline

## Audio Data Directory Structure

Place your testing data in the `data/audio/` directory using the following structure:

- `data/audio/audio_files/` - Store your input audio files (MP3, WAV, etc.)
    - Note: `AUDIO_FILES` should point to the file(s) you want to transcribe. If left blank, the pipeline will process all audio files in `data/audio/audio_files/`
- `data/audio/transcript_files/` - Store your input transcript files (CSV format)  
    - Output from audio-to-transcript service and  input for transcript-to-meaning services
    - Note: `TRANSCRIPT_FILES` should point to the transcript(s) you wish to extract meaning from. If left blank, the pipeline will process all audio files in `data/audio/transcript_files/`
- `data/audio/meaning_files/` - Output files from transcript-to-meaning services (CSV format)
- `data/audio/output_files/` - Combined output files from transcript-to-meaning services (CSV format) so eventually, each audio file has 1 corresponding output file instead of 1 from each service.

**Naming Convention**

All files are named in the format: `"{timestamp}_{service}_{input_filename}.csv"`
- The `timestamp` is important to have at the start for ordering. It is in the format YYYYMMDD_HHMMSS
- The `service` here is either `medX`, `intervention`,`semantic`, or `output` (the combined output file)
- The `input_filename` is whatever audio file was transcribed or transcription file was processed.

## Running AUDIO Services (Pipeline)

### Run All Audio Services

Run all audio services from the **root folder** using the following command (no arguments):

```sh
python -m src_audio.main
```

This will run transcription -> medication extraction -> intervention extraction
(semantic filtering is currently excluded from the full pipeline run until med/inter extraction is complete, it can still be ran independently.)

### Run Individual Audio Services

Run services from the **root folder** using the following command pattern:

```sh
python -m src_audio.main <service_name>
```

### Available Services

| Service Name     | Description                       |
| ---------------- | --------------------------------- |
| `transcribe`     | Run audio-to-transcript service   |
| `meds`           | Run medication extraction service |
| `inter`          | Run NLP intervention extraction   |
| `sem`            | Run semantic filtering service |


### Examples

```sh
# Run medication extraction
python -m src_audio.main meds

# Run transcription
python -m src_audio.main transcribe
```

### Important Notes:
- Always run from the **root** folder.
- Run as a module (`src_audio.main`), not as a file (`src_audio/main.py`).


# Video Processing Pipeline
A detection + crop extraction pipeline with an injury-classification inference step.

## Video Data Directory Structure

Place your video/image test data under:

- `data/video/source_files/` - Input images to process
- `data/video/output_files/` - All video pipeline outputs
   - `DetectionOutput/`
      - `annotated/` - Annotated detection visualizations
      - `crops/` - Body-part crops (organized per input image)
      - `vis/` - Simple visualization images
   - `ClassificationOutput/`
      - `injury_predictions.json` - Per-crop predictions + summary
      - `injury_predictions_summary.csv` - Per-image/per-body-part summary

## Video .ENV Variables

Video pipeline configuration lives in the **VIDEO PIPELINE ENVIRONMENT VARIABLES** section of `config/.env.template`.
Create your local `.env` in the repo root (as described above) and set at minimum:

- `PIPELINE_DETECTION_SOURCE` (defaults to `data/video/source_files`)
- `PIPELINE_ROOT` (defaults to `data/video/output_files`)
- `PIPELINE_DETECTION_OUTPUT` (defaults to `data/video/output_files/DetectionOutput`)
- `PIPELINE_INJURY_CHECKPOINT` (checkpoint used for injury inference)

## Running the Video Pipeline

Run the full video pipeline (no arguments) from the **root folder**:

```sh
python src_video/main.py
```

This will run:
1. Detection + crop extraction
2. Injury inference on crops (using the configured checkpoint)

### Important Notes:
- If you see no outputs, ensure `data/video/source_files/` is not empty (or set `PIPELINE_DETECTION_SOURCE` to a folder that contains images).
- De-identification is currently a placeholder step and is disabled by default.

## Model Checkpoints

The video pipeline uses a trained injury classifier checkpoint configured via `PIPELINE_INJURY_CHECKPOINT`.

**Important:** this checkpoint is required for injury inference. If the file is missing, the pipeline will still run detection/crop extraction, but the injury inference step will fail.

A pre-trained checkpoint is to heave to push so you will have to train it to get the checkpoint:

To train the injury classifier checkpoint locally, you must first download the wound classification dataset (see **Sources** below:)

Then run:

```sh
python scripts/train_video_injury_classifier.py
```

By default it writes to `checkpoints/classificationModel/injury/` and produces:
- `best_swin_tiny_patch4_window7_224.pt` (the checkpoint the pipeline loads)
- `metrics_swin_tiny_patch4_window7_224.json` (training metrics)

The default training script assumes the dataset is located at:
`data/video/source_files/Images/Wound_dataset/`

If you save to a different location (or store the dataset elsewhere), set `PIPELINE_INJURY_CHECKPOINT` in your `.env` to point at the resulting `.pt` file and/or pass `--data-dir` to the training script.

### Training with No-Injury Class (run before creating checkpoint)

To reduce false positives and improve real-world performance, you can augment the training dataset with a "no_injury" class using actual detection crops from your pipeline:

**Step 1: Create the no_injury dataset**

```sh
python scripts/create_no_injury_dataset.py
```

This will:
- Sample 300 detection crops (stratified across source images)
- Split into 80% training / 20% held-out test (by default)
- Copy training samples to `data/video/source_files/Images/Wound_dataset/no_injury/`

You can customize with:
```sh
python scripts/create_no_injury_dataset.py --num-samples 500 --train-ratio 0.8
```

**Step 2: Retrain the classifier**

```sh
python scripts/train_video_injury_classifier.py
```

The classifier will now have 8 classes (7 injury types + no_injury) and should significantly reduce false positives on clean body parts.

**Note:** The 80/20 split ensures that the held-out 20% is completely separate from training, preventing data leakage when evaluating pipeline performance.

