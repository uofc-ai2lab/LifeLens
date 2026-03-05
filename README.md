# LifeLens Codebase
A complete audio-to-transcript and transcript-to-meaning pipeline with services for transcription, intervention extraction, medication extraction, and semantic filtering.

Line for restarting camera: sudo systemctl restart nvargus-daemon

For technical details, see [TECHNICAL.md](TECHNICAL.md).

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

First, ensure you have Python 3.11 installed. If not, download and install it from [python.org](https://www.python.org/downloads/). During installation, make sure to check "Add Python to PATH".

Check which Python executable is currently active:

In Command Prompt or PowerShell:
```cmd
where python
```
or, check Python3:
```cmd
where python3
```

In bash (Git Bash or WSL):
```bash
which python
```
or:
```bash
which python3
```

Check the Python version:
```cmd
python --version
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
- A system Python location (e.g. `C:\Python311\python.exe`)

You are now ready to create and activate a virtual environment.

### Step 2: Setup Virtual Environment
Once you have the expected output, create a **dedicated virtual environment**:
Create venv

For Jetson:
```bash
python -m venv --system-site-packages .venv
```

For Laptop:
```bash
python3.11 -m venv .venv
```

Activate on Windows:

In Command Prompt:
```cmd
.venv\Scripts\activate
```

In PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

In bash (Git Bash or WSL):
```bash
source .venv/Scripts/activate
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
sudo pip install ultralytics --no-deps 
sudo pip install deface --no-deps
sudo apt update
sudo apt install -y python3-gi gir1.2-gstreamer-1.0 gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good
sudo apt install -y evtest

# Need to manually install ultralytics due to dependency issues 
```

###  Download MedCAT Model (One-Time Setup) - for Medication Intervention
To keep setup simple and cross-platform, we provide a Python script that:
- Creates the required data directory
- Downloads the MedCAT model
- Downloads example clinical notes

From the project root, run the setup script:
```bash
python -m scripts.setup_medcat
```

### Download Dataset for Training Classifier
Download the `wounds_dataset` (Classification) from Kaggle:

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
Note that ImageSamples is a subset of this dataset


### Download Dataset for Object Detection
Image set link: https://github.com/xiaojie1017/Human-Parts 

Object detection link: https://huggingface.co/MnLgt/yolo-human-parse/tree/main 

Dataset Assets (Priv_personpart):

## .ENV Setup

1. **Hugging Face Token**
   - Register at [Hugging Face](https://huggingface.co)
   - Get your access token with READ permissions from [Settings > Access Tokens](https://huggingface.co/settings/tokens)

2. **Create .ENV file**
   - Copy `env.template` from `/config` to `.env` in the root folder
   - Add your API key:
     ```dotenv
     HUGGINGFACE_TOKEN=your_hf_token_here
     ```
   - Add all additional variables 

3. **Video .ENV Variables**
   Video pipeline configuration lives in the `VIDEO PIPELINE ENVIRONMENT VARIABLES` section. Set at minimum:

   - `PIPELINE_DETECTION_SOURCE` (defaults to `data/video/source_files`)
   - `PIPELINE_ROOT` (defaults to `data/video/output_files`)
   - `PIPELINE_DETECTION_OUTPUT` (defaults to `data/video/output_files/DetectionOutput`)
   - `PIPELINE_INJURY_CHECKPOINT` (checkpoint used for injury inference)

   For Jetson memory stability, you can also tune camera load:

   - `VIDEO_CAPTURE_WIDTH` / `VIDEO_CAPTURE_HEIGHT` (defaults: `1280x720`)
   - `VIDEO_DISPLAY_WIDTH` / `VIDEO_DISPLAY_HEIGHT` (defaults: `640x360`)
   - `VIDEO_FRAME_RATE` (default: `20`)
   - `VIDEO_FLIP_METHOD` (default: `0`)

**Keep env.template updated with any new variables your services require.**

# Audio Processing Pipeline

## Audio Data Directory Structure
Place all testing and input data under the `data/audio/` directory using the structure below:

```
data/audio/
├── audio_files/
│   └── <parent_audio>.<wav|mp3|m4a>
├── audio_chunks/
│   └── recording_<timestamp>_chunk_<n>.wav
└── processed_audio/
   └── recording_<timestamp>_chunk_<n>/
      ├── recording_<timestamp>_chunk_<n>.wav
      ├── <service>_*.csv
      └── <service>_*.json
```

#### `audio_files/`

* Stores **raw input audio files** (e.g., WAV, MP3).
* These are the source files that will be chunked and processed.
* If `AUDIO_FILES` is specified in configuration, only those files will be processed.
* If `AUDIO_FILES` is left empty, **all files in this directory will be processed**.

#### `audio_chunks/`

* Temporary directory for recorded chunks (created by the audio pipeline).

#### `processed_audio/`

* Automatically generated by the pipeline.
* Each parent audio file gets its own folder.
* Each chunk of the audio is processed independently and stored in its own subdirectory.
* All downstream outputs (transcripts, semantic analysis, medication extraction, interventions, anonymization, and final merged CSVs) are saved **alongside the chunk they belong to**.

**Naming Convention**

All files are named in the format: `"{service}_{input_filename}.csv"`
- `service`: whatever processing was done such as `medX`, `transcription`, etc.
- `input_filename`: outputs are always colocated with their corresponding audio chunk and transcript.

## Running Audio Pipeline

### Dev Mode (No Mic Required)

Process whatever is already in `data/audio/audio_chunks/`:

```sh
python -m src_audio.main_audio --dev
```

### Audio Pipeline (Mic Required)

Run the audio pipeline (mic + processing) from the **root folder** using the following command (no arguments):

```sh
python -m src_audio.main_audio
```

This runs: record -> transcription -> anonymization -> medication extraction -> intervention extraction.

### Important Notes:
- Always run from the **root** folder.
- Run as a module (`src_audio.main_audio`), not as a file.
- Individual service CLIs are not currently wired; call the service modules directly if needed.


# Video Processing Pipeline
A detection + crop extraction pipeline with an injury-classification inference step.

## Video Data Directory Structure

Place your video/image test data under:

- `data/video/source_files/` - Input images to process
- `data/video/saved_imgs/` - Snapshots captured from the camera
- `data/video/output_files/` - All video pipeline outputs
   - `DetectionOutput/`
      - `annotated/` - Annotated detection visualizations
      - `crops/` - Body-part crops (organized per input image)
      - `vis/` - Simple visualization images
   - `ClassificationOutput/`
      - `injury_predictions.json` - Per-crop predictions + summary
      - `injury_predictions_summary.csv` - Per-image/per-body-part summary

## Running the Video Pipeline

### Dev Mode (Image Processing Only)

Process whatever is already in `data/video/saved_imgs/`:

```sh
python -m src_video.main_video --dev
```

This will run:
1. Detection + crop extraction
2. Injury inference on crops (using the configured checkpoint)


### Video Pipeline (Camera Required)

Run the video pipeline with live camera capture (no audio):

```sh
python -m src_video.main_video
```

This captures frames from the CSI camera and processes them through the full video pipeline.

### Important Notes:
- If you see no outputs in dev mode, ensure `data/video/saved_ims/` is not empty.
- The live camera pipeline can be run standalone with `python -m src_video.main_video` or together with audio via [main.py](main.py).


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

# Full System 

## Through Terminal (Camera + Audio)

Run both pipelines together (camera + mic):

```sh
python -m main
```

This runs:
- Video capture from CSI camera → detection → injury classification
- Audio recording from USB mic → transcription → medication/intervention extraction

## Using Button (Camera + Audio)

Run both pipelines together (camera + mic):

```sh
python -m power_toggle
```

This does the following:
1. Waits for a button press. On the first press, it starts the full system (`python -m main` runs in the background) and turns the LED on.
2. Saves all output logs under the data directory.
3. On the second button press, it cleanly shuts down the camera and microphone (audio capture stops and no new files are written); the LED briefly blinks to indicate shutdown.
4. After the program fully exits, the LED turns off.
5. While the system is shutting down, additional button presses are ignored until the shutdown is complete.


### Important Notes:
- If the camera fails to initialize, the audio pipeline will not start.
- De-identification is currently enabled but is a placeholder implementation.

