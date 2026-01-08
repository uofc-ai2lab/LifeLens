# 🎙 Audio Processing Pipeline
A complete audio-to-transcript and transcript-to-meaning pipeline with services for transcription, intervention extraction, medication extraction, and semantic filtering.

**Note: Ensure you are running all of the following commands from the project root unless otherwise specified.**

## 🔧 Environment Setup
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

If the output points to Anaconda (❌ not desired):
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

## 📦 Install Dependencies
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

## 🔑.ENV Setup

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

**⚠️ Keep env.template updated with any new variables your services require.**

## 📂 Data Directory Structure

Place your testing data in the `data/` directory using the following structure:

- `data/audio_files/` - Store your input audio files (MP3, WAV, etc.)
    - Note: `AUDIO_FILES` should point to the file(s) you want to transcribe. If left blank, the pipeline will process all audio files in `data/audio_files/`
- `data/transcript_files/` - Store your input transcript files (CSV format)  
    - Output from audio-to-transcript service and  input for transcript-to-meaning services
    - Note: `TRANSCRIPT_FILES` should point to the transcript(s) you wish to extract meaning from. If left blank, the pipeline will process all audio files in `data/transcript_files/`
- `data/meaning_files/` - Output files from transcript-to-meaning services (CSV format)
- `data/output_files/` - Combined output files from transcript-to-meaning services (CSV format) so eventually, each audio file has 1 corresponding output file instead of 1 from each service.

**Naming Convention**

All files are named in the format: `"{timestamp}_{service}_{input_filename}.csv"`
- The `timestamp` is important to have at the start for ordering. It is in the format YYYYMMDD_HHMMSS
- The `service` here is either `medX`, `intervention`,`semantic`, or `output` (the combined output file)
- The `input_filename` is whatever audio file was transcribed or transcription file was processed.

## ▶️ Running Services (Pipeline)

### Run All Services

Run all services combined from the **root folder** using the following command (no arguments):

```sh
python -m src.main
```

This will run transcription -> medication extraction -> semantic filtering

### Run Individual Services

Run services from the **root folder** using the following command pattern:

```sh
python -m src.main <service_name>
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
python -m src.main meds

# Run transcription
python -m src.main transcribe
```

### Important Notes:
- Always run from the **root** folder.
- Run as a module (`src.main`), not as a file (`src/main.py`).