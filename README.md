# 🎙 Audio Processing Pipeline
A complete audio-to-transcript and transcript-to-meaning pipeline with services for transcription, intervention extraction, and medication extraction.

## 🔧 Environment Setup
### Step 1: Verify Python Location

First, verify the Python location by running the following command:
```bash
which python
```

This should return:
```
.../current_working_directory/bin/python
```

If the output points to Anaconda, e.g.:
```bash.../anaconda3/bin/python```

then run:
```bash
conda deactivate
```

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
python.exe -m pip install --upgrade pip
```

## 📦 Install Dependencies

For all operating systems, run the following:
```sh
pip install -r requirements.txt
```

If operating on a jetson-nano, also install jetson-specific dependencies:
```sh
pip install -r requirements-jetson.txt
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
| `intervention`   | Run NLP intervention extraction   |
| `meds`           | Run medication extraction service |
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