# capstone-audio-to-text
## Create and Activate Python Virtual Environment

```sh
python3.11 -m venv venv311
# On Windows:
venv311\Scripts\activate
# On Mac/Linux:
source venv311/bin/activate
```

## Upgrade pip

```sh
python.exe -m pip install --upgrade pip
```

## Install Dependencies

```sh
pip install -r requirements.txt
```

## .ENV Setup

1. Register at [Hugging Face](https://huggingface.co).
2. Get your access token with READ permissions.
3. Create `.env` file in this folder holding the hugging face token and the rest of the variables found in the `sample.env` **Importantly set the AUDIO_FILE_PATH variable to the file you wish to transcribe.**:

```
HUGGING_FACE_TOKEN=
CHUNK_LENGTH=10 # in minutes
CHUNK_OVERLAP=0.5 # in minutes
DEVICE=cpu # or cuda for NVIDIA GPU
AUDIO_FILE_PATH=
OUTPUT_DIR=./output

# using offline models for pyannote
PYANNOTE_CACHE_DIR=./pyannote_models
# 0 is false (use online model), 1 is true (use offline model)
USE_OFFLINE_MODELS = 0
```

**Ensure you are keeping the sample.env updated for any variables your services require.**