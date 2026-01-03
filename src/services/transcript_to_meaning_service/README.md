Here’s a **concise README snippet** for your project:

---

# LifeLens NLP Pipeline
* The pipeline will:

  1. Load the output of the Whisper pipeline within ../output/transcript.csv.
  2. Extract **interventions** via MedCAT.
  3. Extract **medications** BioNER.
  4. Save results to `./output/nlp_extracted.csv`.

## Setup

1. **Create and activate a Python virtual environment**:
* Until we have a unified venv for the project with percise versions we will need to create separate ones.

```bash
python3.11 -m venv venv_nlp
source venv_nlp/bin/activate
```

2. **Install requirements**:
* Once the correct versions of python and whisper are determined for the Jetson, the following will be added to the requirements.txt
```bash
pip install medcat~=1.16.0
python -m spacy download en_core_web_sm
```

3. **Download required models and data**:

* How to install wget for bellow MedCAT model: 
  - Windows  :  winget install GnuWin32.Wget or use Git Bash (comes with wget)
  - Mac  :  brew install wget


* cd nlpPipeline
```bash
cd ..
mkdir -p data_p3.2
DATA_DIR="./data_p3.2/"

# Download MedCAT model and example data
wget -N https://cogstack-medcat-example-models.s3.eu-west-2.amazonaws.com/medcat-example-models/medmen_wstatus_2021_oct.zip -P $DATA_DIR
wget -N https://raw.githubusercontent.com/CogStack/MedCATtutorials/main/notebooks/introductory/data/pt_notes.csv -P $DATA_DIR
```

---

## Run the NLP Pipeline

Run the main script with the `nlp` argument:

```bash
cd LifeLens
python main.py nlp
```

---

## Notes

* Keep the `data_p3.2/` folder **local only** — do **not push it to Git**.
* All large model files are downloaded automatically if missing.

