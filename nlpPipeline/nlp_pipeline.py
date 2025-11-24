import os
import pandas as pd
import spacy

# OPTIONAL FUTURE NER (BioBERT)
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# MEDCAT
try:
    from medcat.cat import CAT
except:
    print("ERROR: MedCAT not installed or environment broken.")
    exit()

# 2. Initialize SpaCy and BioClinicalBERT NER
nlp = spacy.load("en_core_web_sm")  # SpaCy for linguistic features
model_name = "d4data/biomedical-ner-all"  # Pretrained biomedical NER
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")


DATA_DIR = os.path.join(os.path.dirname(__file__), "data_p3.2")
model_pack_path = os.path.join(DATA_DIR, "medmen_wstatus_2021_oct.zip")
cat = CAT.load_model_pack(model_pack_path)

# SPACY
nlp = spacy.load("en_core_web_sm")


# LOAD CSV
def load_transcript_csv(file_path="./output/transcript.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcript file missing at {file_path}")
    return pd.read_csv(file_path, names=["start", "end", "text", "speaker"],
                       header=None, skiprows=1)



# STANDARDIZE / NORMALIZE TEXT

def normalize_text(text):
    replacements = {
        "pneumo": "pneumothorax",
        "pneumo.": "pneumothorax"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text



# INTERVENTION LIST 

INTERVENTION_PRETTY = {
    "cardiopulmonary resuscitation",
    "cpr",
    "pneumothorax",
    "airway",
    "thoracostomy",
    "breathing thoracostomy",
    "cardiac arrest",
    "ventilator",
    "spinal motion restriction",
    "smr",
}



# Unused language processing kept for future fine tuning of models

def full_linguistic_processing(text):
    """
    Full SpaCy POS, dependency parsing, & lemmas.
    UNUSED right now but here for future fine-tuning.
    """
    doc = nlp(text)

    tokens = [{
        "text": t.text,
        "pos": t.pos_,
        "lemma": t.lemma_
    } for t in doc if not t.is_stop]

    dependencies = [{
        "word": t.text,
        "head": t.head.text,
        "relation": t.dep_
    } for t in doc]

    return tokens, dependencies



# MAIN PIPELINE
async def run_nlp(output_path="./output/nlp_extracted.csv"):
    df = load_transcript_csv()
    extracted_rows = []

    for _, row in df.iterrows():
        text = row["text"]
        text_norm = normalize_text(text)

        #  MedCAT: interventions + medications 
        medcat_out = cat.get_entities(text_norm, only_cui=False)

        #  BioNER: ONLY MEDICATION 
        ner_results = ner_pipeline(text)  # BioClinicalBERT NER

        #  Process MedCAT entities 
        for ent in medcat_out["entities"].values():
            pname = ent["pretty_name"].lower()
            category = None

            if pname in INTERVENTION_PRETTY:
                category = "Intervention"

            if category is not None:
                extracted_rows.append({
                    "start": row["start"],
                    "end": row["end"],
                    "text": row["text"],
                    "category": [category, "MedCat"],
                    "entity_text": ent["pretty_name"],
                    "entity_type": ", ".join(ent["types"]),
                })

        #  Process BioNER entities (only medication) 
        for e in ner_results:
            if e["entity_group"].upper() == "MEDICATION":
                extracted_rows.append({
                    "start": row["start"],
                    "end": row["end"],
                    "text": row["text"],
                    "category": "Medication",
                    "entity_text": e["word"],
                    "entity_type": "Medication (BioNER)",
                })

    #  Output CSV 
    out_df = pd.DataFrame(extracted_rows)
    out_df.to_csv(output_path, index=False)
    print(f"Done. Saved extracted results → {output_path}")

if __name__ == "__main__":
    run_nlp()
