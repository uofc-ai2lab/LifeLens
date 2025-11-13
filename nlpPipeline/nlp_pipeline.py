import spacy
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# 1. Load transcript CSV
def load_transcript_csv(file_path="../output/transcript.csv"):
    """Load diarized transcript CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcript file not found at {file_path}")
    return pd.read_csv(file_path, names=["start", "end", "text", "speaker"], header=None, skiprows=1)

# 2. Initialize SpaCy and BioClinicalBERT NER
nlp = spacy.load("en_core_web_sm")  # SpaCy for linguistic features
model_name = "d4data/biomedical-ner-all"  # Pretrained biomedical NER
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")

def process_nlp():
    """NLP pipeline: SpaCy + BioClinicalBERT NER."""
    df = load_transcript_csv()
    processed_data = []

    for _, row in df.iterrows():
        doc = nlp(row["text"])

        # SpaCy features
        tokens = []
        dependencies = []
        for token in doc:
            if not token.is_stop:
                tokens.append({
                    "text": token.text,
                    "pos": token.pos_,
                    "lemma": token.lemma_
                })
            dependencies.append({
                "word": token.text,
                "head": token.head.text,
                "relation": token.dep_
            })

        # BioClinicalBERT NER
        ner_results = ner_pipeline(row["text"])
        entities = [{"entity": e["entity_group"], "text": e["word"], "score": round(e["score"], 3)} for e in ner_results]

        processed_data.append({
            "speaker": row["speaker"],
            "tokens": tokens,
            "dependencies": dependencies,
            "entities": entities  # Added NER results
        })
    return processed_data

def run_nlp():
    print("Running full NLP pipeline...")
    data = process_nlp()
    for item in data:
        print(item)

run_nlp()