import spacy
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import json
import torch


# Load transcript CSV
def load_transcript_csv(file_path="./output/transcript.csv"):
    """Load diarized transcript CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcript file not found at {file_path}")
    return pd.read_csv(file_path, names=["start", "end", "text", "speaker"], header=None, skiprows=1)

# Initialize SpaCy
nlp = spacy.load("en_core_web_sm")

# Biomedical NER
model_name = "d4data/biomedical-ner-all"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")

# Intervention Classification (BioBERT) 
cls_model_name = "dmis-lab/biobert-base-cased-v1.1"
cls_tokenizer = AutoTokenizer.from_pretrained(cls_model_name)
cls_model = AutoModelForSequenceClassification.from_pretrained(cls_model_name, num_labels=3)  # CPR, Medication, Airway

def classify_intervention(text):
    """Predict intervention type using BioBERT."""
    inputs = cls_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = cls_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label_id = torch.argmax(probs, dim=-1).item()
    return label_id, probs[0][label_id].item()  # label index + confidence


def process_nlp():
    df = load_transcript_csv()
    processed_data = []

    for _, row in df.iterrows():
        doc = nlp(row["text"])

        # SpaCy Linguistic Features
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

        # Biomedical NER
        ner_results = ner_pipeline(row["text"])
        entities = [
            {
                "entity": e["entity_group"],
                "text": e["word"],
                "score": round(e["score"], 3)
            }
            for e in ner_results
        ]

        # Intervention Classification
        label_idx, confidence = classify_intervention(row["text"])
        label_map = {0: "CPR", 1: "Medication", 2: "Airway"}  # adjust based on your fine-tuned labels
        intervention = {"label": label_map[label_idx], "confidence": round(confidence, 3)}

        # output
        processed_data.append({
            "start": row["start"],
            "end": row["end"],
            "speaker": row["speaker"],
            "text": row["text"],
            "tokens": tokens,
            "dependencies": dependencies,
            "entities": entities,
            "intervention": intervention

        })

    return processed_data


def save_nlp_output_csv(processed_data, file_path="./output/nlpoutput.csv"):
    """Save NLP processed data to CSV with JSON strings for nested fields."""
    for item in processed_data:
        # Convert nested fields to JSON strings
        item["intervention"] = json.dumps(item[])
        item["tokens"] = json.dumps(item["tokens"])
        item["dependencies"] = json.dumps(item["dependencies"])

        # Convert float32 to float in entities
        for e in item["entities"]:
            e["score"] = float(e["score"])
        item["entities"] = json.dumps(item["entities"])

    df = pd.DataFrame(processed_data)
    df.to_csv(file_path, index=False)
    print(f"NLP output saved to {file_path}")



async def run_nlp():
    print("Running full NLP pipeline...")
    data = process_nlp()
    save_nlp_output_csv(data)

    for item in data:
        print(item)

