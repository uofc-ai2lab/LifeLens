import spacy
import pandas as pd
import os

def load_transcript_csv(file_path="./output/transcript.csv"):
    """Load diarized transcript CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcript file not found at {file_path}")
    return pd.read_csv(file_path, names=["start", "end", "text", "speaker"])

def process_nlp():
    """NLP pipeline Steps: tokenization --> POS tagging --> lemmatization."""
    nlp = spacy.load("en_core_web_sm")
    df = load_transcript_csv()

    processed_data = []
    for ind, row in df.iterrows():
        doc = nlp(row["text"])
        tokens = []
        for token in doc:
            tokens.append({
                "text": token.text,
                "pos": token.pos_,
                "lemma": token.lemma_
            })
        processed_data.append({
            "speaker": row["speaker"],
            "tokens": tokens
        })
    return processed_data

async def run_nlp():
    print("Running full NLP pipeline...")
    data = process_nlp()
    print(data)  
