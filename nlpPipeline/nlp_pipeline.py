import spacy
import pandas as pd
import os

def load_transcript_csv(file_path="../output/transcript.csv"):
    """Load diarized transcript CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcript file not found at {file_path}")
    return pd.read_csv(file_path, names=["start", "end", "text", "speaker"], header=None, skiprows=1)

def process_nlp():
    """NLP pipeline: tokenization -> POS -> lemma -> dependency parsing."""
    nlp = spacy.load("en_core_web_sm")  # SpaCy model
    df = load_transcript_csv()

    processed_data = []
    for _, row in df.iterrows():
        doc = nlp(row["text"])

        tokens = []
        dependencies = []
        for token in doc:
            if not token.is_stop:  # remove stop words
                tokens.append({
                    "text": token.text,
                    "pos": token.pos_,
                    "lemma": token.lemma_
                })

            # Dependency info for every token (including stop words for full tree)
            dependencies.append({
                "word": token.text,
                "head": token.head.text,
                "relation": token.dep_
            })

        processed_data.append({
            "speaker": row["speaker"],
            "tokens": tokens,
            "dependencies": dependencies
        })
    return processed_data

def run_nlp():
    print("Running full NLP pipeline...")
    data = process_nlp()
    for item in data:
        print(item)

run_nlp()