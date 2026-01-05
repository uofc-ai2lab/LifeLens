import csv
import os
import pandas as pd

def load_csv_file(file_path):
    """Load diarized transcript CSV file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Transcript file not found at {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        raise IOError(f"Error loading transcript file: {e}")

def load_csv_as_rows(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))