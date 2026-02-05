import os, sys
from pathlib import Path
import pandas as pd
import spacy
import re
from src_audio.utils.export_to_csv import export_to_csv
from src_audio.utils.load_csv_file import load_csv_file 
from config.audio_settings import MODEL_PACK
from src_audio.domain.constants import INTERVENTIONS, REPLACEMENTS, INTER_COLUMNS

def normalize_text(text):
    """Normalize text for better matching"""
    text = text.lower()
    for k, v in REPLACEMENTS.items():
        text = text.replace(k, v)
    return text


def match_intervention(text_norm, entity_text):
    """Match text to intervention category using word boundaries"""
    entity_lower = entity_text.lower()
    
    for intervention_type, keywords in INTERVENTIONS.items():
        for keyword in keywords:
            # word boundaries for short keywords
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, entity_lower) or re.search(pattern, text_norm):
                return intervention_type
    return None


def has_intervention_keyword(text_norm):
    """
    Check if text contains ANY intervention keyword using word boundaries.
    Returns: True if any keyword found, False otherwise
    """
    for intervention_type, keywords in INTERVENTIONS.items():
        for keyword in keywords:
            #  word boundaries to match whole words only
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_norm):
                return True
    return False


def run_intervention_extraction(chunk_path: str, transcript_path: str):
    """Main extraction pipeline for interventions only"""
    df = load_csv_file(transcript_path)
    if MODEL_PACK is None:
        print("ERROR2: MedCAT not installed or environment broken.")
        sys.exit(1)
    
    #  dict to group by start_time only (one row per start time)
    interventions_dict = {}
    
    for idx, row in df.iterrows():
        text_norm = normalize_text(row["text"])
        
        if not has_intervention_keyword(text_norm):
            continue 
        
        key = row["start_time"]  # One row per start time
        
        # Initialize entry
        if key not in interventions_dict:
            interventions_dict[key] = {
                "start_time": row["start_time"],
                "end_time": row["end_time"] if pd.notna(row["end_time"]) else "N/A",
                "event_type": "intervention",
                "event_categories": set(),
                "entities_detected": [],
                "full_text": row["text"],
            }
        
        # MedCAT for entity extraction
        medcat_out = MODEL_PACK.get_entities(text_norm, only_cui=False)
        
        # Process MedCAT entities
        for ent in medcat_out["entities"].values():
            entity_text = ent["pretty_name"]
            intervention_type = match_intervention(text_norm, entity_text)
            
            if intervention_type:
                interventions_dict[key]["event_categories"].add(intervention_type)
                if entity_text not in interventions_dict[key]["entities_detected"]:
                    interventions_dict[key]["entities_detected"].append(entity_text)
        
        # Keyword-search fallback with word boundaries
        for intervention_type, keywords in INTERVENTIONS.items():
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_norm):
                    interventions_dict[key]["event_categories"].add(intervention_type)
                    if keyword not in interventions_dict[key]["entities_detected"]:
                        interventions_dict[key]["entities_detected"].append(keyword)
                    break
    
    # Convert dict to list and join entities/categories
    extracted_interventions = []
    for intervention_data in interventions_dict.values():
        # Convert set to sorted, joined string
        categories = "; ".join(sorted(intervention_data["event_categories"]))
        entities = "; ".join(intervention_data["entities_detected"])
        
        extracted_interventions.append({
            "start_time": intervention_data["start_time"],
            "end_time": intervention_data["end_time"],
            "event_type": intervention_data["event_type"],
            "event_category": categories,
            "entity_detected": entities,
            "full_text": intervention_data["full_text"]
        })
    
    inter_path = export_to_csv(
        data=extracted_interventions,
        audio_chunk_path=Path(chunk_path),
        service="intervention",
        columns=INTER_COLUMNS, 
        empty_ok=True,
    )