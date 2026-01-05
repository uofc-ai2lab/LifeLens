import os
import pandas as pd
import spacy
import re

# MedCAT
try:
    from medcat.cat import CAT
except:
    print("ERROR: MedCAT not installed or environment broken.")
    exit()

nlp = spacy.load("en_core_web_sm")

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data_p3.2")
model_pack_path = os.path.join(DATA_DIR, "medmen_wstatus_2021_oct.zip")
cat = CAT.load_model_pack(model_pack_path)


# INTERVENTION DEFINITIONS
INTERVENTIONS = {
    "CPR": ["cardiopulmonary resuscitation", "cpr", "chest compressions"],
    "Airway Management": ["airway", "intubation", "intubated", "endotracheal", "et tube"],
    "Needle Decompression": ["pneumothorax", "pneumo", "needle decompression", "thoracostomy", "chest decompression", "breathing thoracostomy"],
    "Spinal Immobilization": ["spinal motion restriction", "smr", "cervical collar", "c-collar", "backboard"],
    "Hemorrhage Control": ["pressure", "tourniquet", "bleeding control", "bandage", "dressing"],
    "IV/Fluid Administration": ["iv", "intravenous", "fluid", "saline", "line started", "io", "intraosseous"],
    "Ventilator": ["ventilator", "mechanical ventilation", "vent"]
}


def load_transcript_csv(file_path="./output/transcript.csv"):
    """Load transcript CSV file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcript file missing at {file_path}")
    return pd.read_csv(file_path, names=["start", "end", "text", "speaker"],
                       header=None, skiprows=1)


def normalize_text(text):
    """Normalize text for better matching"""
    replacements = {
        "pneumo.": "pneumothorax",
        "pneumo ": "pneumothorax ",
        "c collar": "cervical collar",
    }
    text = text.lower()
    for k, v in replacements.items():
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


async def run_nlp(output_path="./output/interventions_extracted.csv"):
    """Main extraction pipeline for interventions only"""
    df = load_transcript_csv()
    
    #  dict to group by start_time only (one row per start time)
    interventions_dict = {}
    
    for idx, row in df.iterrows():
        text = row["text"]
        text_norm = normalize_text(text)
        
        if not has_intervention_keyword(text_norm):
            continue 
        
        key = row["start"]  # One row per start time
        
        # Initialize entry
        if key not in interventions_dict:
            interventions_dict[key] = {
                "start_time": row["start"],
                "end_time": row["end"] if pd.notna(row["end"]) else "N/A",
                "event_type": "intervention",
                "event_categories": set(),
                "entities_detected": [],
                "full_text": row["text"],
            }
        
        # MedCAT for entity extraction
        medcat_out = cat.get_entities(text_norm, only_cui=False)
        
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
    
    # Output DataFrame
    if extracted_interventions:
        out_df = pd.DataFrame(extracted_interventions)
        out_df = out_df[["start_time", "end_time","event_type", "event_category",  
                         "entity_detected", "full_text"]]
        out_df.to_csv(output_path, index=False)
        print(f"Extracted {len(extracted_interventions)} interventions → {output_path}")
    else:
        print("No interventions detected in transcript.")
        pd.DataFrame(columns=["start_time", "end_time","event_type", "event_category", 
                              "entity_detected", "full_text"]).to_csv(output_path, index=False)


if __name__ == "__main__":
    run_nlp()