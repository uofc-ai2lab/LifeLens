import os
import pandas as pd
import spacy

# MedCAT
try:
    from medcat.cat import CAT
except:
    print("ERROR: MedCAT not installed or environment broken.")
    exit()

# Initialize SpaCy
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
    """Match text to intervention category"""
    entity_lower = entity_text.lower()
    
    for intervention_type, keywords in INTERVENTIONS.items():
        for keyword in keywords:
            if keyword in entity_lower or keyword in text_norm:
                return intervention_type
    return None


async def run_nlp(output_path="./output/interventions_extracted.csv"):
    """Main extraction pipeline for interventions only"""
    df = load_transcript_csv()
    
    # Use dict to group by (start_time, intervention_type)
    interventions_dict = {}
    
    for idx, row in df.iterrows():
        text = row["text"]
        text_norm = normalize_text(text)
        key = (row["start"], text)  # Unique key per line
        
        # MedCAT for entity extraction
        medcat_out = cat.get_entities(text_norm, only_cui=False)
        
        # Process MedCAT entities
        for ent in medcat_out["entities"].values():
            entity_text = ent["pretty_name"]
            intervention_type = match_intervention(text_norm, entity_text)
            
            if intervention_type:
                if key not in interventions_dict:
                    interventions_dict[key] = {
                        "start_time": row["start"],
                        "end_time": row["end"] if pd.notna(row["end"]) else "N/A",
                        "event_type": "intervention",
                        "event_category": intervention_type,
                        "entities_detected": [],
                        "full_text": row["text"],
                    }
                # Add entity to list if not already there
                if entity_text not in interventions_dict[key]["entities_detected"]:
                    interventions_dict[key]["entities_detected"].append(entity_text)
        
        # Keyword-search fallback
        for intervention_type, keywords in INTERVENTIONS.items():
            for keyword in keywords:
                if keyword in text_norm:
                    if key not in interventions_dict:
                        interventions_dict[key] = {
                            "start_time": row["start"],
                            "end_time": row["end"] if pd.notna(row["end"]) else "N/A",
                            "event_type": "intervention",
                            "event_category": intervention_type,
                            "entities_detected": [],
                            "full_text": row["text"],
                        }
                    if keyword not in interventions_dict[key]["entities_detected"]:
                        interventions_dict[key]["entities_detected"].append(keyword)
                    break
    
    # Convert dict to list and join entities
    extracted_interventions = []
    for intervention_data in interventions_dict.values():
        intervention_data["entity_detected"] = "; ".join(intervention_data["entities_detected"])
        del intervention_data["entities_detected"]
        extracted_interventions.append(intervention_data)
    
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