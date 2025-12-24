import csv
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os
import pandas as pd
import argparse


def load_transcript_csv(file_path):
    """Load diarized transcript CSV file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Transcript file not found at {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        raise IOError(f"Error loading transcript file: {e}")


def extract_medication_info_from_ner(segmented_text):
    """Extract medication-related information from NER processed text segments."""
    NER_MODEL = "d4data/biomedical-ner-all"

    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL, use_fast=True)
    model.eval()

    enc = tokenizer(
        segmented_text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True
    )

    offsets = enc.pop("offset_mapping")[0]
    word_ids = enc.word_ids()

    with torch.no_grad():
        outputs = model(**enc)

    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=-1)

    # Collect token predictions by word_id
    word_groups = {}

    for i, word_id in enumerate(word_ids):
        if word_id is None:
            continue

        label_id = probs[i].argmax().item()
        label = model.config.id2label[label_id]
        score = probs[i, label_id].item()
        start, end = offsets[i].tolist()

        if label == "O":
            continue

        if word_id not in word_groups:
            word_groups[word_id] = {
                "entity": label,
                "start": start,
                "end": end,
                "scores": [score]
            }
        else:
            word_groups[word_id]["end"] = end
            word_groups[word_id]["scores"].append(score)

    # Finalize entities
    entities = []
    for w in sorted(word_groups.keys()):
        item = word_groups[w]
        text_span = segmented_text[item["start"]: item["end"]]
        scores = item.pop("scores")

        entities.append({
            "entity": item["entity"],
            "word": text_span,
            "start": item["start"], 
            # "end": item["end"], # I don't think we need this for now so commenting out
            "score": sum(scores) / len(scores)
        })
        
    return entities

MEDICATIONS = {
    "Pressure Infuser": [],
    "Normal Saline": ["NS", "Electrolyte"],
    "Syringe": [],
    "ASA": ["ASA"],
    "Adenosine": ["Adeno"],
    "Amiodarone": ["Amio"],
    "Atropine": ["Atropine"],
    "Calcium Chloride": ["CaCl₂"],
    "Carboprost": ["Hemabate"],
    "Clopidogrel": ["Plavix"],
    "D5W": ["D5W"],
    "Dextrose": ["D50"],
    "Epinephrine": ["Epi"],
    "Furosemide": ["Lasix"],
    "Isoproterenol": ["Iso", "Isuprel"],
    "Labetalol": ["Labetalol"],
    "Metoprolol": ["Metoprolol", "Lopressor"],
    "Nitroglycerin": ["NTG"],
    "Norepinephrine": ["Levo", "NE"],
    "Phenylephrine": ["Neo", "Phenyl"],
    "Sodium Bicarbonate": ["Bicarb"],
    "Salbutamol": ["Salbutamol", "Ventolin"],
    "Ipratropium": ["Ipratropium", "Atrovent"],
    "Dimenhydrinate": ["Gravol"],
    "Diphenhydramine": ["Benadryl"],
    "Metoclopramide": ["Maxeran", "Reglan"],
    "Loperamide": ["Imodium"],
    "Ondansetron": ["Zofran"],
    "Enoxaparin": ["Lovenox"],
    "Heparin": ["Heparin"],
    "Humulin R": ["Regular insulin", "R-insulin"],
    "Magnesium sulfate": ["Mag sulfate", "MgSO₄"],
    "Vitamin K": ["Phytonadione"],
    "Misoprostol": ["Miso"],
    "Oxytocin": ["Oxy", "Pitocin"],
    "Propofol": ["Propofol", "Diprivan"],
    "Lidocaine": ["Lido", "Lido oint"],
    "Naloxone": ["Narcan"],
    "Intralipid": ["Intralipid"],
    "Dexamethasone": ["Dex", "Decadron"],
    "Solu-Medrol": ["Solu-Medrol", "Methylpred"],
    "Plavix": ["Plavix"],
    "Ticagrelor": ["Brilinta"],
    "TNKase": ["TNK"],
    "Tranexamic Acid": ["TXA"],
    "Indomethacin": ["Indomethacin", "Indocin"],
    "Bug Spray": [],
    "NS flush": ["NS flush"],
    "Sterile Water": ["SW"]
}

def create_all_med_list(med_list=MEDICATIONS):
    """Create a list of all medication terms including aliases."""
    all_med_terms = set()
    for med, aliases in med_list.items():
        all_med_terms.add(med.lower())
        for alias in aliases:
            all_med_terms.add(alias.lower())
            
    med_terms_sorted = sorted(all_med_terms, key=len, reverse=True)
    return med_terms_sorted
        
        
def missed_medication_info(text, med_list):
    """Aim is to find any medication-related information that may have been missed by the NER model."""
    pattern = r'\b(' + '|'.join(re.escape(term) for term in med_list) + r')\b'
    med_regex = re.compile(pattern, re.IGNORECASE)
    if not text:
            return []

    matches = []
    for match in med_regex.finditer(text):
        matched_text = match.group(0)  # original case as in text
        start_idx = match.start()
        matches.append({
            "medication": matched_text,
            "start": start_idx,
        })
    
    return matches

def ensure_proper_medication_name(entities, sentence):
    """Ensure that medication names are properly captured in full from the sentence."""
    for ent in entities:
        found_med = ent["word"]
        tokens = re.findall(r"[\w'-]+", sentence)
        if found_med not in tokens:
            start = ent["start"]
            end = start
            while end < len(sentence) and not sentence[end].isspace():
                end += 1
            ent["word"] = sentence[start:end]
            
    return entities
        

ROUTES = {
    "infusion", "iv", "intra-venous", "iv push", "ivp", "iv bolus", "ivb", "bolus",
    "im", "intramuscular",
    "po", "oral",
    "pr", "rectal",
    "subcutaneous", "sc", "sq",
    "sl", "sublingual",
    "io", "intraosseous",
    "neb", "nebulized",
    "inhaled",
    "topical",
}

# still need to implement logic
TIMES = {"hours", "seconds", "minutes"}


DOSAGES = {d.lower() for d in {
    "mg", "ml", "g", "mg/ml", "mills", "unit", "units", "mcg",
    "milligrams", "milliliters", "grams", "micrograms",
    "l", "liters", "litres", "litre", "liter",
    "cc", "cc's", "drops", "drop", "tablet", "tablets",
    "puffs", "puff", "spray", "sprays", "inhaler",
    "capsule", "capsules", "pills", "pill",
    "inhalations", "mmol", "millimoles",
    "micron", "iu", "iu's", "international units"
}}

TEXT_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "half": 0.5,
    "quarter": 0.25
}

NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?(?:/\d+)?")

def fallback_dosage(sentence, med_start_idx):
    text = sentence.lower()
    after_med = text[med_start_idx:]

    # Tokenize: numbers, hyphens, slashes, and alpha words
    tokens = re.findall(r"\d+(?:\.\d+)?(?:/\d+)?|[a-z']+", after_med)

    for i in range(len(tokens) - 1):
        number_token = tokens[i]
        unit_token = tokens[i + 1]

        # Check numeric value
        is_number = NUMBER_PATTERN.fullmatch(number_token)

        # Check text-number (like "five")
        if not is_number and number_token in TEXT_NUMBERS:
            is_number = True

        # Unit validation
        if is_number and unit_token in DOSAGES:
            return f"{number_token} {unit_token}"

    return None

def fallback_route(sentence, med_start_idx):
    text = sentence.lower()
    after_med = text[med_start_idx:]

    # Tokenize words
    tokens = re.findall(r"[a-z']+", after_med)

    for token in tokens:
        if token in ROUTES:
            return token

    return None

def mean(scores):
    return sum(scores) / len(scores) if scores else None

def extract_med_admins_with_confidence(segments):
    administrations = []

    for segment in segments:
        ents = segment.get("entities", [])
        i = 0

        while i < len(ents):
            ent = ents[i]

            if ent["entity"] == "B-Medication" or ent["entity"] == "MEDICATION":
                record = {
                    "medication": ent["word"],
                    "medication_score": ent["score"],
                    "dosage": None,
                    "dosage_score": None,
                    "route": None,
                    "route_score": None,
                    "time": segment.get("segment_time")
                }

                i += 1

                while i < len(ents):
                    next_ent = ents[i]

                    # ---- DOSAGE ----
                    if next_ent["entity"] == "B-Dosage":
                        dosage_words = [next_ent["word"]]
                        dosage_scores = [next_ent["score"]]
                        i += 1

                        while i < len(ents) and ents[i]["entity"] == "I-Dosage":
                            dosage_words.append(ents[i]["word"])
                            dosage_scores.append(ents[i]["score"])
                            i += 1

                        record["dosage"] = " ".join(dosage_words)
                        record["dosage_score"] = mean(dosage_scores)
                        continue

                    # ---- ROUTE ----
                    if (
                        next_ent["entity"] in {"B-Lab_value", "B-Administration"}
                        and next_ent["word"].lower() in ROUTES
                    ):
                        record["route"] = next_ent["word"]
                        record["route_score"] = next_ent["score"]
                        i += 1
                        continue

                    break
                
                # fallback dosage and route extraction from text if not found by NER
                if not record["dosage"]:
                    fallback_dose = fallback_dosage(
                        segment["original_text"],
                        ent["start"]
                    )
                    if fallback_dose:
                        record["dosage"] = fallback_dose
                        record["dosage_score"] = 0.5  # Indicate lower confidence for fallback
                        
                if not record["route"]:
                    fallback_rte = fallback_route(
                        segment["original_text"],
                        ent["start"]
                    )
                    if fallback_rte:
                        record["route"] = fallback_rte
                        record["route_score"] = 0.5  # Indicate lower confidence for fallback
                administrations.append(record)
            else:
                i += 1
    return administrations

def write_med_csv(administrations, filename):
    new_filename = filename.replace(".csv", "_medications_output.csv")
    try:
        with open(new_filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "TIME",
                "MEDICATION (CONFIDENCE SCORE)",
                "DOSAGE (CONFIDENCE SCORE)",
                "ROUTE (CONFIDENCE SCORE)"
            ])

            # Rows
            for a in administrations:
                med = (
                    f"{a['medication']} ({a['medication_score']:.3f})"
                    if a.get("medication") else "Not Found"
                )
                dose = (
                    f"{a['dosage']} ({a['dosage_score']:.3f})"
                    if a.get("dosage") else "Not Found"
                )
                route = (
                    f"{a['route']} ({a['route_score']:.3f})"
                    if a.get("route") else "Not Found"
                )

                writer.writerow([
                    a.get("time", ""),
                    med,
                    dose,
                    route
                ])
    except Exception as e:
        raise IOError(f"Error writing medication output file: {e}")


def medication_extraction_pipeline(transcript_path):
    """Run the medication extraction and CSV creation pipeline."""
    transcript_data = []
    df = load_transcript_csv(transcript_path)
    for _, row in df.iterrows():
        extracted_entities = extract_medication_info_from_ner(row["text"])
        already_found_meds = [e["word"] for e in extracted_entities if e["entity"] == "B-Medication"]
        med_list = create_all_med_list()
        possible_missed_medications = missed_medication_info(row["text"].lower(), med_list)
        if possible_missed_medications and extracted_entities:
            for med in possible_missed_medications:
                if med["medication"] not in map(str.lower, already_found_meds):
                    extracted_entities.append({
                        "entity": "MEDICATION",
                        "word": med["medication"],
                        "start": med["start"],
                        "score": 1.0  # Since we found through exact match. (Want to implement fuzzy matching later which will likely have a proper score attached to it)
                    })
        extracted_entities.sort(key=lambda x: x["start"])
        extracted_entities = ensure_proper_medication_name(extracted_entities, row["text"])
        nlp_data = {
            "original_text": row["text"],
            "segment_time": f"{row['start']} - {row['end']}",
            "entities": extracted_entities
        }
        transcript_data.append(nlp_data)
    
    full_medication_info = extract_med_admins_with_confidence(transcript_data)
    write_med_csv(full_medication_info, filename=transcript_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract medication information from transcript CSV.")
    parser.add_argument("transcript_path", help="Path to the input transcript CSV file.")
    args = parser.parse_args()
    medication_extraction_pipeline(args.transcript_path)