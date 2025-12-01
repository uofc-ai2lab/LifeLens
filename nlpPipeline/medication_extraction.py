import csv
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os
import pandas as pd


def load_transcript_csv(file_path="../output/transcript.csv"):
    """Load diarized transcript CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcript file not found at {file_path}")
    return pd.read_csv(file_path)


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
    "Normal Saline": ["NS"],
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

def create_all_med_list():
    """Create a list of all medication terms including aliases."""
    ALL_MEDICATION_TERMS = set()
    for med, aliases in MEDICATIONS.items():
        ALL_MEDICATION_TERMS.add(med.lower())
        for alias in aliases:
            ALL_MEDICATION_TERMS.add(alias.lower())
            
    MEDICATION_TERMS_SORTED = sorted(ALL_MEDICATION_TERMS, key=len, reverse=True)
    return MEDICATION_TERMS_SORTED

MED_LIST_SORTED = create_all_med_list()
        
        
def missed_medication_info(text):
    """Aim is to find any medication-related information that may have been missed by the NER model."""
    pattern = r'\b(' + '|'.join(re.escape(term) for term in MED_LIST_SORTED) + r')\b'
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

def process_ner_extraction_and_add_more_info_plus_meaning(extracted_data):
    """Process extracted medication data to add more context or meaning if needed."""
    med_objects = []
    
    possible_routes = [
        "iv", "intra-venous", "iv push", "ivp", "iv bolus", "ivb", "bolus",
        "im", "intramuscular",
        "po", "oral",
        "pr", "rectal",
        "subcutaneous", "sc", "sq",
        "sl", "sublingual",
        "io", "intraosseous",
        "neb", "nebulized",
        "inhaled",
        "topical",
    ]

    for item in extracted_data:
        medications = item.get("medications")
        dosages = item.get("dosages")
        text = item["original_sentence"].lower()
        only_words = re.findall(r'\b\w+\b', text)

        if medications and dosages:
            if len(medications) == len(dosages):
                for med in medications:
                    med_index = only_words.index(med.lower())

                    start = max(0, med_index - 3)
                    end = min(len(only_words), med_index + 3)
                    context_window = only_words[start:end]
                    route = next((w for w in context_window if w in possible_routes), None)

                    med_objects.append({
                        "medication": med,
                        "dosage": dosages[medications.index(med)],
                        "route": route if route else None,
                        "time": item["segment_time"]
                    })

        elif medications and not dosages:
            for med in medications:
                med_index = only_words.index(med.lower())

                start = max(0, med_index - 3)
                end = min(len(only_words), med_index + 3)
                context_window = only_words[start:end]
                route = next((w for w in context_window if w in possible_routes), None)

                med_objects.append({
                    "medication": med,
                    "dosage": None,
                    "route": route if route else None,
                    "time": item["segment_time"]
                })

    return med_objects
            
ROUTES = {
    "iv", "intra-venous", "iv push", "ivp", "iv bolus", "ivb", "bolus",
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

#might not need. Leave here for now
DOSAGES = {"mg", "ml", "g", "unit", "units", "mcg", "milligrams", "milliliters",
"grams", "micrograms", "l", "liters", "litres", "litre", "liter",
"mmol", "millimoles", "micron"}

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

                administrations.append(record)
            else:
                i += 1

    return administrations

def write_med_csv(administrations, filename):
    new_filename = filename.replace(".csv", "_medications_output.csv")
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
                if a.get("medication") else ""
            )
            dose = (
                f"{a['dosage']} ({a['dosage_score']:.3f})"
                if a.get("dosage") else ""
            )
            route = (
                f"{a['route']} ({a['route_score']:.3f})"
                if a.get("route") else ""
            )

            writer.writerow([
                a.get("time", ""),
                med,
                dose,
                route
            ])


def medication_extraction_pipeline(output_path="../output/mvc_trauma_transcript.csv"):
    """Run the medication extraction and CSV creation pipeline."""
    transcript_data = []
    df = load_transcript_csv(output_path)
    for _, row in df.iterrows():
        extracted_entities = extract_medication_info_from_ner(row["text"])
        already_found_meds = [e["word"] for e in extracted_entities if e["entity"] == "B-Medication"]
        possible_missed_medications = missed_medication_info(row["text"].lower())
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
        print(extracted_entities)
        nlp_data = {
            "original_text": row["text"],
            "segment_time": f"{row['start']} - {row['end']}",
            "entities": extracted_entities
        }
        transcript_data.append(nlp_data)
    
    full_medication_info = extract_med_admins_with_confidence(transcript_data)
    write_med_csv(full_medication_info, filename=output_path)

if __name__ == "__main__":
    medication_extraction_pipeline()