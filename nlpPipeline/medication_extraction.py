import json
import re


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

def missed_medication_info(tokenized_text, med_dict=MEDICATIONS):
    """Aim is to find any medication-related information that may have been missed by the NER model."""
    possible_meds = [item for pair in med_dict.items() for item in pair]
    print(possible_meds)
    matches = [t for t in tokenized_text if t in possible_meds]
    return list(set(matches))
    
    
def collect_info_ner_found(data):
    """Aim is to extract medication-related entities from NLP processed data, and find all related information."""
    extracted_med_info = {}
    medications = []
    dosages = []
    frequencies = []
    entities = data["entities"]
    sentence = data["original_text"]
    only_words = re.findall(r'\b\w+\b', sentence.lower())
    possible_repeats = []
    for entity in entities:
        if entity["entity"].lower() == "medication" and entity["score"] > 0.85:
            med_name = entity["text"]
            result = next((w for w in only_words if med_name in w), None) # Potential problem is that if we have multiple mentioned meds in a sentence that have all been broken up and have same starting but they are different meds. We will miss initial one. Will need to revist this logic later.
            if result:
                 if result in possible_repeats:
                    continue
                 else:
                    med_name_in_text = result
                    possible_repeats.append(result)
                    medications.append(med_name_in_text)
            else:
                continue
        elif entity["entity"].lower() == "dosage" and entity["score"] > 0.85:
            dosage_info = entity["text"]
            dosages.append(dosage_info)


    missed = missed_medication_info(only_words, MEDICATIONS)
    if missed:
        all_medications = medications + missed
        final_medications_list = list(set(all_medications))
    else:
        final_medications_list = medications

    if len(final_medications_list) > 0:
        extracted_med_info["medications"] = final_medications_list
        extracted_med_info["dosages"] = dosages if len(dosages) > 0 else None
        extracted_med_info["segment_time"] = data["segment_time"]
        extracted_med_info["original_sentence"] = sentence
    return extracted_med_info


def process_ner_extraction_and_add_more_info_plus_meaning(extracted_data):
    med_objects = []
    """Process extracted medication data to add more context or meaning if needed."""
    possible_dosage_units = ["mg", "ml", "g", "unit", "units", "mcg", "milligrams", "milliliters", "grams", "micrograms", "l", "liters", "litres", "litre", "liter", "mmol", "millimoles", "micron"]
    possible_routes = [
    "iv", "intra-venous", "iv push", "ivp", "iv bolus", "ivb" "bolus"
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
    # still need to implement logic
    possiblle_rates = ["hours", "seconds", "minutes"]

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
                        "time": item["segment_time"]})
                
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
                    "time": item["segment_time"]})
                               
                               
    return med_objects
            
def generate_medication_nlg(med_entries):
    """
    med_entries: list of dicts, each possibly containing:
        - 'medication'
        - 'dosage'
        - 'route'
        - 'time'

    Returns: list of NLG sentences (strings)
    """

    sentences = []

    for entry in med_entries:
        med = entry.get("medication")
        dose = entry.get("dosage")
        route = entry.get("route")
        time = entry.get("time")

        # Skip empty dicts or dicts missing even the medication
        if not med:
            continue

        # Case 1: medication + dosage + route → full sentence
        if med and dose and route:
            if time:
                sentences.append(
                    f"{med} was administered {route} at {dose} around {time}."
                )
            else:
                sentences.append(
                    f"{med} was administered {route} at {dose}."
                )
            continue

        # Case 2: medication + dosage, but NO route
        if med and dose and not route:
            if time:
                sentences.append(
                    f"{med} at {dose} was mentioned at {time}, "
                    f"but the administration route could not be identified."
                )
            else:
                sentences.append(
                    f"{med} at {dose} was mentioned, but the administration route could not be identified."
                )
            continue

        # Case 3: medication + route, but NO dosage
        if med and route and not dose:
            if time:
                sentences.append(
                    f"{med} was administered via the {route} route around {time}, "
                    f"but the dosage could not be identified."
                )
            else:
                sentences.append(
                    f"{med} was administered via the {route} route, but the dosage could not be identified."
                )
            continue

        # Case 4: only medication exists
        if med and not dose and not route:
            if time:
                sentences.append(
                    f"{med} was mentioned around {time}, but the dosage and route could not be identified."
                )
            else:
                sentences.append(
                    f"{med} was mentioned, but the dosage and route could not be identified."
                )
            continue

    return sentences

                

def run_extraction_and_meaning():
    with open("nlpOutput.json", "r") as f:
        nlp_data = json.load(f)

    ner_collected_info = []
    ner_collected_info.extend(
    collect_info_ner_found(item) for item in nlp_data if collect_info_ner_found(item))
    #print(ner_collected_info)
    final_med_objects = process_ner_extraction_and_add_more_info_plus_meaning(ner_collected_info)
    nlg_sentences = generate_medication_nlg(final_med_objects)
    print(final_med_objects)
    print("Generated NLG Sentences:")
    for sentence in nlg_sentences:
        print("-", sentence)
        

run_extraction_and_meaning()