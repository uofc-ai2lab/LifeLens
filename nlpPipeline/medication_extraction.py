import json
import re

def collect_info_ner_found(data):
    """Aim is to extract medication-related entities from NLP processed data, and find all related information."""
    medication_event = {}
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
            medication_event["dosage"] = dosage_info

        # Is frequency needed separately from dosage? I don't think this is important in a truama context, but I'll leave it here for now.
        elif entity["entity"].lower() == "frequency" and entity["score"] > 0.85:
            frequency_info = entity["text"]
            medication_event["frequency"] = frequency_info

    if len(medication_event) > 0:
        medication_event["segment_time"] = data["segment_time"]
        medication_event["original_sentence"] = sentence
        return medication_event
    else:
        return None

def process_ner_extraction_and_add_more_info_plus_meaning(extracted_data):
    formatted_pretty_print = []
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

    for item in extracted_data:
        output_text = ''
        medication = item.get("medication")
        dosage = item.get("dosage")
        frequency = item.get("frequency")
        text = item["original_sentence"].lower()
        route = any(r for r in possible_routes if r in text)
        
        if dosage and not medication:
            # Medication might
        
        if medication and not dosage:
            found_dosage_units = any(unit in text for unit in possible_dosage_units)
            if len(found_dosage_units) > 0:
                dosage = f"Possible Dosage(s) {', '.join(found_dosage_units)} mentioned but not clearly identified."
            else:
                dosage = None
                
        output_text += f"Medication: {medication}\n"
        if dosage:
            output_text += f"Dosage: {dosage}\n"
        else:
            output_text += "Dosage: Not specified\n"
        if frequency:
            output_text += f"Frequency: {frequency}\n"
            
        if route:
            output_text += f"Route: {route}\n"
        else:
            output_text += "Route: Not specified\n"
        output_text += f"Medication event occured at : {item['segment_time']}\n"
        
        formatted_pretty_print.append(output_text)

def run_extraction_and_meaning():
    with open("nlpOutput.json", "r") as f:
        nlp_data = json.load(f)

    ner_collected_info = []
    ner_collected_info.extend(
    collect_info_ner_found(item) for item in nlp_data if collect_info_ner_found(item))
    process_ner_extraction_and_add_more_info_plus_meaning(ner_collected_info)
        

run_extraction_and_meaning()