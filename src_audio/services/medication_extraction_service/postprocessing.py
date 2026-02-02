import re
from functools import lru_cache
from src_audio.domain.constants import ROUTES, DOSAGES, TEXT_NUMBERS, NUMBER_PATTERN, MEDICATIONS, LOW_CONFIDENCE_SCORE, HIGH_CONFIDENCE_SCORE, DOSAGE_TOKEN_PATTERN
from src_audio.domain.entities import MedicationEntity

@lru_cache(maxsize=1)
def create_all_med_list(med_list=MEDICATIONS) -> list[str]:
    """
    Build one master list of all medication names including aliases.
    Cached to avoid recomputation for every sentence.
    """
    all_med_terms = set()
    for med, med_info in med_list.items():
        all_med_terms.add(med.lower())
        aliases = med_info.get("aliases", [])
        for alias in aliases:
            all_med_terms.add(alias.lower())
            
    return sorted(all_med_terms, key=len, reverse=True)

def missed_medication_info(text, med_list):
    """
    Find medications in text that the NER model might have missed.
    Args:
        text (str): Input sentence or text.
        med_list (list[str]): List of medication names to search for.
    Returns:
        List[dict]: List of missed medications with keys: medication (str), start_index (int) 
    """
    if not text:
        return []
    pattern = r'\b(' + '|'.join(re.escape(term) for term in med_list) + r')\b' # regex from the medication list
    med_regex = re.compile(pattern, re.IGNORECASE)
    
    matches = []
    for match in med_regex.finditer(text): # Scan the sentence for exact word matches
        matched_text = match.group(0)  # original case as in text
        start_idx = match.start()
        matches.append({
            "medication": matched_text,
            "start_idx": start_idx,
        })
    
    return matches

def ensure_proper_medication_name(entities, sentence):
    """
    Correct partial or cut-off medication names using the sentence context.
    Args:
        entities (list[MedicationEntity]): NER-extracted entities.
        sentence (str): Original sentence.
    Returns:
        list[MedicationEntity]: Entities with corrected medication names.
    """
    for ent in entities:
        found_med = ent.word
        tokens = re.findall(r"[\w'-]+", sentence)
        if found_med not in tokens:
            start_idx = ent.start_idx
            end_idx = start_idx
            while end_idx < len(sentence) and not sentence[end_idx].isspace():
                end_idx += 1
            ent.word = sentence[start_idx:end_idx]
            
    return entities

def postprocess_entities(entities, sentence):
    """
    Add any missed medications from a master list and ensure proper spans.
    Args:
        entities (list[MedicationEntity]): NER-extracted entities.
        sentence (str): Original sentence.
    Returns:
        list[MedicationEntity]: Fully post-processed entities.
    """
    med_list = create_all_med_list()
    already_found = {e.word.lower() for e in entities if e.entity.startswith("B-Medication")}
    missed = missed_medication_info(sentence.lower(), med_list)
    for m in missed:
        if m["medication"].lower() not in already_found:
            entities.append(MedicationEntity(
                entity="MEDICATION",
                word=m["medication"],
                start_idx=m["start_idx"],
                score=HIGH_CONFIDENCE_SCORE
            ))
    entities.sort(key=lambda e: e.start_idx)
    return ensure_proper_medication_name(entities, sentence)
        
def fallback_dosage_or_route(sentence: str, med_start_idx: int, mode: str = "dosage") -> str | None:
    """
    Extract medication dosage or route if NER missed it.
    
    Args:
        sentence (str): Input sentence.
        med_start_idx (int): Start index of medication in sentence.
        mode (str): "dosage" or "route" — what to extract.
        
    Returns:
        Optional[str]: Extracted dosage or route, or None if not found.
    """
    text = sentence.lower()
    after_med = text[med_start_idx:]

    if mode == "dosage":
        # Tokenize: Grab numbers/fractions OR words
        tokens = DOSAGE_TOKEN_PATTERN.findall(after_med)
        
        for i in range(len(tokens) - 1):
            number_token, unit_token = tokens[i], tokens[i + 1]
            
        # Check if the second word is a valid unit
        if unit_tok in DOSAGES:
            # Check if the first word is a numeric string
            if NUMBER_PATTERN.fullmatch(num_tok):
                return f"{num_tok} {unit_tok}"
            
            # Check if the first word is a text-based number
            if num_tok in TEXT_NUMBERS:
                return f"{TEXT_NUMBERS.get(num_tok)} {unit_tok}" 

    elif mode == "route":
        # Look through whole sentence (that has been tokenized) for a possible route.
        for token in re.findall(r"[a-z']+", after_med):
            if token in ROUTES:
                return token

    return None

def get_default_dosage(medication_name: str) -> str | None:
    med_lower = medication_name.lower()
    for med_name, info in MEDICATIONS.items():
        if med_name.lower() == med_lower:
            return info.get("default_dosage")
        if med_lower in [a.lower() for a in info.get("aliases", [])]:
            return info.get("default_dosage")
    return None