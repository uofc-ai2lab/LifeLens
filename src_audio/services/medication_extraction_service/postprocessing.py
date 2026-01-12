import re
from functools import lru_cache
from src.domain.constants import ROUTES, DOSAGES, TEXT_NUMBERS, NUMBER_PATTERN, MEDICATIONS, LOW_CONFIDENCE_SCORE, HIGH_CONFIDENCE_SCORE
from src.domain.entities import MedicationEntity

@lru_cache(maxsize=1)
def create_all_med_list(med_list=MEDICATIONS) -> list[str]:
    """
    Build one master list of all medication names including aliases.
    Cached to avoid recomputation for every sentence.
    """
    all_med_terms = set()
    for med, aliases in med_list.items():
        all_med_terms.add(med.lower())
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
                start_idx=m["start"],
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
        # Tokenize numbers, hyphens, slashes, and words
        tokens = re.findall(r"\d+(?:\.\d+)?(?:/\d+)?|[a-z']+", after_med)
        for i in range(len(tokens) - 1):
            number_token, unit_token = tokens[i], tokens[i + 1]
            
            # Numeric check
            is_number = NUMBER_PATTERN.fullmatch(number_token)
            if not is_number and number_token in TEXT_NUMBERS:
                is_number = True

            if is_number and unit_token in DOSAGES:
                return f"{number_token} {unit_token}"
    
    elif mode == "route":
        # Tokenize words
        tokens = re.findall(r"[a-z']+", after_med)
        for token in tokens:
            if token in ROUTES:
                return token
    
    return None