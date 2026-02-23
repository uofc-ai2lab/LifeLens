import re
from functools import lru_cache
from src_audio.domain.constants import ROUTES, DOSAGES, TEXT_NUMBERS, NUMBER_PATTERN, MEDICATIONS, LOW_CONFIDENCE_SCORE, HIGH_CONFIDENCE_SCORE, DOSAGE_TOKEN_PATTERN
from src_audio.domain.entities import MedicationEntity, MedicationAdministration

@lru_cache(maxsize=1)
def create_all_med_list() -> list[str]:
    """Build one master list of all medication names including aliases. 
    Cached to avoid recomputation for every sentence. 

    Args:
        None
    Returns:
        list[str]: List of all medication terms to check for in text.
    """

    all_med_terms = {name.lower() for name in MEDICATIONS.keys()}
    for info in MEDICATIONS.values():
        all_med_terms.update(a.lower() for a in info.get("aliases", []))

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
    tokens = set(re.findall(r"[\w'-]+", sentence))

    for ent in entities:
        if ent.word in tokens:
            continue

        match = re.match(r"[\w'-]+", sentence[ent.start_idx:])
        if match:
            ent.word = match.group(0)

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
    corrected_entities = ensure_proper_medication_name(entities, sentence)
    already_found = {e.word.lower() for e in corrected_entities if e.entity.startswith("B-Medication")}
    missed = missed_medication_info(sentence.lower(), med_list)
    for m in missed:
        if m["medication"].lower() not in already_found:
            corrected_entities.append(MedicationEntity(
                entity="MEDICATION",
                word=m["medication"],
                start_idx=m["start_idx"],
                score=HIGH_CONFIDENCE_SCORE
            ))
    corrected_entities.sort(key=lambda e: e.start_idx)
    return corrected_entities

def fallback_dosage_or_route(sentence: str, med_record: MedicationAdministration, mode: str = "dosage") -> str | None:
    """
    Extract medication dosage or route if NER missed it. We do this by looking at a 
    small window of words around the medication mentioned in the sentence and extracting 
    dosage or route information from that local context using predefined lists and patterns. 

    Args:
        sentence (str): Input sentence.
        med_start_idx (int): Character start index of medication.
        mode (str): "dosage" or "route".

    Returns:
        Optional[str]: Extracted dosage or route, or None if not found.
    """

    text = sentence.lower()
    medication = med_record.medication.lower()

    # Tokenize sentence and medication
    words = re.findall(r"\S+", text)
    med_tokens = re.findall(r"\S+", medication)

    if not med_tokens:
        return None

    # Find medication token sequence in sentence tokens
    med_word_index = None

    for i in range(len(words) - len(med_tokens) + 1):
        if words[i:i + len(med_tokens)] == med_tokens:
            med_word_index = i
            break

    if med_word_index is None:
        return None  # Medication not found in sentence

    # Build local word window (±5 words)
    window_start = max(0, med_word_index - 5)
    window_end = min(len(words), med_word_index + len(med_tokens) + 5)
    window_words = words[window_start:window_end]
    window_text = " ".join(window_words)

    if mode == "dosage":
        # use locals to avoid repeated global lookups
        token_pattern = DOSAGE_TOKEN_PATTERN
        number_re = NUMBER_PATTERN
        text_numbers = TEXT_NUMBERS
        dosages = DOSAGES

        tokens = [t.lower() for t in token_pattern.findall(window_text)]
        for i, token in enumerate(tokens):
            if number_re.fullmatch(token):
                number_value = token
            elif token in text_numbers:
                number_value = str(text_numbers[token])
            else:
                continue

            if i + 1 < len(tokens) and tokens[i + 1] in dosages:
                return f"{number_value} {tokens[i + 1]}"

        return None

    elif mode == "route":
        routes = ROUTES
        for token in re.findall(r"[a-z']+", window_text):
            if token in routes:
                return token

        return None

    return None

def get_default_dosage(medication_name: str) -> str | None:
    med_lower = medication_name.lower()
    for med_name, info in MEDICATIONS.items():
        if med_name.lower() == med_lower:
            return info.get("default_dosage")
        for a in info.get("aliases", []):
            if med_lower == a.lower():
                return info.get("default_dosage")
    return None