import re
from functools import lru_cache
from rapidfuzz import process as fuzz_process, fuzz
from src_audio.domain.constants import (
    ROUTES, DOSAGES, TEXT_NUMBERS, NUMBER_PATTERN,
    MEDICATIONS, HIGH_CONFIDENCE_SCORE, DOSAGE_TOKEN_PATTERN, 
    PRE_NEGATION_TRIGGERS, POST_NEGATION_TRIGGERS,
    REVISED_SIGNALS, ADMINISTERED_SIGNALS, QUESTIONED_SIGNALS,
    CONSIDERED_SIGNALS, ORDERED_SIGNALS, FUZZY_CONF_SCALE, FUZZY_THRESHOLD
)
from src_audio.domain.entities import MedicationEntity, MedicationAdministration

@lru_cache(maxsize=1)
def create_all_med_list() -> list[str]:
    """
    Build one master list of all medication names including aliases.
    Cached to avoid recomputation for every sentence.

    Returns:
        list[str]: All medication terms sorted longest-first so that greedy
        matching gives multi-word names precedence over single-word ones.
    """
    all_med_terms = {name.lower() for name in MEDICATIONS.keys()}
    for info in MEDICATIONS.values():
        all_med_terms.update(a.lower() for a in info.get("aliases", []))
    return sorted(all_med_terms, key=len, reverse=True)

def missed_medication_info(text: str, med_list: list[str]) -> list[dict]:
    """
    Detect medication mentions missed by the NER model.

    Two-pass strategy:
    1) Exact word-boundary regex match (high confidence).
    2) Fuzzy n-gram matching (rapidfuzz) for transcription errors.

    Returns a list of dicts with:
        medication (str), start_idx (int),
        score (float: 0.90 exact; proportional fuzzy),
        match_type ("exact" | "fuzzy").

    Args:
        text (str): Original-case input text.
        med_list (list[str]): Sorted medication terms.

    Returns:
        list[dict]: Detected medications (may be empty).
    """
    if not text:
        return []

    # --- Pass 1: exact match ---
    pattern   = r'\b(' + '|'.join(re.escape(term) for term in med_list) + r')\b'
    med_regex = re.compile(pattern, re.IGNORECASE)

    matches: list[dict] = []
    covered: set[tuple[int, int]] = set()

    for m in med_regex.finditer(text):
        start, end = m.start(), m.end()
        matches.append({
            "medication": m.group(0),
            "start_idx":  start,
            "score":      HIGH_CONFIDENCE_SCORE,  # 0.90 for exact dict match
            "match_type": "exact",
        })
        covered.add((start, end))

    # --- Pass 2: fuzzy match ---

    # Build unigram, bigram, and trigram candidates.
    # Multi-word names ("normal saline") need bigram candidates to be found.
    word_spans = [
        (m.start(), m.end())
        for m in re.finditer(r'\S+', text)
    ]

    for n in range(1, 4):
        for i in range(len(word_spans) - n + 1):
            span_start = word_spans[i][0]
            span_end   = word_spans[i + n - 1][1]

            # Skip if already covered by an exact match
            if any(s <= span_start and span_end <= e for s, e in covered):
                continue

            candidate = text[span_start:span_end].lower()
            result = fuzz_process.extractOne(
                candidate,
                med_list,
                scorer=fuzz.token_set_ratio,
                score_cutoff=FUZZY_THRESHOLD,
            )

            if result:
                _, score, _ = result
                confidence = round((score / 100) * FUZZY_CONF_SCALE, 3)
                matches.append({
                    "medication": text[span_start:span_end],  # original case
                    "start_idx":  span_start,
                    "score":      confidence,
                    "match_type": "fuzzy",
                })
                covered.add((span_start, span_end))

    return matches

def _is_duplicate(
    word: str, 
    start_idx: int, 
    already_found: list[tuple[str, int]]
) -> bool:
    """
    Return True if the word is a near-duplicate of a previously found match.

    Duplicates are the same lowercase word occurring within 4 characters
    of an existing start index.

    Args:
        word (str): Candidate medication word.
        start_idx (int): Character index of the candidate word.
        already_found (list[tuple[str, int]]): List containing words already found by NER

    Returns:
        bool: True if a duplicate a duplicate is found, False otherwise.

    """
    word_lower = word.lower()
    return any(
        found_word == word_lower
        and abs(found_idx - start_idx) <= 4 # allow small position variance to account for minor tokenization differences
        for found_word, found_idx in already_found
    )

def postprocess_entities(
    entities: list[MedicationEntity],
    sentence: str,
) -> list[MedicationEntity]:
    """
    Augment NER-detected medications with dictionary fallback matches.

    Adds exact and fuzzy matches for medications the NER model may have
    missed. Deduplicates by character position so repeated mentions are
    preserved. Fuzzy matches retain their confidence scores.

    Args:
        entities (list[MedicationEntity]): Existing NER entities (mutated in place).
        sentence (str): Original-case sentence text.

    Returns:
        list[MedicationEntity]: All entities sorted by start_idx.
"""
    med_list = create_all_med_list()

    # Position-aware index of already-found drug entities.
    # Stored as (normalised_word, start_idx) pairs.
    # Two entries are the same span if they share the same word AND their
    # positions are within 5 characters of each other.
    already_found: list[tuple[str, int]] = [
        (e.word.lower(), e.start_idx)
        for e in entities
        if e.entity == "DRUG"
    ]

    for m in missed_medication_info(sentence, med_list):
        if not _is_duplicate(m["medication"], m["start_idx"], already_found):
            entities.append(MedicationEntity(
                entity="DRUG",
                word=m["medication"],
                start_idx=m["start_idx"],
                score=m["score"],
            ))
            # Register so subsequent loop iterations don't double-add
            already_found.append((m["medication"].lower(), m["start_idx"]))

    entities.sort(key=lambda e: e.start_idx)
    return entities


def _build_context_window(
    sentence: str,
    medication_word: str,
    window_size: int = 7,
) -> tuple[str, str]:
    """
    Return lowercased pre- and post-medication token windows.

    Tokens are stripped of leading/trailing punctuation before matching,
    so "epi?" aligns with "epi" and "ketamine." with "ketamine".

    Args:
        sentence (str): Input sentence (case-insensitive).
        medication_word (str): Medication name to locate.
        window_size (int): Tokens to include on each side.

    Returns:
        tuple[str, str]: (pre_window, post_window).
        If the medication is not found, returns (sentence.lower(), "").
    """
    words_raw   = sentence.lower().split()
    # Strip sentence-boundary punctuation from each token for comparison, preserving internal characters like hyphens 
    words_clean = [re.sub(r'^[^\w]+|[^\w]+$', '', w) for w in words_raw]
    med_tokens = [re.sub(r'^[^\w]+|[^\w]+$', '', t)
        for t in medication_word.lower().split()]
    n = len(med_tokens)

    for i in range(len(words_clean) - n + 1):
        if words_clean[i:i + n] == med_tokens:
            pre = " ".join(words_raw[max(0, i - window_size): i])
            post = " ".join(words_raw[i + n: min(len(words_raw), i + n + window_size)])
            return pre, post

    return sentence.lower(), ""


def classify_intent(sentence: str, medication_word: str) -> str:
    """
    Classify the administration intent of a medication mention.

    Uses a ±7-token rule-based context window and returns the highest-priority match:

    Priority (high → low):
        NEGATED > REVISED > ADMINISTERED > QUESTIONED > CONSIDERED > ORDERED (default)

    Negation:
        Pre- and post-negation triggers are evaluated only on their respective
        sides of the medication to avoid cross-negation (e.g., "instead of").
        Bare "no"/"not" use a 3-token proximity check.
        "no" is ignored if a REVISED signal is present (e.g., "no wait").

    Args:
        sentence (str): Transcript segment.
        medication_word (str): Medication as written.

    Returns:
        str: Intent label.
    """
    pre, post = _build_context_window(sentence, medication_word)
    full_window = f"{pre} {post}"
    sentence_lower = sentence.lower()

    close_pre = pre.split()[-3:]

    # Negation (highest priority)
    
    # --- Broad pre-negation (unambiguous triggers in the pre-window) ---
    if any(t in pre for t in PRE_NEGATION_TRIGGERS):
        return "NEGATED"
    
     # Close-range "no"
    if "no" in close_pre and not any(r in sentence_lower for r in REVISED_SIGNALS):
        return "NEGATED"

    # Close-range "not"
    if "not" in close_pre:
        return "NEGATED"

    # Post-negation
    if any(t in post for t in POST_NEGATION_TRIGGERS):
        return "NEGATED"

    # --- Revision: correcting a previously recorded dosage/route ---
    if any(s in sentence_lower for s in REVISED_SIGNALS):
        return "REVISED"
    
    # --- Administered: past-tense / confirmed delivery ---
    if any(s in full_window for s in ADMINISTERED_SIGNALS):
        return "ADMINISTERED"

    # --- Questioned: verifying a prior administration ---
    if any(s in sentence_lower for s in QUESTIONED_SIGNALS):
        return "QUESTIONED"

    # --- Considered: preparation or discussion without commitment ---
    if any(s in full_window for s in CONSIDERED_SIGNALS):
        return "CONSIDERED"

    # --- Ordered: imperative instruction ---
    if any(s in full_window for s in ORDERED_SIGNALS):
        return "ORDERED"

    # Default: treat an unqualified medication mention as an order
    return "ORDERED"


def fallback_dosage_or_route(
    sentence: str,
    med_record: MedicationAdministration,
    mode: str = "dosage",
) -> str | None:
    """
    Extract dosage or route from a ±5-word window if NER missed it.

    Args:
        sentence (str): Input sentence.
        med_record (MedicationAdministration): The record being built.
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
    window_text = " ".join(words[window_start:window_end])

    if mode == "dosage":
        tokens = [t.lower() for t in DOSAGE_TOKEN_PATTERN.findall(window_text)]
        for i, token in enumerate(tokens):
            if NUMBER_PATTERN.fullmatch(token):
                number_value = token
            elif token in TEXT_NUMBERS:
                number_value = str(TEXT_NUMBERS[token])
            else:
                continue
            
            if i + 1 < len(tokens) and tokens[i + 1] in DOSAGES:
                return f"{number_value} {tokens[i + 1]}"

        return None

    elif mode == "route":
        for token in re.findall(r"[a-z']+", window_text):
            if token in ROUTES:
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