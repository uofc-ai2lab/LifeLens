from pathlib import Path
from src_audio.utils.export_to_csv import export_to_csv
from src_audio.utils.load_csv_file import load_csv_file
import pandas as pd
from src_audio.domain.constants import (
    MED_COLUMNS, SENTENCE_SPLIT,SENTENCE_END, 
    DEFAULT_DOSAGE_SCORE, ALIAS_TO_CANONICAL
)
from src_audio.domain.entities import MedicationEntity, MedicationAdministration
from src_audio.services.medication_extraction_service.extractor import (
    MedicationExtractor,
)
from src_audio.services.medication_extraction_service.postprocessing import (
    postprocess_entities,
    fallback_dosage_or_route,
    get_default_dosage,
    classify_intent,
)
from config.logger import Logger

log = Logger("[audio][medication]")


def _resolve_canonical_name(word: str) -> str:
    """
    Return the canonical lowercase name for a medication.

    Resolves aliases and fallback matches to the base name in MEDICATIONS
    (e.g., "epi" and "epinephrine" → same key). If not found, returns
    word.lower() for a stable fallback key.

    Args:
        word (str): Extracted medication name.

    Returns:
        str: Canonical lowercase name.
    """
    return ALIAS_TO_CANONICAL.get(word.lower(), word.lower())


class MedicationStateTracker:
    """
    Track medications administered across chunks in a session.

    A single instance is created per recording and shared across
    run_medication_extraction() calls to suppress duplicate confirmations
    in later chunks (e.g., "Epi is in" after "Give epi").

    Skips emitting repeat mentions unless a revision signal is present.
    """

    def __init__(self) -> None:
        # Set of canonical lowercase medication names seen across all chunks
        self._known: dict[str, float] = {}

    def is_known(self, canonical: str, current_time, window_minutes: float = 3.0) -> bool:
        """
        Check if a medication has been registered within the time window.
        Args:
            canonical (str): Canonical medication name to check.        
            current_time (str | int | float): Current timestamp in HH:MM:SS.mmm format or seconds.          
            window_minutes (float): Time window in minutes to consider for suppression.
        Returns:            
            bool: True if the medication is known and within the time window, False otherwise.
        """
        if canonical not in self._known:
            return False
        elapsed_minutes = (self._to_seconds(current_time) - self._known[canonical]) / 60.0
        return elapsed_minutes < window_minutes
    
    def register(self, canonical: str, timestamp) -> None:
        """Mark a medication as administered so future chunks can detect confirmations."""
        self._known[canonical] = self._to_seconds(timestamp)

    def reset_session(self) -> None:
        """Clear all state. Call between independent recording sessions."""
        self._known.clear()

    def _to_seconds(self, timestamp) -> float:
        """
        Convert HH:MM:SS.mmm string to seconds, or pass through if already numeric.
        Args:
            timestamp (str | int | float): Time in HH:MM:SS.mmm format or already in seconds.
        Returns:
            float: Time in seconds.
        """
        if isinstance(timestamp, (int, float)):
            return float(timestamp)
        try:
            h, m, s = timestamp.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            return 0.0


def _assign_attrs_to_drugs(
    drug_ents: list[MedicationEntity],
    attr_ents: list[MedicationEntity],
) -> dict[int, list[MedicationEntity]]:
    """
    Assign DOSAGE and ROUTE entities to drugs using a nearest-preceding rule.

    Each attribute is linked to the drug with the largest start_idx less
    than its own. If no drug precedes it (e.g., "10 mcg of epinephrine"),
    the attribute is assigned to the first drug.

    Prevents attributes from crossing drug boundaries in multi-drug sentences.

    Args:
        drug_ents (list[MedicationEntity]): Drug entities.
        attr_ents (list[MedicationEntity]): DOSAGE and ROUTE entities.

    Returns:
        dict[int, list[MedicationEntity]]: Mapping of drug start_idx to owned attributes.
    """
    sorted_drugs = sorted(drug_ents, key=lambda e: e.start_idx)
    assignment: dict[int, list[MedicationEntity]] = {
        d.start_idx: [] for d in sorted_drugs
    }

    for attr in attr_ents:
        preceding = [d for d in sorted_drugs if d.start_idx < attr.start_idx]
        owner = preceding[-1] if preceding else sorted_drugs[0]
        assignment[owner.start_idx].append(attr)

    return assignment


def _apply_fallbacks(
    record: MedicationAdministration, text: str, allow_default: bool
) -> None:
    # Dosage
    if not record.dosage:
        fallback_dosage_or_route(text, record, mode="dosage")
        dose = None
        if not record.dosage and allow_default:
            dose = get_default_dosage(record.medication)
        if dose:
            record.dosage = dose
            record.dosage_score = DEFAULT_DOSAGE_SCORE

    # Route
    if not record.route:
        fallback_dosage_or_route(text, record, mode="route")


def build_medication_record(
    drug_ent: MedicationEntity,
    attr_ents: list[MedicationEntity],
    segment: dict,
) -> MedicationAdministration:
    """
    Create a MedicationAdministration record from a drug and its attributes.

    Reads the first DOSAGE and ROUTE from pre-assigned attr_ents (already
    linked via nearest-preceding logic).

    Args:
        drug_ent (MedicationEntity): Drug entity ("DRUG").
        attr_ents (list[MedicationEntity]): Assigned DOSAGE/ROUTE entities.
        segment (dict): Contains 'start_time' and 'end_time'.

    Returns:
        MedicationAdministration: Populated record. Dosage and route are None
        if not provided.
    """
    dosage_ent = next((e for e in attr_ents if e.entity == "DOSAGE"), None)
    route_ent = next((e for e in attr_ents if e.entity == "ROUTE"), None)

    return MedicationAdministration(
        medication=drug_ent.word,
        medication_score=drug_ent.score,
        dosage=dosage_ent.word if dosage_ent else None,
        dosage_score=dosage_ent.score if dosage_ent else None,
        route=route_ent.word if route_ent else None,
        route_score=route_ent.score if route_ent else None,
        start_time=segment["start_time"],
        end_time=segment["end_time"],
    )


def extract_med_admins_with_confidence(
    segments: list[dict],
    tracker: MedicationStateTracker,
    audit_log: list[dict],
    chunk_path: str,
) -> list[MedicationAdministration]:
    """
    Extract confirmed medication administrations from transcript segments.

    Each drug entity is intent-classified before record creation.
    Only ADMINISTERED, ORDERED, and REVISED proceed; other intents are
    logged to audit_log and skipped.

    Deduplication:
    - Within-chunk: one record per canonical medication; confirmations
    are suppressed unless intent is REVISED (which updates in place).
    - Cross-chunk: tracker suppresses confirmations spanning chunks.

    REVISED updates an existing record if present, otherwise creates one.

    Fallback (after record creation):
        1) NER dosage/route
        2) Regex window fallback
        3) Default dosage (no default route)

    Args:
        segments (list[dict]): Transcript segments with entities and timestamps.
        tracker (MedicationStateTracker): Cross-chunk state.
        audit_log (list[dict]): Collects skipped non-administered events.
        chunk_path (str): Current audio chunk path for audit context.

    Returns:
        list[MedicationAdministration]: Unique confirmed medications for
        the chunk with latest dosage/route values.
    """
    # Drug entity labels accepted as medication anchors
    accepted_entity_types: frozenset[str] = frozenset({"DRUG"})

    # Intents that proceed past the gate to record building
    actionable_intents: frozenset[str] = frozenset(
        {"ADMINISTERED", "ORDERED", "REVISED"}
    )

    audio_chunk_file = Path(chunk_path).name

    # Within-chunk record store: canonical_name → MedicationAdministration
    chunk_records: dict[str, MedicationAdministration] = {}

    for segment in segments:
        ents = segment["entities"]
        text = segment["original_text"]

        drug_ents = [e for e in ents if e.entity in accepted_entity_types]
        if not drug_ents:
            continue

        attr_ents = [e for e in ents if e.entity not in accepted_entity_types]
        attr_map = _assign_attrs_to_drugs(drug_ents, attr_ents)

        for drug_ent in drug_ents:
            intent = classify_intent(text, drug_ent.word)
            canonical = _resolve_canonical_name(drug_ent.word)

            if intent not in actionable_intents:
                audit_log.append(
                    {
                        "audio_chunk_file": audio_chunk_file,
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"],
                        "intent": intent,
                        "medication": drug_ent.word,
                        "original_text": text,
                    }
                )
                continue

            owned_attrs = attr_map[drug_ent.start_idx]
            record = build_medication_record(drug_ent, owned_attrs, segment)

            if intent == "REVISED":
                _apply_fallbacks(record, text, allow_default=False)

                existing = chunk_records.get(canonical)
                if existing:
                    if record.dosage:
                        existing.dosage = record.dosage
                        existing.dosage_score = record.dosage_score
                    if record.route:
                        existing.route = record.route
                        existing.route_score = record.route_score
                else:
                    chunk_records[canonical] = record

                tracker.register(canonical, record.start_time)
                continue

            # Confirmation suppression
            if canonical in chunk_records or tracker.is_known(canonical, record.start_time):
                audit_log.append(
                    {
                        "audio_chunk_file": audio_chunk_file,
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"],
                        "intent": f"{intent} AND SUPRESSED AS DUPLICATE",
                        "medication": drug_ent.word,
                        "original_text": text,
                    }
                )
                tracker.register(canonical, record.start_time) # update timestamp of the medication mention even if suppressed, to prevent future duplicates in the session
                continue

            _apply_fallbacks(record, text, allow_default=True)
            chunk_records[canonical] = record
            tracker.register(canonical, record.start_time)

    return list(chunk_records.values())


def prepare_medication_rows(
    administrations: list[MedicationAdministration],
) -> list[dict]:
    """
    Convert MedicationAdministration records into row dicts for CSV export.

    Args:
        administrations (list[MedicationAdministration]): Confirmed records.

    Returns:
        list[MedicationAdministration]: List of finalized administrations with confidence scores.
    """
    return [
        {
            "start_time": a.start_time,
            "end_time": a.end_time,
            "event_type": "medication",
            "medication (confidence score)": (
                f"{a.medication} ({a.medication_score:.3f})"
                if a.medication
                else "Not Found"
            ),
            "dosage (confidence score)": (
                f"{a.dosage} ({a.dosage_score:.3f})" if a.dosage else "Not Found"
            ),
            "route (confidence score)": (
                f"{a.route} ({a.route_score:.3f})" if a.route else "Not Found"
            ),
            "full_text": "",
        }
        for a in administrations
    ]


def merge_incomplete_segments(df: pd.DataFrame) -> list[dict]:
    """
    Merge transcript rows into complete sentences before NER/intent processing.

    ASR segments on timing, not sentence boundaries, which can split
    dosage/route information or negations across rows. This function
    combines rows into full sentences to preserve context.

    Each segment retains the start_time of its first row and the end_time
    of its last row.

    Args:
        df (pd.DataFrame): Transcript with columns text, start_time, end_time.

    Returns:
        list[dict]: Sentence-level segments with text, start_time, end_time.
    """
    segments: list[dict] = []
    buffer_text: list[str] = []
    buffer_start: str | None = None
    buffer_end: str | None = None

    for _, row in df.iterrows():
        # Split on mid-row sentence boundaries before buffering.
        # A row with no internal boundary yields a single-element list.
        sub_sentences = SENTENCE_SPLIT.split(str(row["text"]).strip())

        for sub in sub_sentences:
            sub = sub.strip()
            if not sub:
                continue

            if buffer_start is None:
                buffer_start = row["start_time"]

            buffer_text.append(sub)
            buffer_end = row["end_time"]

            if SENTENCE_END.search(sub):
                segments.append(
                    {
                        "text": " ".join(buffer_text),
                        "start_time": buffer_start,
                        "end_time": buffer_end,
                    }
                )
                buffer_text, buffer_start, buffer_end = [], None, None

    # Trailing rows without terminal punctuation
    if buffer_text:
        segments.append(
            {
                "text": " ".join(buffer_text),
                "start_time": buffer_start,
                "end_time": buffer_end,
            }
        )

    return segments


def run_medication_extraction(
    chunk_path: str,
    transcript_path: str,
    tracker: MedicationStateTracker | None = None,
    audit_log: list[dict] | None = None,
) -> None:
    """
    Run the full medication extraction pipeline for one audio chunk.

    Processes the transcript through NER, dictionary fallback, intent
    classification, deduplication, and CSV export. tracker and audit_log
    persist cross-chunk state when provided; otherwise a temporary tracker
    is used.

    Args:
        chunk_path (str): Audio chunk path (used for output naming).
        transcript_path (str): Whisper transcript CSV path.
        tracker (MedicationStateTracker | None): Cross-chunk state.
        audit_log (list[dict] | None): Collects non-administered events.

    Returns:
        None
    """
    log.header("Starting Medication Extraction...")

    # Allow standalone calls without managing tracker/log externally
    if tracker is None:
        tracker = MedicationStateTracker()
    if audit_log is None:
        audit_log = []
    initial_audit_count: int = len(audit_log)
    extractor = MedicationExtractor()
    transcript_data = []

    df = load_csv_file(transcript_path)
    log.info(f"Loaded {len(df)} raw row(s) from transcript")

    segments = merge_incomplete_segments(df)
    log.info(f"Merged into {len(segments)} sentence segment(s)")

    texts = [seg["text"] for seg in segments]
    docs = extractor.nlp.pipe(texts, batch_size=extractor.pipe_batch_size)

    for seg, doc in zip(segments, docs):
        extracted_entities = extractor.extract_entities_from_doc(doc)
        extracted_entities = postprocess_entities(extracted_entities, seg["text"])
        transcript_data.append(
            {
                "original_text": seg["text"],
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "entities": extracted_entities,
            }
        )

    confirmed_admins = extract_med_admins_with_confidence(
        transcript_data, tracker, audit_log, chunk_path
    )

    rows = prepare_medication_rows(confirmed_admins)

    export_to_csv(
        data=rows,
        audio_chunk_path=Path(chunk_path),
        service="medX",
        columns=MED_COLUMNS,
        empty_ok=True,
    )

    log.info(f"{len(rows)} confirmed medication(s) found")
    if initial_audit_count < len(audit_log):
        log.info(
            f"{len(audit_log) - initial_audit_count} non-administered event(s) logged for audit review."
        )

    log.success("Medication extraction completed successfully!")
