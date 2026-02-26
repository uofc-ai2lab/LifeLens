from pathlib import Path
from src_audio.utils.export_to_csv import export_to_csv
from src_audio.utils.load_csv_file import load_csv_file
from src_audio.utils.calculate_mean import mean
from src_audio.domain.constants import (
    MEDICATIONS, LOW_CONFIDENCE_SCORE, MED_COLUMNS, AUDIT_COLUMNS
)
from src_audio.domain.entities import MedicationEntity, MedicationAdministration
from src_audio.services.medication_extraction_service.extractor import MedicationExtractor
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
    word_lower = word.lower()
    for canonical, info in MEDICATIONS.items():
        if canonical.lower() == word_lower:
            return canonical.lower()
        for alias in info.get("aliases", []):
            if alias.lower() == word_lower:
                return canonical.lower()
    return word_lower

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
        self._known: set[str] = set()

    def is_known(self, canonical: str) -> bool:
        """Return True if this medication has been registered in a prior chunk."""
        return canonical in self._known

    def register(self, canonical: str) -> None:
        """Mark a medication as administered so future chunks can detect confirmations."""
        self._known.add(canonical)

    def reset_session(self) -> None:
        """Clear all state. Call between independent recording sessions."""
        self._known.clear()


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
        owner     = preceding[-1] if preceding else sorted_drugs[0]
        assignment[owner.start_idx].append(attr)

    return assignment


def build_medication_record(
    drug_ent:  MedicationEntity,
    attr_ents: list[MedicationEntity],
    segment:   dict,
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
    route_ent  = next((e for e in attr_ents if e.entity == "ROUTE"),  None)

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

    Returns:
        list[MedicationAdministration]: Unique confirmed medications for
        the chunk with latest dosage/route values.
    """
    # Drug entity labels accepted as medication anchors
    accepted_entity_types: frozenset[str] = frozenset({"DRUG"})

    # Intents that proceed past the gate to record building
    actionable_intents: frozenset[str] = frozenset({"ADMINISTERED", "ORDERED", "REVISED"})

    # Within-chunk record store: canonical_name → MedicationAdministration
    chunk_records: dict[str, MedicationAdministration] = {}

    for segment in segments:
        ents = segment["entities"]
        original_text = segment["original_text"]

        # Pre-split entities by type so _assign_attrs_to_drugs can work
        # across the full sentence without a position-tracking index.
        drug_ents = [e for e in ents if e.entity in accepted_entity_types]
        attr_ents = [e for e in ents if e.entity not in accepted_entity_types]

        if not drug_ents:
            continue

        # Assign each DOSAGE / ROUTE to its nearest preceding drug.
        attr_map = _assign_attrs_to_drugs(drug_ents, attr_ents)

        for drug_ent in drug_ents:
            # Intent classification 
            intent = classify_intent(original_text, drug_ent.word)
            canonical = _resolve_canonical_name(drug_ent.word)

            if intent not in actionable_intents:
                audit_log.append({
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "intent": intent,
                    "medication": drug_ent.word,
                    "original_text": original_text,
                })
                continue

            owned_attrs = attr_map[drug_ent.start_idx]

            # Dosage revision: update existing record if present, otherwise create new
            if intent == "REVISED":
                record = build_medication_record(drug_ent, owned_attrs, segment)

                if not record.dosage:
                    dose = fallback_dosage_or_route(original_text, record, mode="dosage")
                    if dose:
                        record.dosage = dose
                        record.dosage_score = LOW_CONFIDENCE_SCORE

                if not record.route:
                    rte = fallback_dosage_or_route(original_text, record, mode="route")
                    if rte:
                        record.route = rte
                        record.route_score = LOW_CONFIDENCE_SCORE

                if canonical in chunk_records:
                    existing = chunk_records[canonical]
                    if record.dosage:
                        existing.dosage = record.dosage
                        existing.dosage_score = record.dosage_score
                    if record.route:
                        existing.route = record.route
                        existing.route_score = record.route_score
                else:
                    chunk_records[canonical] = record

                tracker.register(canonical)
                continue

            # Confirmation suppression 
            if canonical in chunk_records or tracker.is_known(canonical):
                continue

            # New administration record (build with whatever attributes are present, then apply fallbacks)
            record = build_medication_record(drug_ent, owned_attrs, segment)

            if not record.dosage:
                dose = fallback_dosage_or_route(original_text, record, mode="dosage")
                if not dose:
                    dose = get_default_dosage(record.medication)
                if dose:
                    record.dosage       = dose
                    record.dosage_score = LOW_CONFIDENCE_SCORE

            if not record.route:
                rte = fallback_dosage_or_route(original_text, record, mode="route")
                if rte:
                    record.route       = rte
                    record.route_score = LOW_CONFIDENCE_SCORE

            chunk_records[canonical] = record
            tracker.register(canonical)

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
            "end_time":   a.end_time,
            "event_type": "medication",
            "medication (confidence score)": (
                f"{a.medication} ({a.medication_score:.3f})"
                if a.medication else "Not Found"
            ),
            "dosage (confidence score)": (
                f"{a.dosage} ({a.dosage_score:.3f})"
                if a.dosage else "Not Found"
            ),
            "route (confidence score)": (
                f"{a.route} ({a.route_score:.3f})"
                if a.route else "Not Found"
            ),
            "full_text": "",
        }
        for a in administrations
    ]

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

    extractor = MedicationExtractor()
    transcript_data = []

    df = load_csv_file(transcript_path)
    log.info(f"Processing {len(df)} segments")

    for _, row in df.iterrows():
        extracted_entities = extractor.extract_medication_info_from_ner(row["text"])
        extracted_entities = postprocess_entities(extracted_entities, row["text"])
        transcript_data.append({
            "original_text": row["text"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "entities": extracted_entities,
        })

    confirmed_admins = extract_med_admins_with_confidence(
        transcript_data, tracker, audit_log
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

    # Export audit log entries produced by this chunk (if any)
    if audit_log:
        export_to_csv(
            data=audit_log,
            audio_chunk_path=Path(chunk_path),
            service="medX_audit",
            columns=AUDIT_COLUMNS,
            empty_ok=False,
        )
        log.info(f"{len(audit_log)} non-administered event(s) written to audit log")

    log.success("Medication extraction completed successfully!")