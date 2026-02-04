from pathlib import Path
from src_audio.utils.export_to_csv import export_to_csv
from src_audio.utils.load_csv_file import load_csv_file 
from src_audio.utils.calculate_mean import mean
from src_audio.domain.constants import ROUTES, LOW_CONFIDENCE_SCORE, HIGH_CONFIDENCE_SCORE, MED_COLUMNS
from src_audio.domain.entities import MedicationEntity, MedicationAdministration
from src_audio.services.medication_extraction_service.extractor import MedicationExtractor
from src_audio.services.medication_extraction_service.postprocessing import postprocess_entities, fallback_dosage_or_route

def build_medication_record(
    ent: MedicationEntity, 
    ents: list[MedicationEntity], 
    segment: dict, 
    i: int
) -> tuple[MedicationAdministration, int]:
    """
    Build a MedicationAdministration record from a medication entity and subsequent related entities.
    Args:
        ent (MedicationEntity): The main medication entity (B-Medication or MEDICATION).
        ents (list[MedicationEntity]): All entities in the segment, sorted by start index.
        segment (dict): Original segment containing 'start', 'end', 'original_text', etc.
        i (int): Current index of the medication entity in ents.
    Returns:
        tuple:
            - MedicationAdministration: Filled record with medication, dosage, route, and scores.
            - int: Updated index after processing related entities.
    """
    
    record = MedicationAdministration(
        medication=ent.word,
        medication_score=ent.score,
        dosage=None,
        dosage_score=None,
        route=None,
        route_score=None,
        start_time=segment["start_time"],
        end_time=segment["end_time"],
    )
    
    i+=1
    while i < len(ents):
        next_ent = ents[i]
        if next_ent.entity == "B-Dosage":
            words, scores = [next_ent.word], [next_ent.score]
            i+=1 
            while i < len(ents) and ents[i].entity == "I-Dosage":
                words.append(ents[i].word)
                scores.append(ents[i].score)
                i+=1 
            record.dosage = " ".join(words)
            record.dosage_score = mean(scores)
            continue
        if (
            next_ent.entity in {"B-Lab_value", "B-Administration"}
            and next_ent.word.lower() in ROUTES
        ):
            record.route = next_ent.word
            record.route_score = next_ent.score
            i += 1
            continue

        break

    return record, i

def extract_med_admins_with_confidence(segments: list[dict]) -> list[MedicationAdministration]:
    """
    Extract all medication administrations including fallback dosage and route.

    Args:
        segments (list[dict]): List of segments with keys:
            - 'original_text': str
            - 'start_time': float
            - 'end_time': float
            - 'entities': List[MedicationEntity]

    Returns:
        list[MedicationAdministration]: List of finalized administrations with confidence scores.
    """
    administrations = []
    for segment in segments:
        ents = segment["entities"]
        i = 0
        while i < len(ents):
            ent = ents[i]

            if ent.entity in {"B-Medication", "MEDICATION"}:
                record, i = build_medication_record(ent, ents, segment, i)

                if not record.dosage:
                    dose = fallback_dosage_or_route(segment["original_text"], ent.start_idx, mode="dosage")
                    if dose:
                        record.dosage = dose
                        record.dosage_score = LOW_CONFIDENCE_SCORE

                if not record.route:
                    rte = fallback_dosage_or_route(segment["original_text"], ent.start_idx, mode="route")
                    if rte:
                        record.route = rte
                        record.route_score = LOW_CONFIDENCE_SCORE

                administrations.append(record)
            else:
                i += 1

    return administrations

def prepare_medication_rows(administrations: list[MedicationAdministration]) -> list[dict]:
    """
    Convert MedicationAdministration records into row dictionaries suitable for CSV export.
    Args:
        administrations (list[MedicationAdministration]): List of medication administrations.
    Returns:
        list[MedicationAdministration]: List of finalized administrations with confidence scores.
    """
    return [{
        "start_time": a.start_time,
        "end_time": a.end_time,
        "event_type": "medication",
        "medication (confidence score)" : (
            f"{a.medication} ({a.medication_score:.3f})" if a.medication else "Not Found"
        ),
        "dosage (confidence score)" : (
            f"{a.dosage} ({a.dosage_score:.3f})" if a.dosage else "Not Found"
        ),
        "route (confidence score)" : (
            f"{a.route} ({a.route_score:.3f})" if a.route else "Not Found"
        ),
        "full_text": ""
    } for a in administrations]

def medication_extraction_pipeline(chunk_path: str, transcript_path: str, extractor: MedicationExtractor) -> None:
    """
    Run the full medication extraction pipeline:
    - Load transcript CSV
    - Extract NER medication entities
    - Postprocess missed medications
    - Extract dosage and route
    - Prepare rows and export to CSV

    Args:
        transcript_path (str): Path to transcript CSV file.
        extractor (MedicationExtractor): NER model instance.

    Returns:
        None
    """
    transcript_data = []
    df = load_csv_file(transcript_path)
    for _, row in df.iterrows():
        extracted_entities = extractor.extract_medication_info_from_ner(row["text"])
        extracted_entities = postprocess_entities(extracted_entities, row["text"])
        nlp_data = {
            "original_text": row["text"],
            "start_time": row['start_time'],
            "end_time": row['end_time'],
            "entities": extracted_entities
        }
        transcript_data.append(nlp_data)
    
    full_medication_info = extract_med_admins_with_confidence(transcript_data)

    rows = prepare_medication_rows(full_medication_info)

    med_path = export_to_csv(
        data=rows,
        audio_chunk_path=Path(chunk_path),
        service="medX",
        columns=MED_COLUMNS,
        empty_ok=True,
    )

async def run_medication_extraction(chunk_path: str, transcript_path: str):
    """Async wrapper to run the medication extraction pipeline."""
    extractor = MedicationExtractor()
    medication_extraction_pipeline(chunk_path, transcript_path, extractor)
