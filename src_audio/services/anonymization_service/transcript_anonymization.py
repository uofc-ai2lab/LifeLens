from src_audio.utils.load_csv_file import load_csv_file 
from src_audio.utils.export_to_csv import export_to_csv
from src_audio.services.anonymization_service.anonymizer import TranscriptAnonymizer
from config.logger import Logger
from pathlib import Path

log = Logger("[audio][anonymization]")

def run_anonymization(chunk_path: str, transcript_path: str) -> None:
    """
    Run the full transcript anonymization pipeline:
    - Load transcript CSV
    - Anonymize text using Presidio

    Args:
        transcript_path (str): Path to transcript CSV file.
        extractor (MedicationExtractor): NER model instance.

    Returns:
        None
    """
    log.header("Starting Anonymization...")
    anonymizer = TranscriptAnonymizer()
    df = load_csv_file(transcript_path)
    log.info(f"Processing {len(df)} segments")
    
    anonymized_texts = []
    for _, row in df.iterrows():
        anonymized_text = anonymizer.anonymize(row["text"])
        anonymized_texts.append({
            "start_time": row['start_time'],
            "end_time": row['end_time'],
            "text": anonymized_text,
            "speaker": row.get("speaker", "UNKNOWN")  # Retain speaker info if available
        })

    # Export anonymized transcript to CSV
    anon_path = export_to_csv(
        data=anonymized_texts,
        audio_chunk_path=Path(chunk_path),
        service="anonymization",
        columns=["start_time", "end_time", "text", "speaker"],
        empty_ok=True,
    )
    log.info(f"Saved anonymization file to {anon_path.name if anon_path else 'file'}")
    log.success("Anonymization completed successfully!")