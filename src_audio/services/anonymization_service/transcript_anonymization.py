from src_audio.utils.load_csv_file import load_csv_file 
from src_audio.utils.export_to_csv import export_to_csv
from src_audio.services.anonymization_service.anonymizer import TranscriptAnonymizer
from config.audio_settings import TRANSCRIPT_FILES_LIST
from pathlib import Path

def run_anonymization(transcript_path: str, anonymizer: TranscriptAnonymizer) -> None:
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
    df = load_csv_file(transcript_path)
    anonymized_texts = []
    for _, row in df.iterrows():
        anonymized_text = anonymizer.anonymize(row["text"])
        anonymized_texts.append({
            "start_time": row['start_time'],
            "end_time": row['end_time'],
            "text": anonymized_text
        })

    # Export anonymized transcript to CSV
    export_to_csv(
        data=anonymized_texts,
        output_path=Path(transcript_path).parent,
        input_file_path=Path(transcript_path),
        service="anonymization",
        columns=["start_time", "end_time", "text"],
        empty_ok=True, # should not be empty unless incoming csv transcript had empty cells which I don't think is possible, so can I set this to False?
    )

async def run_anonymization_service():
    """
    Async wrapper to run the transcript anonymization script
    
    Args: 
        None
    
    Returns: 
        None
    """
    anonymizer = TranscriptAnonymizer()
    for transcript in TRANSCRIPT_FILES_LIST:
        run_anonymization(transcript, anonymizer)
    