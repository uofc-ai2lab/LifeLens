from pathlib import Path
import sys
from config.settings import GENAI_MODEL, MEANING_DIR, TRANSCRIPT_FILES_LIST
import google.generativeai as genai
import json, time
from typing import List
from src.utils.load_csv_file import load_csv_as_rows
from src.utils.export_to_csv import export_to_csv
from src.entities import ClinicalIntervention
from src.services.semantic_filtering_service.llm_prompt import llm_prompt

def extract_valid_interventions(transcript_rows: list[dict]) -> List[ClinicalIntervention]:
    """
    Use Gemini to filter transcript rows into valid clinical interventions.

    Args:
        transcript_rows (list[dict]): Rows from CSV transcript.

    Returns:
        List[ClinicalIntervention]: Filtered clinical actions.
    """
    try:
        result = GENAI_MODEL.generate_content(
            llm_prompt(transcript_rows),
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=list[ClinicalIntervention]
            )
        )
        return json.loads(result.text)
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return []

def prepare_intervention_rows(interventions: List[ClinicalIntervention]):
    
    return [
        {
            "start": i["start_time"],
            "end": i["end_time"],
            "intervention": i["intervention"]
        }
        for i in interventions
    ]
    
async def run_semantic_filtering():
    """
    Run semantic filtering on extracted interventions CSV.
    """
    if GENAI_MODEL is None:
        print("GENAI_MODEL does not exist, is semantic filtering enabled in your environment?")
        sys.exit(0)
        
    for input_transcript in TRANSCRIPT_FILES_LIST:
        time_start = time.time()
        
        transcript_rows = load_csv_as_rows(input_transcript)
        if not transcript_rows:
            print("No transcript rows found.")
            return

        interventions = extract_valid_interventions(transcript_rows)
        csv_rows = prepare_intervention_rows(interventions)

        export_to_csv(
            data=csv_rows,
            output_path=MEANING_DIR,
            input_filename=Path(input_transcript).stem,
            service="semantic",
            columns=["start", "end", "intervention"],
            empty_ok=True
        )
        
        total_time = time.time() - time_start
        print(f"\nTotal time of function: {total_time} for file: {Path(input_transcript).stem}")