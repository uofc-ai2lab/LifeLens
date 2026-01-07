from pathlib import Path
import sys
from config.settings import GENAI_CLIENT, GENAI_MODEL, MEANING_DIR, TRANSCRIPT_FILES_LIST
from google import genai
from google.genai import types
import json, time
from typing import List
import asyncio
from src.utils.load_csv_file import load_csv_as_rows
from src.utils.export_to_csv import export_to_csv
from src.entities import ClinicalIntervention
from src.services.semantic_filtering_service.llm_prompt import llm_prompt

async def extract_valid_interventions(transcript_rows: list[dict]) -> List[ClinicalIntervention]:
    """
    Use Gemini to filter transcript rows into valid clinical interventions.

    Args:
        transcript_rows (list[dict]): Rows from CSV transcript.

    Returns:
        List[ClinicalIntervention]: Filtered clinical actions.
    """
    try:
        response = GENAI_CLIENT.models.generate_content(
            model=GENAI_MODEL, 
            contents="review the provided transcript",
            config=types.GenerateContentConfig(
                system_instruction=llm_prompt(transcript_rows),
                response_mime_type="application/json",
                response_schema=list[ClinicalIntervention],
                # max_output_tokens= 400,
                temperature= 0.2,
            ),
        )

        total_tokens = getattr(response, "total_tokens", "N/A")
        print(f"\nTotal tokens used: {response.usage_metadata.total_token_count}")

        if response.text is None:
            print("Gemini returned no text. Check the request or transcript input.")
            return []

        return json.loads(response.text)
        
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            print(f"ERROR: 429 RESOURCE_EXHAUSTED\nYou exceeded your current quota, please check your plan and billing details.\nFor more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits.")
        else:
            print(f"Error calling Gemini: {e}")
        
        print(f"Since calling Gemini has proposed an error, we will be using mock data to contibue the program.")
        
        mock_data = [
            {"start_time": "00:01", "end_time": "00:05", "intervention": "CPR"},
            {"start_time": "00:06", "end_time": "00:10", "intervention": "Administer Morphine"},
        ]
        return mock_data

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

        interventions = await extract_valid_interventions(transcript_rows)
        csv_rows = prepare_intervention_rows(interventions)

        export_to_csv(
            data=csv_rows,
            output_path=MEANING_DIR,
            input_filename=Path(input_transcript).stem,
            service="semantic",
            columns=["start_time", "end_time", "intervention"],
            empty_ok=True
        )
        
        total_time = time.time() - time_start
        print(f"\nTotal time of function: {total_time:.3f} seconds for file: {Path(input_transcript).stem}")