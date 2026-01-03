import google.generativeai as genai
import typing_extensions as typing
import pandas as pd
import csv, os, json
from dotenv import load_dotenv
import time

# 1. Setup
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables.")
genai.configure(api_key=api_key)

# Start time
time_start = time.time()
time_end = time_start

# 2. Define the Output Schema
# This tells Gemini: "I only want these three fields, nothing else."
class ClinicalAction(typing.TypedDict):
    start_time: str
    end_time: str
    intervention: str  # The specific medical action (e.g., "Administered Morphine")

# 3. The Filtering Function
def extract_valid_interventions(transcript_data):
    model = genai.GenerativeModel("gemini-flash-latest")

    # We convert the dictionary to a string so Gemini can read it
    data_str = json.dumps(transcript_data, indent=2)
    
    prompt = f"""
    You are an expert Clinical Scribe. Your goal is to review the following raw transcript data (derived from a CSV) and extract ONLY valid clinical interventions or medical commands.
    
    ### RULES for Filtering:
    1. **INCLUDE (Crucial) **: 
       - Medical procedures (CPR, Intubation, Needle Decompression).
       - Medications (Morphine, Epinephrine, Fluids, Pneumothorax).
       - Assessment Vitals (BP, Pulse, Pupil checks).
       - Equipment usage (Defibrillator, Suction).
    2. **EXCLUDE / IGNORE** (Crucial): 
       - Interpersonal chatter (e.g., "See you partner", "Good job man", "Okay").
       - Administrative/Instructional talk (e.g., "Dispatch, we are en route", "This is a scenario").
       - Meta-commentary about simulation equipment (e.g., "The mannequin is expensive").
       - Vague or incomplete statements.
       
    ### INPUT DATA:
    {data_str}
    """
    # Call the API with Strict Structured Output
    try:
        result = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=list[ClinicalAction]
            )
        )
        return json.loads(result.text)
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return []


# 4. Main Execution Block
async def run_semantic_filtering(output_path="./output/cleaned_clinical_events.json"):

    file_path = "./output/interventions_extracted.csv"
    file_name = "interventions_extracted.csv"
    transcript = {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                transcript[file_name] = rows
            else:
                print(f"Warning: CSV file '{file_name}' is empty or has no rows.")

        # Call the function
        if transcript:
            print("Processing transcript with Gemini...")
            cleaned_data = extract_valid_interventions(transcript)

            # Output the results
            print(f"\nFound {len(cleaned_data)} valid interventions:")
            print("-" * 60)
            print(f"{'Start':<12} | {'End':<12} | {'Intervention'}")
            print("-" * 60)
            for item in cleaned_data:
                print(f"{item['start_time']:<12} | {item['end_time']:<12} | {item['intervention']}")
                
            # Optional: Save back to JSON
            with open(output_path, "w") as out_f:
                json.dump(cleaned_data, out_f, indent=2)
                print("\nSaved cleaned data to 'cleaned_clinical_events.json'")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the path.")
        

    # Print the result
    print(f"{'Start':<12} | {'End':<12} | {'Intervention'}")
    print("-" * 50)
    for item in cleaned_data:
        print(f"{item['start_time']:<12} | {item['end_time']:<12} | {item['intervention']}")
    time_end = time.time()
    total_time = time_end - time_start
    print(f"\nTotal time of function: {total_time}")