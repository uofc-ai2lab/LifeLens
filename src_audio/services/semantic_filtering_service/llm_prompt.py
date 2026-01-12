import json 

def llm_prompt(input_data):
    return f"""
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
    3. **RETURN JSON WITH**:
        - start_time
        - end_time
        - intervention
       
    ### INPUT DATA:
    {json.dumps(input_data, indent=2)}
    """