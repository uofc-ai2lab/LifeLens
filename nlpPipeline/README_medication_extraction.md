# Medication Extraction Script

## Overview

The `medication_extraction.py` script is designed to extract medication-related information from diarized transcript CSV files. It leverages a pre-trained Named Entity Recognition (NER) model to identify key entities such as medications, dosages, and routes of administration from text segments in emergency medical transcripts. The script processes the input CSV, applies NER, and uses fallback mechanisms to ensure comprehensive extraction, then outputs a new CSV file with structured medication administration records including confidence scores.

## Key Features

- **NER-Based Extraction**: Utilizes the `d4data/biomedical-ner-all` model from Hugging Face Transformers to detect biomedical entities like medications, dosages, and administration routes.
- **Fallback Mechanisms**: Employs regex-based pattern matching and predefined lists (from `medication_extraction_constants.py`) to capture information that might be missed by the NER model.
- **Confidence Scoring**: Assigns confidence scores to extracted entities, with lower scores for fallback extractions to indicate reliability.
- **CSV Processing**: Reads diarized transcript CSVs (expected columns: `text`, `start`, `end`) and outputs a structured CSV with medication details.
- **Comprehensive Coverage**: Handles medication aliases, text numbers (e.g., "five" as 5), and various dosage formats.

## Dependencies

The script requires the following Python packages (install via `pip install -r requirements.txt` from the project root):

- `torch`
- `transformers`
- `pandas`
- `argparse` (standard library)
- `csv` (standard library)
- `re` (standard library)
- `os` (standard library)

Additionally, it imports constants from `medication_extraction_constants.py`, which must be in the same directory.

## Input Format

The input must be a CSV file containing diarized transcript data with at least the following columns:
- `text`: The transcribed text segment.
- `start`: Start time of the segment (e.g., in seconds or timestamp format).
- `end`: End time of the segment.

Example CSV structure:
```
text,start,end
"Patient received 5 mg of morphine IV.",0.0,5.5
"Nurse administered epinephrine subcutaneously.",5.5,10.2
```

## Output Format

The script generates a new CSV file named `<input_filename>_medications_output.csv` in the same directory as the input file. The output includes:

- **TIME**: Time range of the segment (e.g., "0.0 - 5.5").
- **MEDICATION (CONFIDENCE SCORE)**: Extracted medication name with confidence score (e.g., "morphine (0.987)").
- **DOSAGE (CONFIDENCE SCORE)**: Dosage amount and unit with score (e.g., "5 mg (0.945)").
- **ROUTE (CONFIDENCE SCORE)**: Route of administration with score (e.g., "IV (0.892)").

If an entity is not found, it displays "Not Found".

## How to Run

1. Ensure all dependencies are installed and the script has access to `medication_extraction_constants.py`.
2. Prepare your transcript CSV file in the required format.
3. Run the script from the command line:

   ```bash
   python medication_extraction.py /path/to/your/transcript.csv
   ```

   Replace `/path/to/your/transcript.csv` with the actual path to your input CSV file.

4. The output CSV will be created in the same directory as the input file.

## Important Notes

- **Model Loading**: The NER model is loaded from Hugging Face on first run, which may take time and require internet access.
- **Performance**: Processing large CSVs may be resource-intensive due to model inference.
- **Customization**: Medication lists, routes, and dosages are defined in `medication_extraction_constants.py`. Modify this file to adapt to specific vocabularies.
- **Confidence Scores**: Scores range from 0 to 1. Higher scores indicate greater confidence from the NER model. Fallback extractions use predefined low-confidence scores.
- **Error Handling**: The script raises exceptions for file not found or loading errors, which will terminate execution.
- **Future Enhancements**: The code includes comments for potential fuzzy matching improvements.

## Example Usage

Assuming you have a transcript file `emergency_call.csv` in the `output/` directory:

```bash
python nlpPipeline/medication_extraction.py output/emergency_call.csv
```

This will produce `output/emergency_call_medications_output.csv` with the extracted medication information.