# Supported Entities -> https://microsoft.github.io/presidio/supported_entities/
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import json

# --- 1. Setup the Engines ---
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# --- 2. Input Text ---
text = ''
with open('inputExample1.txt', 'r') as file:
    for line in file:
        text += line.strip()

# --- 3. Analyze (Detection) ---
# The Analyzer returns a list of detected entities, their locations, and a confidence score.
analyzer_results = analyzer.analyze(
    text=text,
    language="en",
    # Optionally, specify which entities to look for
    entities=["PERSON"]
)

# Optional: Print what was found
print("--- Analysis Results ---")
for result in analyzer_results:
    print(f"Found {result.entity_type} at [{result.start}-{result.end}] with score {result.score:.2f}")

# --- 4. Anonymize (Encryption) ---
# Anonymize the text using the analysis results.
crypto_key = "WmZq4t7w!z%C&F)J"
anonymized_result = anonymizer.anonymize(
    text=text,
    analyzer_results=analyzer_results,
    operators={
        "PERSON": OperatorConfig("encrypt", {"key": crypto_key}),
        "DEFAULT": OperatorConfig("redact"),
    }
)

# --- 5. Output ---
# Save anonymized items to a json file (needed for decryption -> this tells us where
# in the transcript the anonymized texts are in the code)
# I personally do not like this as that means the transcript outputted must be the exact same
# one read in since we're going by char position
results_as_dicts = [item.to_dict() for item in anonymized_result.items] # converts items in analyzer object into dicts so we can save into a json file 
print(results_as_dicts)
with open("Presidio_Anonymized_Items.json", 'w') as f:
    json.dump(results_as_dicts, f, indent=4) # indent=4 for pretty-printing

# Saving the anonymized text to a file output
with open('Presidio_Simple_Output.txt', 'w') as file:
    file.write(anonymized_result.text)