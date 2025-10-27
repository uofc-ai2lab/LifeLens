# Supported Entities -> https://microsoft.github.io/presidio/supported_entities/
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

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
print("\n--- Anonymized Text ---")
# The final output: 'Hello, my name is <NAME>, and my phone number is 212-***-****. <REDACTED>.'
print(anonymized_result.text)