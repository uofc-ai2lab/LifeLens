from presidio_anonymizer import DeanonymizeEngine
from presidio_anonymizer.entities import OperatorResult, OperatorConfig
import json


# Initialize the engine:
engine = DeanonymizeEngine()

crypto_key = "WmZq4t7w!z%C&F)J"
encrypted_text = ''
with open('Presidio_Simple_Output.txt', 'r') as file:
    for line in file:
        encrypted_text += line.strip()

with open("Presidio_Anonymized_Items.json", "r") as f:
    results_as_dicts = json.load(f)

operator_results = [OperatorResult(**item) for item in results_as_dicts]

# Invoke the deanonymize function with the text, anonymizer results and
# Operators to define the deanonymization type.
result = engine.deanonymize(
    text=encrypted_text,
    entities=operator_results,
    operators={"DEFAULT": OperatorConfig("decrypt", {"key": crypto_key})},
)

print(result)