from presidio_anonymizer import DeanonymizeEngine
from presidio_anonymizer.entities import OperatorResult, OperatorConfig
from presidio_analyzer import AnalyzerEngine


# Initialize the engine:
engine = DeanonymizeEngine()
analyzer = AnalyzerEngine()

crypto_key = "WmZq4t7w!z%C&F)J"
encrypted_text = ''
with open('Presidio_Simple_Output.txt', 'r') as file:
    for line in file:
        encrypted_text += line.strip()


analyzer_results = analyzer.analyze(
    text=encrypted_text,
    language="en",
)

print("--- Analysis Results ---")
for result in analyzer_results:
    print(f"Found {result.entity_type} at [{result.start}-{result.end}] with score {result.score:.2f}")
# # Invoke the deanonymize function with the text, anonymizer results and
# # Operators to define the deanonymization type.
# result = engine.deanonymize(
#     text="My name is S184CMt9Drj7QaKQ21JTrpYzghnboTF9pn/neN8JME0=",
#     entities=[
#         OperatorResult(start=11, end=55, entity_type="PERSON"),
#     ],
#     operators={"DEFAULT": OperatorConfig("decrypt", {"key": crypto_key})},
# )

# print(result)