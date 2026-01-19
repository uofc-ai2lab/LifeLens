from presidio_analyzer import (
    AnalyzerEngine,
    PatternRecognizer,
    Pattern
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import hmac
import hashlib
import base64
import os

class TranscriptAnonymizer:
    """
    A class for anonymizing sensitive information in transcripts using Presidio.
    
    This class handles the detection and replacement of personally identifiable information (PII)
    such as names, dates, ages, locations, and custom entities like EMS units and case numbers.
    """
    def __init__(self):
        """
        Initializes the TranscriptAnonymizer with necessary engines and operators.
        
        Sets up the analyzer and anonymizer engines, loads the secret key for pseudonymization,
        and creates the entity operators for anonymization.
        
        Raises:
            RuntimeError: If the PHI_PSEUDONYM_KEY environment variable is not set.
        """
        self.secret_key = os.getenv("PHI_PSEUDONYM_KEY")
        if not self.secret_key:
            raise RuntimeError("PHI_PSEUDONYM_KEY environment variable not set.") # putting this here for now as a 'safeguard' but need to have response behavior if this ever fails -> anonmyzation won't run but then we gotta be extra secure w/ the transript
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.entity_operators = self._create_anonymized_entity_operators()
    
    # creates anonymized identifier
    def _pseudonymize(self, value: str, entity: str) -> str:
        """
        Creates a pseudonymized identifier for a given value and entity type.
        
        Uses HMAC-SHA256 with a secret key to generate a consistent, anonymized token
        for the input value, prefixed with the entity type.
        
        Args:
            value (str): The original value to pseudonymize.
            entity (str): The entity type (e.g., "PERSON").
        
        Returns:
            str: The pseudonymized identifier in the format "<ENTITY_TOKEN>".
        """
        digest = hmac.new(
            self.secret_key.encode(),
            f"{entity}:{value}".encode(),
            hashlib.sha256
        ).digest()
        token = base64.urlsafe_b64encode(digest)[:10].decode()
        return f"<{entity}_{token}>"

    def _age_anonymizer(self, age_str: str) -> str:
        """
        Anonymizes age information by categorizing or masking the value.
        
        Ages over 89 are replaced with "<AGE_90+>", others with "<AGE>".
        Invalid inputs are also masked as "<AGE>".
        
        Args:
            age_str (str): The age string to anonymize.
        
        Returns:
            str: The anonymized age representation.
        """
        try:
            age = int(age_str)
            if age > 89:
                return "<AGE_90+>"
            else:
                return "<AGE>"
        except (ValueError, TypeError):
            return "<AGE>"

    def _create_anonymized_entity_operators(self):
        """
        Creates and configures custom recognizers and entity operators for anonymization.
        
        Adds custom pattern recognizers for EMS units and case numbers, and defines
        operator configurations for various entity types to replace or pseudonymize them.
        
        Returns:
            dict: A dictionary mapping entity types to their OperatorConfig.
        """
        #custom trauma specific recognizer (may add more/remove these ones)
        ems_unit_recognizer = PatternRecognizer(
            supported_entity="EMS_UNIT",
            patterns=[
                Pattern(name="ems_unit", regex=r"\bEMS[-\s]?\d{2,5}\b", score=0.8)
            ]
        )

        case_number_recognizer = PatternRecognizer(
            supported_entity="CASE_NUMBER",
            patterns=[
                Pattern(name="case_num", regex=r"\b(Case|Run)\s?#?\d+\b", score=0.8)
            ]
        )

        self.analyzer.registry.add_recognizer(ems_unit_recognizer)
        self.analyzer.registry.add_recognizer(case_number_recognizer)

        # define entities to be anonymized (these come from Presidio)
        ENTITY_OPERATORS = {
            "PERSON": OperatorConfig("custom", {
                "lambda": lambda x: self._pseudonymize(x, "PERSON")
            }),
            "DATE_TIME": OperatorConfig("replace", {
                "new_value": "<DATE>"
            }),
            "AGE": OperatorConfig("custom", {
                "lambda": lambda x: self._age_anonymizer(x)
            }),
            "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
            "GPE": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            "ORGANIZATION": OperatorConfig("replace", {"new_value": "<ORG>"}),
            "EMS_UNIT": OperatorConfig("replace", {"new_value": "<EMS_UNIT>"}),
            "CASE_NUMBER": OperatorConfig("replace", {"new_value": "<CASE_ID>"}),
            "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"})
        }

        return ENTITY_OPERATORS

    def anonymize(self, text: str, language: str = "en") -> str:
        """
        Anonymizes sensitive information in the provided text.
        
        Analyzes the text for specified entities and applies the configured operators
        to replace or pseudonymize them.
        
        Args:
            text (str): The input text to anonymize.
            language (str, optional): The language code for analysis (default: "en").
        
        Returns:
            str: The anonymized text with sensitive information replaced.
        """
        results = self.analyzer.analyze(text=text, language=language, entities=list(self.entity_operators.keys()))
        anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=results, operators=self.entity_operators)
        return anonymized_result.text