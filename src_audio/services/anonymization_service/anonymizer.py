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
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.entity_operators = self._create_anonymized_entity_operators()
    
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
            "DATE_TIME": OperatorConfig("replace", {"new_value": "<ANON>"}),
            "PERSON": OperatorConfig("replace", {"new_value": "<ANON>"}),
            "AGE": OperatorConfig("replace", {"new_value": "<ANON>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<ANON>"}),
            "GPE": OperatorConfig("replace", {"new_value": "<ANON>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<ANON>"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<ANON>"}),
            "ORGANIZATION": OperatorConfig("replace", {"new_value": "<ANON>"}),
            "EMS_UNIT": OperatorConfig("replace", {"new_value": "<ANON>"}),
            "CASE_NUMBER": OperatorConfig("replace", {"new_value": "<ANON>"}),
            "DEFAULT": OperatorConfig("replace", {"new_value": "<ANON>"})
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