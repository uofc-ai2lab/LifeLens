import spacy
from src_audio.domain.entities import MedicationEntity
from src_audio.domain.constants import NER_CONFIDENCE
from config.logger import Logger

log = Logger("[audio][medication]")

class MedicationExtractor:
    allowed_entities: frozenset[str] = frozenset({"DRUG", "DOSAGE", "ROUTE"}) # we ignore all other entities med7 gives us (FORM, FREQUENCY, DURATION)
    map_to_dosage: dict[str, str] = {"STRENGTH": "DOSAGE"} # med7 puts numeric dose values in STRENGTH, but we want them under DOSAGE 

    def __init__(self):
        log.info("Loading Med7 NER model for medication extraction")
        self.nlp = spacy.load("en_core_med7_lg")
        log.success("Med7 NER model ready")

    def extract_entities_from_doc(self, doc) -> list[MedicationEntity]:
        """
        Extract medication entities from an already-processed spaCy Doc.

        Separating entity extraction from NLP processing allows the caller
        to use nlp.pipe() for batched inference and then call this method
        per doc, rather than calling nlp() once per segment.

        Args:
            doc (spacy.tokens.Doc): A Doc produced by this extractor's nlp
                pipeline (i.e. en_core_med7_lg).

        Returns:
            list[MedicationEntity]: DRUG, DOSAGE, and ROUTE entities sorted
                by start_idx. Empty list if no relevant entities are found.
        """
        entities: list[MedicationEntity] = []
        for ent in doc.ents:
            label = self.map_to_dosage.get(ent.label_, ent.label_)
            if label not in self.allowed_entities:
                continue
            entities.append(
                MedicationEntity(
                    entity=label,
                    word=ent.text,
                    start_idx=ent.start_char,
                    score=NER_CONFIDENCE,
                )
            )
        entities.sort(key=lambda e: e.start_idx)
        return entities

    def extract_medication_info_from_ner(self, text: str) -> list[MedicationEntity]:
        """
        Extract medication entities from a raw text string.

        Convenience wrapper for single-text callers. For processing many
        segments, prefer batching texts through nlp.pipe() and calling
        extract_entities_from_doc() on each resulting Doc.

        Args:
            text (str): Transcript segment text.

        Returns:
            list[MedicationEntity]: Extracted entities (may be empty).
        """
        if not text or not text.strip():
            return []
        return self.extract_entities_from_doc(self.nlp(text))