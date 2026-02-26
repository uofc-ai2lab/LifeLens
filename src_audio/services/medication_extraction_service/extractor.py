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

    def extract_medication_info_from_ner(self, text: str) -> list[MedicationEntity]:
        """
        Extract medication-related entities using the Med7 spaCy pipeline.

        Returns DRUG, DOSAGE, STRENGTH, and ROUTE entities as
        MedicationEntity objects (word, start_idx, score), sorted by start_idx.
        All NER entities receive a fixed confidence score.

        Args:
            text (str): Transcript segment text.

        Returns:
            list[MedicationEntity]: Extracted entities (may be empty).
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text)

        entities: list[MedicationEntity] = []
        for ent in doc.ents:
            label = self.map_to_dosage.get(ent.label_, ent.label_)  # remap STRENGTH to DOSAGE, keep others as is
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

        # spaCy returns ents in document order, but sort explicitly
        # to guarantee the contract for all downstream consumers
        entities.sort(key=lambda e: e.start_idx)
        return entities