import spacy
import threading
from src_audio.domain.entities import MedicationEntity
from src_audio.domain.constants import NER_CONFIDENCE
from config.logger import Logger

log = Logger("[audio][medication]")

_EXTRACTOR_SINGLETON = None
_EXTRACTOR_LOCK = threading.Lock()


class MedicationExtractor:
    # we ignore all other entities med7 gives us (FORM, FREQUENCY, DURATION)
    allowed_entities: frozenset[str] = frozenset({"DRUG", "DOSAGE", "ROUTE"})
    # med7 puts numeric dose values in STRENGTH, but we want them under DOSAGE
    map_to_dosage: dict[str, str] = {"STRENGTH": "DOSAGE"}
    _MODEL_LG = "en_core_med7_lg"

    def __init__(self):
        log.info("Initializing Med7 NER (LG model)")

        try:
            self.nlp = spacy.load(self._MODEL_LG)
        except Exception as e:
            log.error(
                f"Failed to load model. Please ensure you have the '{self._MODEL_LG}' model installed. Error details: {e}")
            exit(1)

        self.pipe_batch_size = 32
        log.success(f"Med7 NER ready")

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


def get_medication_extractor() -> MedicationExtractor:
    """Return a process-wide Med7 extractor singleton."""
    global _EXTRACTOR_SINGLETON
    if _EXTRACTOR_SINGLETON is not None:
        return _EXTRACTOR_SINGLETON

    with _EXTRACTOR_LOCK:
        if _EXTRACTOR_SINGLETON is None:
            _EXTRACTOR_SINGLETON = MedicationExtractor()
    return _EXTRACTOR_SINGLETON
