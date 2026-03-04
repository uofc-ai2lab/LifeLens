import spacy
import torch
from src_audio.domain.entities import MedicationEntity
from src_audio.domain.constants import NER_CONFIDENCE
from config.logger import Logger

log = Logger("[audio][medication]")

class MedicationExtractor:
    allowed_entities: frozenset[str] = frozenset({"DRUG", "DOSAGE", "ROUTE"}) # we ignore all other entities med7 gives us (FORM, FREQUENCY, DURATION)
    map_to_dosage: dict[str, str] = {"STRENGTH": "DOSAGE"} # med7 puts numeric dose values in STRENGTH, but we want them under DOSAGE 
    _MODEL_TRF = "en_core_med7_trf"
    _MODEL_LG  = "en_core_med7_lg"

    def _gpu_is_functional() -> bool:
        """
        Check if GPU is available and can run a simple operation. This is more 
        robust than just torch.cuda.is_available(), which can return True even 
        if the GPU isn't working properly.

        Args:
            None
        Returns:
            bool: True if GPU is functional, False otherwise
        """
        try:
            torch.zeros(1).cuda()
            return True
        except Exception:
            return False
        
    def __init__(self):
        log.info("Loading Med7 NER model for medication extraction")
        using_gpu = self._gpu_is_functional()
        if using_gpu:
            log.info("GPU detected — loading transformer model (en_core_med7_trf)")
            self.nlp = spacy.load(self._MODEL_TRF)
            spacy.prefer_gpu()   # tells spacy-transformers to use CUDA
        else:
            log.info("GPU check failed — loading CNN model (en_core_med7_lg)")
            self.nlp = spacy.load(self._MODEL_LG)
        self.model_name = self._MODEL_TRF if using_gpu else self._MODEL_LG
        self.pipe_batch_size = 16 if using_gpu else 32
        log.success(f"Med7 NER model ready: {self.model_name}")

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