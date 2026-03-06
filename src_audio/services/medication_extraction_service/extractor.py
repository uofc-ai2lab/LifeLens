import spacy
import torch
import os
from src_audio.domain.entities import MedicationEntity
from src_audio.domain.constants import NER_CONFIDENCE
from config.logger import Logger

os.environ["CUDA_MODULE_LOADING"] = "LAZY" # Optimizes CUDA kernel loading to save ~600MB-1GB of system RAM

log = Logger("[audio][medication]")

def _gpu_is_functional() -> bool:
    """Robust check for functional CUDA support."""
    try:
        # Check if torch can actually talk to the Orin's iGPU
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False

class MedicationExtractor:
    allowed_entities: frozenset[str] = frozenset({"DRUG", "DOSAGE", "ROUTE"}) # we ignore all other entities med7 gives us (FORM, FREQUENCY, DURATION)
    map_to_dosage: dict[str, str] = {"STRENGTH": "DOSAGE"} # med7 puts numeric dose values in STRENGTH, but we want them under DOSAGE 
    _MODEL_LG  = "en_core_med7_lg"
    onGPU: bool = False
        
    def __init__(self):
        log.info("Initializing Med7 NER (LG model)")
        
        using_gpu = _gpu_is_functional()        
        if using_gpu:
            try:
                spacy.require_gpu() # Called before spacy.load to prevent memory double-allocation
                log.info(f"GPU detected — Loading {self._MODEL_LG} on CUDA")
                self.nlp = spacy.load(self._MODEL_LG)
                log.success("Model loaded successfully on GPU")
                self.onGPU = True
            except Exception as e:
                log.error(f"Failed to load on GPU: {e}. Falling back to CPU.")
                using_gpu = False
                self.nlp = spacy.load(self._MODEL_LG)
        else:
            log.info(f"Using CPU mode for {self._MODEL_LG}")
            self.nlp = spacy.load(self._MODEL_LG)
            
        # Optimization: GPU can handle slightly larger batches for CNN models
        self.pipe_batch_size = 32 if self.onGPU else 16
        log.success(f"Med7 NER ready (Mode: {'GPU' if self.onGPU else 'CPU'})")

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