from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from src.entities import MedicationEntity
from src.utils.calculate_mean import mean

class MedicationExtractor:
    def __init__(self):
        self.NER_MODEL = "d4data/biomedical-ner-all" # pretrained biomedical NER model
        self.model = AutoModelForTokenClassification.from_pretrained(self.NER_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.NER_MODEL, use_fast=True)
        self.model.eval()

    def _run_ner(self, text):
        """
        Run token-level Named Entity Recognition (NER) on input text.

        Tokenizes the text, runs the NER model, and returns:
        - label probability distributions per token
        - character offset mappings for each token
        - word IDs mapping tokens back to original words

        Args:
            text (str): Raw input text (e.g., "Give fentanyl IV now").

        Returns:
            probs (torch.Tensor): Token-level label probability distributions.
            offsets (torch.Tensor):Character start/end positions for each token.
            word_ids (List[Optional[int]]):Token-to-word alignment indices.
        """
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True
        )
        offsets = enc.pop("offset_mapping")[0] # offsets map each token back to the original string:
        word_ids = enc.word_ids() # multiple tokens can belong to the same word -> given the same word_id

        with torch.no_grad():
            logits = self.model(**enc).logits[0] # each token has a score for every label
            probs = torch.softmax(logits, dim=-1)

        return probs, offsets, word_ids

    def _group_tokens(self, probs, offsets, word_ids):
        """
        Group token-level NER predictions into word-level entity candidates.
        Combines subword tokens (e.g., "fen", "##tan", "##yl") that belong to the same word into a single entity span. 

        Args:
            probs (torch.Tensor): Token-level label probability distributions.
            offsets (torch.Tensor):Character start/end positions for each token.
            word_ids (List[Optional[int]]):Token-to-word alignment indices.

        Returns:
            Dict[int, Dict]:
                Mapping of word_id to grouped entity information:
                {
                    word_id: {
                        "entity": str,        # predicted entity label
                        "start": int,         # start char index
                        "end": int,           # end char index
                        "scores": List[float] # confidence scores per token
                    }
                }
        """
        
        word_groups = {}

        for i, word_id in enumerate(word_ids):
            if word_id is None:
                continue

            label_id = probs[i].argmax().item()
            label = self.model.config.id2label[label_id]
            score = probs[i, label_id].item()

            if label == "O":
                continue

            start, end = offsets[i].tolist()
            
            # First token of a word → create group
            # Subsequent tokens → update same group
            group = word_groups.setdefault(word_id, {
                "entity": label,
                "start_idx": start,
                "end_idx": end,
                "scores": []
            })
            group["end"] = end
            group["scores"].append(score)
        
        return word_groups

    def extract_medication_info_from_ner(self, text):
        """
        Extract finalized medication-related entities from input text. 
        Converts grouped tokens into final usable entities with averaged
        confidence scores.

        Args:
            text (str): Raw clinical or conversational text.

        Returns:
            List[MedicationEntity]:
                List of extracted entities containing: entity label, text span, start character index, averaged confidence score
        """
        probs, offsets, word_ids = self._run_ner(text)
        word_groups = self._group_tokens(probs, offsets, word_ids)

        entities = []
        for w in sorted(word_groups):
            item = word_groups[w]
            span = text[item["start_idx"]: item["end_idx"]]
            entities.append(
                MedicationEntity(
                    entity=item["entity"],
                    word=span,
                    start_idx=item["start_idx"], 
                    score=mean(item["scores"])
                ))
            
        return entities
