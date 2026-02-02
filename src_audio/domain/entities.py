from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class AudioFileMetaData:
    parent_audio_path: Optional[Path] = None
    chunk_audio_path: Optional[Path] = None
    transcript_path: Optional[Path] = None
    denoised_audio_path: Optional[Path] = None
    anonymization_path: Optional[Path] = None
    medication_path: Optional[Path] = None
    semantic_path: Optional[Path] = None
    intervention_path: Optional[Path] = None
    output_path: Optional[Path] = None
    
AUDIO_PIPELINE_METADATA: list[AudioFileMetaData] = []

@dataclass
class MedicationEntity:
    entity: str
    word: str
    start_idx: int
    score: float

@dataclass
class MedicationAdministration:
    medication: str
    medication_score: float
    dosage: Optional[str]
    dosage_score: Optional[float]
    route: Optional[str]
    route_score: Optional[float]
    start_time: float
    end_time: float

@dataclass
class ClinicalIntervention:
    start_time: float
    end_time: float
    intervention: str
