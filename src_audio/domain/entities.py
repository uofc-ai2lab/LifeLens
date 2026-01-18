from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class AudioFileMetaData:
    audio_file: Optional[str] = None
    transcript_filename: Optional[Path] = None
    anonymization_filename: Optional[Path] = None
    medication_filename: Optional[Path] = None
    semantic_filename: Optional[Path] = None
    intervention_filename: Optional[Path] = None
    output_file: Optional[Path] = None
    
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
