from dataclasses import dataclass
from typing import Optional
from pathlib import Path

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
