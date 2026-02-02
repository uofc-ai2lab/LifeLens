import os
import re
import threading

### ------------------------------- AUDIO RECORDING SERVICE ------------------------------- ###
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac"}

MAX_RECORD_SECONDS = 300  # 5 minutes
RECORDING_DIR = "/home/capstone/recordings"
SIGNAL_FILE = os.path.join(RECORDING_DIR, "recording_done.flag")

# GStreamer Audio Pipeline Configuration
ARECORD_DEVICE = "hw:CARD=ArrayUAC10,DEV=0"
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 6
AUDIO_FORMAT = "S16LE"
CHUNK_SECONDS=20

### ------------------------------- INTERVENTION SERVICE ------------------------------- ###
INTERVENTIONS = {
    "CPR": ["cardiopulmonary resuscitation", "cpr", "chest compressions"],
    "Airway Management": ["airway", "intubation", "intubated", "endotracheal", "et tube"],
    "Needle Decompression": ["pneumothorax", "pneumo", "needle decompression", "thoracostomy", "chest decompression", "breathing thoracostomy"],
    "Spinal Immobilization": ["spinal motion restriction", "smr", "cervical collar", "c-collar", "backboard"],
    "Hemorrhage Control": ["pressure", "tourniquet", "bleeding control", "bandage", "dressing"],
    "IV/Fluid Administration": ["iv", "intravenous", "fluid", "saline", "line started", "io", "intraosseous"],
    "Ventilator": ["ventilator", "mechanical ventilation", "vent"]
}

REPLACEMENTS = {
    "pneumo.": "pneumothorax",
    "pneumo ": "pneumothorax ",
    "c collar": "cervical collar",
}

INTER_COLUMNS = [
    "start_time", 
    "end_time",
    "event_type", 
    "event_category",
    "entity_detected", 
    "full_text"
]

### ------------------------------- MEDICATION SERVICE ------------------------------- ###
MEDICATIONS = {
    "Pressure Infuser": {
        "aliases": [],
        "default_dosage": None
    },
    "Normal Saline": {
        "aliases": ["NS", "Electrolyte"],
        "default_dosage": "1000 ml"
    },
    "Syringe": {
        "aliases": [],
        "default_dosage": None
    },
    "ASA": {
        "aliases": ["ASA"],
        "default_dosage": "325 mg"
    },
    "Adenosine": {
        "aliases": ["Adeno"],
        "default_dosage": "6 mg"
    },
    "Amiodarone": {
        "aliases": ["Amio"],
        "default_dosage": "300 mg"
    },
    "Atropine": {
        "aliases": ["Atropine"],
        "default_dosage": "1 mg"
    },
    "Calcium Chloride": {
        "aliases": ["CaCl₂"],
        "default_dosage": "1 g"
    },
    "Carboprost": {
        "aliases": ["Hemabate"],
        "default_dosage": "250 mcg"
    },
    "Clopidogrel": {
        "aliases": ["Plavix"],
        "default_dosage": "300 mg"
    },
    "D5W": {
        "aliases": ["D5W"],
        "default_dosage": "500 ml"
    },
    "Dextrose": {
        "aliases": ["D50"],
        "default_dosage": "25 g"
    },
    "Epinephrine": {
        "aliases": ["Epi"],
        "default_dosage": "1 mg"
    },
    "Furosemide": {
        "aliases": ["Lasix"],
        "default_dosage": "40 mg"
    },
    "Isoproterenol": {
        "aliases": ["Iso", "Isuprel"],
        "default_dosage": "2 mcg/min"
    },
    "Labetalol": {
        "aliases": ["Labetalol"],
        "default_dosage": "20 mg"
    },
    "Metoprolol": {
        "aliases": ["Metoprolol", "Lopressor"],
        "default_dosage": "5 mg"
    },
    "Nitroglycerin": {
        "aliases": ["NTG"],
        "default_dosage": "0.4 mg"
    },
    "Norepinephrine": {
        "aliases": ["Levo", "NE"],
        "default_dosage": "0.1 mcg/kg/min"
    },
    "Phenylephrine": {
        "aliases": ["Neo", "Phenyl"],
        "default_dosage": "100 mcg"
    },
    "Sodium Bicarbonate": {
        "aliases": ["Bicarb"],
        "default_dosage": "50 mEq"
    },
    "Salbutamol": {
        "aliases": ["Salbutamol", "Ventolin"],
        "default_dosage": "2.5 mg"
    },
    "Ipratropium": {
        "aliases": ["Ipratropium", "Atrovent"],
        "default_dosage": "0.5 mg"
    },
    "Dimenhydrinate": {
        "aliases": ["Gravol"],
        "default_dosage": "50 mg"
    },
    "Diphenhydramine": {
        "aliases": ["Benadryl"],
        "default_dosage": "25 mg"
    },
    "Metoclopramide": {
        "aliases": ["Maxeran", "Reglan"],
        "default_dosage": "10 mg"
    },
    "Loperamide": {
        "aliases": ["Imodium"],
        "default_dosage": "4 mg"
    },
    "Ondansetron": {
        "aliases": ["Zofran"],
        "default_dosage": "4 mg"
    },
    "Enoxaparin": {
        "aliases": ["Lovenox"],
        "default_dosage": "40 mg"
    },
    "Heparin": {
        "aliases": ["Heparin"],
        "default_dosage": "5000 units"
    },
    "Humulin R": {
        "aliases": ["Regular insulin", "R-insulin"],
        "default_dosage": "10 units"
    },
    "Magnesium sulfate": {
        "aliases": ["Mag sulfate", "MgSO₄"],
        "default_dosage": "2 g"
    },
    "Vitamin K": {
        "aliases": ["Phytonadione"],
        "default_dosage": "10 mg"
    },
    "Misoprostol": {
        "aliases": ["Miso"],
        "default_dosage": "800 mcg"
    },
    "Oxytocin": {
        "aliases": ["Oxy", "Pitocin"],
        "default_dosage": "10 units"
    },
    "Propofol": {
        "aliases": ["Propofol", "Diprivan"],
        "default_dosage": "50 mg"
    },
    "Lidocaine": {
        "aliases": ["Lido", "Lido oint"],
        "default_dosage": "100 mg"
    },
    "Naloxone": {
        "aliases": ["Narcan"],
        "default_dosage": "0.4 mg"
    },
    "Intralipid": {
        "aliases": ["Intralipid"],
        "default_dosage": "100 ml"
    },
    "Dexamethasone": {
        "aliases": ["Dex", "Decadron"],
        "default_dosage": "10 mg"
    },
    "Solu-Medrol": {
        "aliases": ["Solu-Medrol", "Methylpred"],
        "default_dosage": "125 mg"
    },
    "Plavix": {
        "aliases": ["Plavix"],
        "default_dosage": "300 mg"
    },
    "Ticagrelor": {
        "aliases": ["Brilinta"],
        "default_dosage": "180 mg"
    },
    "TNKase": {
        "aliases": ["TNK"],
        "default_dosage": "50 mg"
    },
    "Tranexamic Acid": {
        "aliases": ["TXA"],
        "default_dosage": "1 g"
    },
    "Indomethacin": {
        "aliases": ["Indomethacin", "Indocin"],
        "default_dosage": "50 mg"
    },
    "Bug Spray": {
        "aliases": [],
        "default_dosage": None
    },
    "NS flush": {
        "aliases": ["NS flush"],
        "default_dosage": "10 ml"
    },
    "Sterile Water": {
        "aliases": ["SW"],
        "default_dosage": "10 ml"
    }
}


ROUTES = {
    "infusion", "iv", "intra-venous", "iv push", "ivp", "iv bolus", "ivb", "bolus",
    "im", "intramuscular",
    "po", "oral",
    "pr", "rectal",
    "subcutaneous", "sc", "sq",
    "sl", "sublingual",
    "io", "intraosseous",
    "neb", "nebulized",
    "inhaled",
    "topical",
}

# still need to implement logic
TIMES = {"hours", "seconds", "minutes"}

DOSAGES = {d.lower() for d in {
    "mg", "ml", "g", "mg/ml", "mills", "unit", "units", "mcg",
    "milligrams", "milliliters", "grams", "micrograms",
    "l", "liters", "litres", "litre", "liter",
    "cc", "cc's", "drops", "drop", "tablet", "tablets",
    "puffs", "puff", "spray", "sprays", "inhaler",
    "capsule", "capsules", "pills", "pill",
    "inhalations", "mmol", "millimoles",
    "micron", "iu", "iu's", "international units"
}}

TEXT_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "half": 0.5,
    "quarter": 0.25
}

NUMBER_PATTERN = re.compile(r"(?:\d*\.\d+|\d+)(?:/\d+)?")
DOSAGE_TOKEN_PATTERN = re.compile(r"(?:\d*\.\d+|\d+)(?:/\d+)?|[a-z']+")
LOW_CONFIDENCE_SCORE = 0.5
HIGH_CONFIDENCE_SCORE = 1.0

MED_COLUMNS = [
    "start_time",
    "end_time",
    "event_type",
    "medication (confidence score)",
    "dosage (confidence score)",
    "route (confidence score)",
    "full_text"
]
