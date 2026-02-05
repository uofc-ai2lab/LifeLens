import os
import re

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac"}


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
    "Pressure Infuser": [],
    "Normal Saline": ["NS", "Electrolyte"],
    "Syringe": [],
    "ASA": ["ASA"],
    "Adenosine": ["Adeno"],
    "Amiodarone": ["Amio"],
    "Atropine": ["Atropine"],
    "Calcium Chloride": ["CaCl₂"],
    "Carboprost": ["Hemabate"],
    "Clopidogrel": ["Plavix"],
    "D5W": ["D5W"],
    "Dextrose": ["D50"],
    "Epinephrine": ["Epi"],
    "Furosemide": ["Lasix"],
    "Isoproterenol": ["Iso", "Isuprel"],
    "Labetalol": ["Labetalol"],
    "Metoprolol": ["Metoprolol", "Lopressor"],
    "Nitroglycerin": ["NTG"],
    "Norepinephrine": ["Levo", "NE"],
    "Phenylephrine": ["Neo", "Phenyl"],
    "Sodium Bicarbonate": ["Bicarb"],
    "Salbutamol": ["Salbutamol", "Ventolin"],
    "Ipratropium": ["Ipratropium", "Atrovent"],
    "Dimenhydrinate": ["Gravol"],
    "Diphenhydramine": ["Benadryl"],
    "Metoclopramide": ["Maxeran", "Reglan"],
    "Loperamide": ["Imodium"],
    "Ondansetron": ["Zofran"],
    "Enoxaparin": ["Lovenox"],
    "Heparin": ["Heparin"],
    "Humulin R": ["Regular insulin", "R-insulin"],
    "Magnesium sulfate": ["Mag sulfate", "MgSO₄"],
    "Vitamin K": ["Phytonadione"],
    "Misoprostol": ["Miso"],
    "Oxytocin": ["Oxy", "Pitocin"],
    "Propofol": ["Propofol", "Diprivan"],
    "Lidocaine": ["Lido", "Lido oint"],
    "Naloxone": ["Narcan"],
    "Intralipid": ["Intralipid"],
    "Dexamethasone": ["Dex", "Decadron"],
    "Solu-Medrol": ["Solu-Medrol", "Methylpred"],
    "Plavix": ["Plavix"],
    "Ticagrelor": ["Brilinta"],
    "TNKase": ["TNK"],
    "Tranexamic Acid": ["TXA"],
    "Indomethacin": ["Indomethacin", "Indocin"],
    "Bug Spray": [],
    "NS flush": ["NS flush"],
    "Sterile Water": ["SW"]
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

NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?(?:/\d+)?")

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

### ------------------------------- AUDIO RECORDING SERVICE ------------------------------- ###
MAX_RECORD_SECONDS = 300  # 5 minutes
RECORDING_DIR = "/home/capstone/recordings"
SIGNAL_FILE = os.path.join(RECORDING_DIR, "recording_done.flag")
ARECORD_DEVICE = "hw:CARD=ArrayUAC10,DEV=0"
CHUNK_SECONDS = 20

### ------------------------------- GSTREAMER AUDIO CONFIGURATION ------------------------------- ###
# GStreamer audio pipeline settings
AUDIO_SAMPLE_RATE = 16000  # Hz
AUDIO_CHANNELS = 6         # Multi-channel from ArrayUAC10
AUDIO_FORMAT = "S16LE"     # Signed 16-bit Little Endian

