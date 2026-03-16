import os
import re

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
MEDICATIONS: dict[str, dict] = {
    "Fentanyl": {
        "aliases": ["Fent", "Sublimaze"],
        "default_dosage": "50 mcg"
    },
    "Ondansetron": {
        "aliases": ["Zofran"],
        "default_dosage": "4 mg"
    },
    "Tranexamic Acid": {
        "aliases": ["TXA"],
        "default_dosage": "1 g"
    },
    "Propofol": {
        "aliases": ["Prop", "Diprivan"],
        "default_dosage": "1000 mg"
    },
    "Rocuronium": {
        "aliases": ["Roc", "Zemuron", "Esmeron"],
        "default_dosage": "50 mg"
    },
    "Ketamine": {
        "aliases": ["Ket"],
        "default_dosage": "50 mg"
    },
    "Norepinephrine": {
        "aliases": ["Levophed", "Norepi", "NE"],
        "default_dosage": "4 mg"
    },
    "Phenylephrine": {
        "aliases": ["Neo-Synephrine", "Phenyl"],
        "default_dosage": "100 mcg"
    },
    "Hypertonic Saline 3%": {
        "aliases": ["3% HTS", "3% Saline", "3% NACL"],
        "default_dosage": "250 mLs"
    },
    "Calcium Chloride": {
        "aliases": ["Calcium", "Cal Chlor"],
        "default_dosage": "1 g"
    },
    "Dimenhydrinate": {
        "aliases": ["Gravol", "DIM"],
        "default_dosage": "50 mg"
    },
    "Epinephrine": {
        "aliases": ["Epi"],
        "default_dosage": "1 mg"
    },
    "Normal Saline IV Solution": {
        "aliases": ["NS", "Electrolyte"],
        "default_dosage": "500 mLs"
    },
    "Cefazolin": {
        "aliases": ["Ancef", "Kefzol"],
        "default_dosage": "2 g"
    },
    "Ceftriaxone": {
        "aliases": ["Rocephin", "CRO"],
        "default_dosage": "1 g"
    },
    "Midazolam": {
        "aliases": ["Midaz", "Versed"],
        "default_dosage": "2 mg"
    },
    "Morphine": {
        "aliases": ["Morph", "MS Contin", "MS"],
        "default_dosage": "5 mg"
    },
    "Sodium Bicarbonate": {
        "aliases": ["Bicarb", "Sodium Bicarb", "Bicarbonate"],
        "default_dosage": "50 mEq"
    },
    "Calcium Gluconate": {
        "aliases": ["Ca Gluconate", "Calglucon", "Kalcinate"],
        "default_dosage": "50 mEq"
    },
    "Humulin R": { # Check with Meg if this is same as Insulin Regular (I think it is)
        "aliases": ["Regular insulin", "R-insulin", "Reg", "Reg Insulin"],
        "default_dosage": "10 units"
    },
    "Levetiracetam": {
        "aliases": ["LEV", "Keppra"],
        "default_dosage": "4500 mg"
    },
    "Atropine": {
        "aliases": ["Atro"],
        "default_dosage": "0.5 mg"
    },
    "Lidocaine 2%": { # Check with Meg that dosage shouldn't be 10 mL??
        "aliases": ["Lido 2%", "2% Lido oint", "Xylocaine 2%"],
        "default_dosage": "40 mg"
    },
    "Naloxone": {
        "aliases": ["Narcan"],
        "default_dosage": "0.4 mg"
    },
    "Piperacillin-Tazobactam": {
        "aliases": ["Zosyn", "Pip-Tazo", "PTZ", "PT"],
        "default_dosage": "4.5 g"
    },
    "Potassium Chloride": {
        "aliases": ["KCl"],
        "default_dosage": "10 mEq"
    },
    "Ringer's Lactate IV Solution": {
        "aliases": ["RL", "LR", "Ringers"],
        "default_dosage": "250 mLs"
    },
    "Adenosine": {
        "aliases": ["Adeno", "Adenocard"],
        "default_dosage": "6 mg"
    },
    "Amiodarone": {
        "aliases": ["Amio", "Ammo"],
        "default_dosage": "300 mg"
    },
    "Azithromycin": {
        "aliases": ["Zpack", "Zithromax", "AZM"],
        "default_dosage": "300 mg"
    },
    "Digibind": {
        "aliases": ["Digifab", "Digoxin Immune Fab"],
        "default_dosage": "160 mg"
    },
    "Haloperidol": {
        "aliases": ["Haldol"],
        "default_dosage": "5 mg"
    },
    "Hydromorphone": {
        "aliases": ["Dilaudid"],
        "default_dosage": "5 mg"
    },
    "Lidocaine 5%": { # Check with Meg that dosage shouldn't be 15 g (ointment)??
        "aliases": ["Lidoderm", "Lido 5%", "Xylocaine 5%"],
        "default_dosage": "80 mg"
    },
    "Lorazepam": {
        "aliases": ["LZP", "Ativan"],
        "default_dosage": "1 mg"
    },
    "Magnesium Sulfate": {
        "aliases": ["Mag sulfate", "Mag"],
        "default_dosage": "2 g"
    },
    "Mannitol 20%": {
        "aliases": ["20% Mannitol", "20% Osmitrol", "M20%"],
        "default_dosage": "80 g"
    },
    "Metoclopramide": {
        "aliases": ["Reglan", "Maxolon"],
        "default_dosage": "10 mg"
    },
    "Thiamine": {
        "aliases": ["Thiam", "Thiamin", "Vit B1"],
        "default_dosage": "300 mg"
    },
    "Vancomycin": {
        "aliases": ["Vanco", "Vancocin", "VAN"],
        "default_dosage": "1 g"
    },
}

ALIAS_TO_CANONICAL: dict[str, str] = {} # Maps every known term (canonical + all aliases, lowercased) → canonical lowercase name. Enables O(1) resolution in _resolve_canonical_name.
CANONICAL_TO_DEFAULT_DOSAGE: dict[str, str | None] = {} # Maps canonical lowercase name → default dosage string and Enables O(1) lookup in get_default_dosage.

for _canonical, _info in MEDICATIONS.items():
    _key = _canonical.lower()
    ALIAS_TO_CANONICAL[_key] = _key
    CANONICAL_TO_DEFAULT_DOSAGE[_key] = _info.get("default_dosage")
    for _alias in _info.get("aliases", []):
        ALIAS_TO_CANONICAL[_alias.lower()] = _key

del _canonical, _info, _key, _alias

ROUTES: frozenset[str] = frozenset({
    "infusion", "iv", "intra-venous", "iv push", "ivp",
    "im", "intramuscular",
    "po", "oral",
    "pr", "rectal",
    "subcutaneous", "sc", "sq",
    "sl", "sublingual",
    "io", "intraosseous",
    "neb", "nebulized",
    "inhaled",
    "topical",
})

DOSAGES: frozenset[str] = frozenset({
    "mg", "ml", "g", "mg/ml", "mills", "unit", "units", "mcg",
    "milligrams", "milliliters", "grams", "micrograms",
    "l", "liters", "litres", "litre", "liter",
    "cc", "cc's", "drops", "drop", "tablet", "tablets",
    "puffs", "puff", "spray", "sprays", "inhaler",
    "capsule", "capsules", "pills", "pill",
    "inhalations", "mmol", "millimoles", "mEq", "milliequivalents"
})

TEXT_NUMBERS: dict[str, float] = {
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

NUMBER_PATTERN: re.Pattern = re.compile(r"(?:\d*\.\d+|\d+)(?:/\d+)?")
DOSAGE_TOKEN_PATTERN: re.Pattern = re.compile(r"(?:\d*\.\d+|\d+)(?:/\d+)?|[a-z']+")
LOW_CONFIDENCE_SCORE: float = 0.4
DEFAULT_DOSAGE_SCORE: float = 0.25
HIGH_CONFIDENCE_SCORE: float = 0.88
FUZZY_THRESHOLD: int = 85        
FUZZY_CONF_SCALE: float = 0.85   
NER_CONFIDENCE: float = 0.90   
SENTENCE_END = re.compile(r'[.!?]\s*$')
SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')   

MED_COLUMNS: list[str] = [
    "start_time",
    "end_time",
    "event_type",
    "medication (confidence score)",
    "dosage (confidence score)",
    "route (confidence score)",
]

AUDIT_COLUMNS: list[str] = [
    "audio_chunk_file",
    "start_time",
    "end_time",
    "intent",
    "medication",
    "original_text",
]

PRE_NEGATION_TRIGGERS: frozenset[str] = frozenset({
    "don't", "do not", "never", "without",
    "hold", "withhold", "hold off", "avoid",
    "instead of", "rather than",
    "let's not", "we're not",
})

POST_NEGATION_TRIGGERS: frozenset[str] = frozenset({
    "contraindicated", "not indicated", "withheld", "held",
    "not given", "not administered",
    "hold", "withhold", "avoid", "hold off",
})

REVISED_SIGNALS: frozenset[str] = frozenset({
    "actually", "correction", "change that", "i mean",
    "make it", "scratch that", "no wait", "check that", "wait",
    "update that",
})

ADMINISTERED_SIGNALS: frozenset[str] = frozenset({
    "gave", "given", "pushed", "administered", "bolused",
    "infusing", "infused", "started", "injected", "applied",
    "nebulizing", "hung", "is in", "are in", "went in", "running",
})

ORDERED_SIGNALS: frozenset[str] = frozenset({
    "give", "giving", "push", "pushing", "administer", "administering", "bolus", "start",
    "run", "hang", "draw up", "nebulize", "use",
    "apply", "load", "get me", "let's get",
})

CONSIDERED_SIGNALS: frozenset[str] = frozenset({
    "should we", "should i", "consider", "thinking about",
    "what about", "maybe", "might want to", "could we",
    "would you", "do we want", "prep", "pulling up",
    "drawing up", "getting ready",
})

QUESTIONED_SIGNALS: frozenset[str] = frozenset({
    "did we give", "have we given", "did you give",
    "was given", "did we already", "already gave",
})

### ------------------------------- TRANSCRIPTION SERVICE ------------------------------- ###
MODEL_NAME_OR_PATH = "openai/whisper-large-v3"
TRANSCRIPTION_MODEL_OUTPUT_DIR = "./whisper-large-v3-medical-lora"
DATASET_NAME = "leduckhai/MultiMed-ST"

