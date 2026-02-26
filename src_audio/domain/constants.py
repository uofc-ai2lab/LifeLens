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
    "Normal Saline": {
        "aliases": ["NS", "Electrolyte"],
        "default_dosage": "100 mL"
    },
    "Acetylsalicylic Acid": {
        "aliases": ["ASA", "Aspirin"],
        "default_dosage": "80 mg"
    },
    "Adenosine": {
        "aliases": ["Adeno"],
        "default_dosage": "6 mg /2 mL"
    },
    "Amiodarone": {
        "aliases": ["Amio"],
        "default_dosage": "150 mg / 3 mL"
    },
    "Atropine": {
        "aliases": ["Atropine"],
        "default_dosage": "1 mg / 1 mL"
    },
    "Calcium Chloride": {
        "aliases": ["Calcium", "Cal Chlor"],
        "default_dosage": "1 g /10 mL"
    },
    "Carboprost": {
        "aliases": ["Hemabate"],
        "default_dosage": "250 mcg / 1 mL"
    },
    "Clopidogrel": {
        "aliases": ["Plavix"],
        "default_dosage": "300 mL"
    },
    "Dextrose 5% in Water": {
        "aliases": ["D5W"],
        "default_dosage": "100 ml"
    },
    "Dextrose 50% in Water": {
        "aliases": ["D50"],
        "default_dosage": None
    },
    "Epinephrine": {
        "aliases": ["Epi"],
        "default_dosage": "1 mg / 10 mL"
    },
    "Norepinephrine": {
        "aliases": ["Levo", "Norepi"],
        "default_dosage": "4 mg / 4 mL"
    },
    "Furosemide": {
        "aliases": ["Lasix"],
        "default_dosage": "40 mg / 4 mL"
    },
    "Isoproterenol": {
        "aliases": ["Iso", "Isuprel"],
        "default_dosage": "0.2 mg / 1 mL"
    },
    "Labetalol": {
        "aliases": ["Labetalol", "Trandate"],
        "default_dosage": "100 mg / 20 mL"
    },
    "Metoprolol": {
        "aliases": ["Metoprolol", "Lopressor"],
        "default_dosage": "5 mg / 5 mL"
    },
    "Nitroglycerin": {
        "aliases": ["NTG", "Nitro", "Tridil"],
        "default_dosage": "0.4 mg"
    },
    "Phenylephrine": {
        "aliases": ["Neo", "Phenyl"],
        "default_dosage": "500 mcg / 10 mL"
    },
    "Sodium Bicarbonate": {
        "aliases": ["Bicarb", "Sodium Bicarb", "Bicarbonate"],
        "default_dosage": "50 mEq / 50 mL"
    },
    "Salbutamol": {
        "aliases": ["Albuterol", "Ventolin"],
        "default_dosage": "0.5 mg / mL"
    },
    "Ipratropium": {
        "aliases": ["Atrovent"],
        "default_dosage": "250 mcg / 2 mL NEB"
    },
    "Dimenhydrinate": {
        "aliases": ["Gravol"],
        "default_dosage": "50 mg / 1 mL"
    },
    "Fentanyl": {
        "aliases": ["Fent", "Sublimaze"],
        "default_dosage": "50 mg / 1 mL"
    },
    "Diphenhydramine": {
        "aliases": ["Benadryl"],
        "default_dosage": "50 mg / 1 mL"
    },
    "Metoclopramide": {
        "aliases": ["Reglan", "Maxeran", "Primperan"],
        "default_dosage": "10 mg / 2 mL"
    },
    "Loperamide": {
        "aliases": ["Imodium"],
        "default_dosage": "2 mg"
    },
    "Ondansetron": {
        "aliases": ["Zofran"],
        "default_dosage": "4 mg / 2 mL"
    },
    "Enoxaparin": {
        "aliases": ["Lovenox"],
        "default_dosage": "100 mg / 1 mL"
    },
    "Heparin": {
        "aliases": ["Hep"],
        "default_dosage": "5000 units / 0.5 mL"
    },
    "Humulin R": {
        "aliases": ["Regular insulin", "R-insulin", "Reg", "Reg Insulin"],
        "default_dosage": "1000 units / 10 mL"
    },
    "Magnesium sulfate": {
        "aliases": ["Mag sulfate", "Mag"],
        "default_dosage": "5 g / 10 mL"
    },
        "Potassium Chloride": {
        "aliases": ["KCl"],
        "default_dosage": "10mmol / 100 mL"
    },
    "Vitamin K": {
        "aliases": ["Phytonadione", "Vit K"],
        "default_dosage": "10 mg / 1 mL"
    },
    "Misoprostol": {
        "aliases": ["Miso"],
        "default_dosage": "200 mcg"
    },
    "Oxytocin": {
        "aliases": ["Oxy", "Pitocin"], # Double check these abbreviations
        "default_dosage": "10 units"
    },
    "Propofol": {
        "aliases": ["Prop", "Diprivan"],
        "default_dosage": "100 mL"
    },
    "Lidocaine": {
        "aliases": ["Lido", "Lido oint", "Xylocaine"],
        "default_dosage": "2% x 10 mL"
    },
    "Naloxone": {
        "aliases": ["Narcan"],
        "default_dosage": "4 mg / 10 mL"
    },
    "Intralipid": {
        "aliases": [],
        "default_dosage": None
    },
    "Dexamethasone": {
        "aliases": ["Dex", "Decadron"],
        "default_dosage": "4 mg / 1 mL"
    },
    "Solu-Medrol": {
        "aliases": ["Solu-Med", "Methylpred", "MPS"],
        "default_dosage": "125 mg"
    },
    "Clopidogrel": { 
        "aliases": ["Plavix"],
        "default_dosage": "75 mg"
    },
    "Ticagrelor": {
        "aliases": ["Brilinta"],
        "default_dosage": "90 mg"
    },
    "TNKase": {
        "aliases": ["TNK"],
        "default_dosage": "50 mg"
    },
    "Tranexamic Acid": {
        "aliases": ["TXA"],
        "default_dosage": "1000 mg / 10 mL"
    },
    "Ceftriaxone": {
        "aliases": ["Rocephin", "CRO"],
        "default_dosage": "1 gram"
    },
    "Haloperidol": {
        "aliases": ["Haldol"],
        "default_dosage": "5 mg / 2 mL"
    },
    "Indomethacin": {
        "aliases": ["Indo", "Indocin", "Tivorbex"],
        "default_dosage": "100 mg"
    },
    "Bug Spray": {
        "aliases": [],
        "default_dosage": None
    },
    "Sterile Water": {
        "aliases": [],
        "default_dosage": "10 ml"
    }
}


ROUTES: frozenset[str] = frozenset({
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
HIGH_CONFIDENCE_SCORE: float = 1.0
FUZZY_THRESHOLD: int = 85        
FUZZY_CONF_SCALE: float = 0.85   
NER_CONFIDENCE: float = 0.90      


MED_COLUMNS: list[str] = [
    "start_time",
    "end_time",
    "event_type",
    "medication (confidence score)",
    "dosage (confidence score)",
    "route (confidence score)",
    "full_text"
]

AUDIT_COLUMNS: list[str] = [
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
