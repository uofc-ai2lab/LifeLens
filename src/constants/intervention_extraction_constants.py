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