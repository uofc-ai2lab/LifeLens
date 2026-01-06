"""Pipeline configuration loader.

Reads configuration from environment variables (typically injected by VS Code via
`python.envFile` pointing at `.env.template`).

Kept separate so `Main/main_pipeline.py` stays focused on orchestration.
"""

from __future__ import annotations

import os
from typing import Optional


def read_boolean_from_environment(variable_name: str, default_value: bool) -> bool:
    string_value = os.getenv(variable_name)
    if string_value is None:
        return default_value

    normalized_value = string_value.strip().lower()
    if normalized_value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized_value in {"0", "false", "f", "no", "n", "off"}:
        return False

    return default_value


def read_integer_from_environment(variable_name: str, default_value: int) -> int:
    string_value = os.getenv(variable_name)
    if string_value is None or string_value.strip() == "":
        return default_value

    try:
        return int(string_value)
    except ValueError:
        return default_value


def read_float_from_environment(variable_name: str, default_value: float) -> float:
    string_value = os.getenv(variable_name)
    if string_value is None or string_value.strip() == "":
        return default_value

    try:
        return float(string_value)
    except ValueError:
        return default_value


def read_string_from_environment(variable_name: str, default_value: str) -> str:
    string_value = os.getenv(variable_name)
    return default_value if string_value is None else string_value


def read_optional_string_from_environment(
    variable_name: str,
    default_value: Optional[str] = None,
) -> Optional[str]:
    string_value = os.getenv(variable_name)
    if string_value is None:
        return default_value

    stripped_value = string_value.strip()
    return stripped_value if stripped_value else None


def read_list_from_environment(variable_name: str, default_value: list[str]) -> list[str]:
    string_value = os.getenv(variable_name)
    if string_value is None or string_value.strip() == "":
        return default_value

    # Comma-separated list, whitespace-tolerant
    split_items = [item.strip() for item in string_value.split(",")]
    return [item for item in split_items if item]


def load_pipeline_config() -> dict:
    """Load pipeline configuration from environment variables.

    Defaults match the prior constants in `Main/main_pipeline.py`.
    """

    pipeline_root = read_string_from_environment("PIPELINE_ROOT", "Main/PipelineOutputs")
    detection_output = read_string_from_environment(
        "PIPELINE_DETECTION_OUTPUT",
        f"{pipeline_root}/DetectionOutput",
    )
    classification_output = read_string_from_environment(
        "PIPELINE_CLASSIFICATION_OUTPUT",
        f"{pipeline_root}/ClassificationOutput",
    )
    classification_export = read_string_from_environment(
        "PIPELINE_CLASSIFICATION_EXPORT",
        f"{classification_output}/parts_dataset",
    )

    return {
        # Source images directory
        "DETECTION_SOURCE": read_string_from_environment(
            "PIPELINE_DETECTION_SOURCE",
            "VisualProcessing/ObjectDetection/Priv_personpart/ImageSamples",
        ),
        # Central pipeline output root
        "PIPELINE_ROOT": pipeline_root,
        # Subfolders for detection and classification stage outputs
        "DETECTION_OUTPUT": detection_output,
        "CLASSIFICATION_OUTPUT": classification_output,
        "CLASSIFICATION_EXPORT": classification_export,
        # If false, detection will also export an ImageFolder layout under CLASSIFICATION_EXPORT.
        # (Not needed for injury inference; kept for debugging / compatibility.)
        "USE_DETECTION_CROPS_FOR_TRAINING": read_boolean_from_environment(
            "PIPELINE_USE_DETECTION_CROPS_FOR_TRAINING",
            True,
        ),
        # Detection parameters
        "DETECTION_MODEL": read_string_from_environment(
            "PIPELINE_DETECTION_MODEL",
            "MnLgt/yolo-human-parse",
        ),
        "MAX_IMAGES": read_integer_from_environment("PIPELINE_MAX_IMAGES", 200),
        "ADD_HEAD": read_boolean_from_environment("PIPELINE_ADD_HEAD", True),
        "ALPHA_PNG": read_boolean_from_environment("PIPELINE_ALPHA_PNG", False),
        "MIN_AREA": read_integer_from_environment("PIPELINE_MIN_AREA", 250),
        "MARGIN": read_float_from_environment("PIPELINE_MARGIN", 0.10),
        "CLASSES": read_list_from_environment(
            "PIPELINE_CLASSES",
            ["face", "arm", "hand", "leg", "foot", "neck", "torso", "head"],
        ),
        "DEVICE": read_optional_string_from_environment("PIPELINE_DEVICE", None),
        "DEBUG": read_boolean_from_environment("PIPELINE_DEBUG", False),
        # Injury classification inference parameters
        "INJURY_CHECKPOINT_PATH": read_string_from_environment(
            "PIPELINE_INJURY_CHECKPOINT",
            "experiments/checkpoints/simple/best_swin_tiny_patch4_window7_224.pt",
        ),
        "INJURY_IMG_SIZE": read_integer_from_environment("PIPELINE_INJURY_IMG_SIZE", 224),
        "INJURY_BATCH_SIZE": read_integer_from_environment("PIPELINE_INJURY_BATCH_SIZE", 32),
        "INJURY_NUM_WORKERS": read_integer_from_environment("PIPELINE_INJURY_NUM_WORKERS", 0),
        "INJURY_REPORT_JSON": read_string_from_environment(
            "PIPELINE_INJURY_REPORT_JSON",
            f"{classification_output}/injury_predictions.json",
        ),
        "INJURY_REPORT_CSV": read_string_from_environment(
            "PIPELINE_INJURY_REPORT_CSV",
            f"{classification_output}/injury_predictions_summary.csv",
        ),

        # Optional classification training parameters (used by training utilities; the default
        # pipeline entrypoint does not train today).
        "CLS_EPOCHS": read_integer_from_environment("PIPELINE_CLS_EPOCHS", 5),
        "CLS_BATCH_SIZE": read_integer_from_environment("PIPELINE_CLS_BATCH_SIZE", 32),
        "CLS_IMG_SIZE": read_integer_from_environment("PIPELINE_CLS_IMG_SIZE", 224),
        "CLS_VAL_RATIO": read_float_from_environment("PIPELINE_CLS_VAL_RATIO", 0.2),
        "CLS_TEST_RATIO": read_float_from_environment("PIPELINE_CLS_TEST_RATIO", 0.0),
        "CLS_LR": read_float_from_environment("PIPELINE_CLS_LR", 3e-4),
        "CLS_SPLIT_SEED": read_integer_from_environment("PIPELINE_CLS_SPLIT_SEED", 42),
        "CLS_FREEZE_BACKBONE": read_boolean_from_environment("PIPELINE_CLS_FREEZE_BACKBONE", False),
        "CLS_FREEZE_BACKBONE_EPOCHS": read_integer_from_environment(
            "PIPELINE_CLS_FREEZE_BACKBONE_EPOCHS",
            0,
        ),
        "CLS_BACKBONE_LR_MULT": read_float_from_environment("PIPELINE_CLS_BACKBONE_LR_MULT", 0.1),
        "CLS_NUM_WORKERS": read_integer_from_environment("PIPELINE_CLS_NUM_WORKERS", 0),
        "CLS_MAKE_CONFUSION_MATRICES": read_boolean_from_environment(
            "PIPELINE_CLS_MAKE_CONFUSION_MATRICES",
            True,
        ),
        "CLS_SAVE_ROOT": read_string_from_environment(
            "PIPELINE_CLS_SAVE_ROOT",
            "experiments/checkpoints/parts_from_detection",
        ),
    }
