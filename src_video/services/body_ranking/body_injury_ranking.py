import csv
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from src_video.domain.entities import create_body_parts
from src_video.domain.constants import BODY_PART_SIDE_SUFFIX_RE, format_sideable_part_label
from config.logger import Logger
import time
import os
from src_video.services.image_anonymization_service.anonymize import run_anonymize_image
from data_transfer.sender_global import get
from config.video_settings import IMAGE_SAVE_DIR

log = Logger("[video][ranking]")

def _normalize_body_part(raw: Any) -> str:
    if raw is None:
        return "unknown"
    label = str(raw).strip().lower()
    if not label:
        return "unknown"

    side_suffix_match = BODY_PART_SIDE_SUFFIX_RE.match(label)
    if side_suffix_match:
        base_part_raw = side_suffix_match.group(1)
        base_part = base_part_raw.replace("_", "").replace("-", "").replace(" ", "")
        side_index = side_suffix_match.group(2)
        return format_sideable_part_label(base_part, side_index)

    return label


def _iter_prediction_rows(predictions_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Prefer per-part aggregated outputs when present (more stable across runs)
    per_part = predictions_data.get("per_part_summary")
    if isinstance(per_part, list) and per_part:
        return [row for row in per_part if isinstance(row, dict)]

    preds = predictions_data.get("predictions")
    if isinstance(preds, list) and preds:
        return [row for row in preds if isinstance(row, dict)]

    return []

def body_ranking(settings: Dict[str, Any]) -> bool:
    """
    Track best injury predictions per body part across multiple captures.
    Saves first detection timestamp (pred_time) and never overwrites it.
    Outputs JSON and CSV.
    """

    classification_output = Path(settings["CLASSIFICATION_OUTPUT"])
    # Read the same JSON path the pipeline is configured to write (when provided),
    # otherwise default to ClassificationOutput/injury_predictions.json.
    prediction_json = Path(settings.get("INJURY_REPORT_JSON", classification_output / "injury_predictions.json"))
    template_json = classification_output / "visual_output.json"
    output_csv = classification_output / "visual_output.csv"

    try:
        if not prediction_json.exists():
            log.warning(f"Prediction file not found: {prediction_json}")
            return False

        # Load predictions
        with open(prediction_json, "r") as f:
            predictions_data = json.load(f)

        # Load or create template
        if template_json.exists():
            with open(template_json, "r") as f:
                best_results = json.load(f)
        else:
            log.info(f"Creating new visual_output file: {template_json}")
            best_results = create_body_parts()

        updated_count = 0

        # Process predictions
        rows = _iter_prediction_rows(predictions_data)
        if not rows:
            log.warning("No predictions found in injury_predictions.json (expected 'predictions' or 'per_part_summary')")
            rows = []

        for pred in rows:

            body_part_raw = pred.get("body_part")
            body_part = _normalize_body_part(body_part_raw)

            injury_pred = pred.get("injury_pred")
            if not injury_pred:
                continue
            injury_prob = pred.get("injury_prob")
            if injury_prob is None:
                injury_prob = pred.get("avg_prob")
            if injury_prob is None:
                injury_prob = pred.get("accuracy", 0.0)
            image_id = pred.get("image_id")

            now = time.strftime("%Y-%m-%d %H:%M:%S")

            def _update_part(part_key: str) -> None:
                nonlocal updated_count

                if part_key not in best_results:
                    best_results[part_key] = {"injuries": {}}
                if "injuries" not in best_results[part_key]:
                    best_results[part_key]["injuries"] = {}

                injuries = best_results[part_key]["injuries"]

                if injury_pred not in injuries:
                    injuries[injury_pred] = {
                        "image_id": image_id,
                        "accuracy": float(injury_prob),
                        "pred_time": now,
                    }
                    updated_count += 1

                    # Send image to server
                    _send_images_to_server(Path(IMAGE_SAVE_DIR / f"{image_id}.jpg"), image_id)
                    return

                if float(injury_prob) > float(injuries[injury_pred].get("accuracy", 0.0)):
                    injuries[injury_pred]["image_id"] = image_id
                    injuries[injury_pred]["accuracy"] = float(injury_prob)
                    updated_count += 1

                    # Send image to server
                    _send_images_to_server(Path(IMAGE_SAVE_DIR / f"{image_id}.jpg"), image_id)


            # Update normalized key
            _update_part(body_part)



        # Save JSON
        with open(template_json, "w", encoding="utf-8") as f:
            json.dump(best_results, f, indent=2)
        
        # Send JSON to server
        data_sender = get()
        data_sender.send_batch(
            pipeline="video",
            files=[(str(template_json), "visual")])

        # Save CSV
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "body_part",
                "image_id",
                "injury_pred",
                "accuracy",
                "pred_time"
            ])

            for body_part, part_data in best_results.items():

                injuries = part_data.get("injuries", {})
                first = True

                for injury, data in injuries.items():

                    writer.writerow([
                        body_part if first else "",
                        data.get("image_id", ""),
                        injury,
                        data.get("accuracy", ""),
                        data.get("pred_time", "")
                    ])

                    first = False

        log.success(f"Body ranking complete. Updated {updated_count} entries")
        log.info(f"JSON output: {template_json}")
        log.info(f"CSV output: {output_csv}")

        return True

    except Exception as e:
        log.error(f"Body ranking failed: {e}")
        return False
    
# Helper to do entire image sending process
def _send_images_to_server(image_path: Path, image_id: Any) -> None:
    if not os.path.exists(image_path):
        log.error(f"Image file not found for sending: {image_path}")
        return
    
    try:
        image_bytes = run_anonymize_image(image_path)
    except Exception as e:
        log.error(f"Failed to anonymize image: {e}")
        log.info("Falling back to original image.")

        try:
            image_bytes = image_path.read_bytes()
        except Exception as e:
            log.error(f"Failed to read original image: {e}")
            return

    try:
        data_sender = get()
        data_sender.send_image_bytes(
            pipeline="video",
            files=[(image_bytes, f"{image_id}", "image")]
        )
    except Exception as e:
        log.error(f"Failed to send image for ranking: {e}")