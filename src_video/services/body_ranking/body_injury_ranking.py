import json
import csv
from pathlib import Path
from typing import List, Optional, Dict, Any
from src_video.domain.entities import create_body_parts
import time




def body_ranking(settings: Dict[str, Any]) -> bool:
    """
    Track best injury predictions per body part across multiple captures.
    Saves first detection timestamp (pred_time) and never overwrites it.
    Outputs JSON and CSV.
    """

    classification_output = Path(settings["CLASSIFICATION_OUTPUT"])
    prediction_json = classification_output / "injury_predictions.json"
    template_json = classification_output / "visual_output.json"
    output_csv = classification_output / "visual_output.csv"

    try:
        if not prediction_json.exists():
            print(f"[WARNING] Prediction file not found: {prediction_json}")
            return False

        # Load predictions
        with open(prediction_json, "r") as f:
            predictions_data = json.load(f)

        # Load or create template
        if template_json.exists():
            with open(template_json, "r") as f:
                best_results = json.load(f)
        else:
            print(f"[INFO] Creating new visual_output file: {template_json}")
            best_results = create_body_parts()

        updated_count = 0

        # Process predictions
        for pred in predictions_data.get("predictions", []):

            body_part = pred.get("body_part")
            injury_pred = pred.get("injury_pred")
            injury_prob = pred.get("injury_prob", 0.0)
            image_id = pred.get("image_id")

            now = time.time()

            # Ensure body part exists
            if body_part not in best_results:
                best_results[body_part] = {"injuries": {}}

            if "injuries" not in best_results[body_part]:
                best_results[body_part]["injuries"] = {}

            injuries = best_results[body_part]["injuries"]

            # First detection
            if injury_pred not in injuries:

                injuries[injury_pred] = {
                    "image_id": image_id,
                    "accuracy": injury_prob,
                    "pred_time": now
                }

                updated_count += 1

            # Better accuracy (keep old timestamp)
            elif injury_prob > injuries[injury_pred]["accuracy"]:

                injuries[injury_pred]["image_id"] = image_id
                injuries[injury_pred]["accuracy"] = injury_prob

                updated_count += 1

        # Save JSON
        with open(template_json, "w") as f:
            json.dump(best_results, f, indent=2)

        # Save CSV
        with open(output_csv, "w", newline="") as f:
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

        print(f"\n[INFO] Body ranking complete. Updated {updated_count} entries.")
        print(f"[INFO] JSON output: {template_json}")
        print(f"[INFO] CSV output: {output_csv}\n")

        return True

    except Exception as e:
        print(f"[ERROR] Body ranking failed: {e}")
        return False