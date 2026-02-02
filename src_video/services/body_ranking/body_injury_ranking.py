import json
import csv
from pathlib import Path
from typing import List, Optional, Dict, Any
from src_video.domain.entities import create_body_parts


def body_ranking(settings: Dict[str, Any]) -> bool:
    """
    Track best injury predictions per body part across multiple captures.
    Creates and updates visual_output.json and visual_output.csv with 
    the highest confidence prediction for each body part.
    
    Returns:
        True if successful, False otherwise.
    """

    
    classification_output = Path(settings["CLASSIFICATION_OUTPUT"])
    prediction_json = classification_output / "injury_predictions.json"
    template_json = classification_output / "visual_output.json"
    output_csv = classification_output / "visual_output.csv"
    
    
    try:
        if not prediction_json.exists():
            print(f"[WARNING] Prediction file not found: {prediction_json}")
            return False
        
        with open(prediction_json, 'r') as f:
            predictions_data = json.load(f)
        
        if template_json.exists():
            with open(template_json, 'r') as f:
                best_results = json.load(f)
        else:
            print(f"[INFO] Creating new visual_output file: {template_json}")
            best_results = create_body_parts()
        
        # Update best results with new predictions
        updated_count = 0
        for pred in predictions_data.get("predictions", []):
            body_part = pred.get("body_part")
            injury_pred = pred.get("injury_pred")
            injury_prob = pred.get("injury_prob", 0.0)
            image_id = pred.get("image_id")
            
            # If body part isn't apart of the template we just add
            if body_part not in best_results:
                best_results[body_part] = {
                    "injuries": {}
                }

            
            # body part has an injuries dict for multiple injuries per part
            if "injuries" not in best_results[body_part]:
                best_results[body_part]["injuries"] = {}

            injuries = best_results[body_part]["injuries"]

            # If new injury OR better accuracy
            if injury_pred not in injuries or injury_prob > injuries[injury_pred]["accuracy"]:
                injuries[injury_pred] = {
                    "image_id": image_id,
                    "accuracy": injury_prob
                }

                updated_count += 1
                print(f"[UPDATE] {body_part} - {injury_pred}: {injury_prob:.3f}")

                    
        with open(template_json, 'w') as f:
            json.dump(best_results, f, indent=2)
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["body_part", "image_id", "injury_pred", "accuracy"])
            
            for body_part, part_data in best_results.items():
                
                first = True
                for injury, data in injuries.items():
                    writer.writerow([
                        body_part if first else "", 
                        data.get("image_id", ""),
                        injury,
                        data.get("accuracy", "")
                    ])
                    first = False
        
        print(f"\n[INFO] Body ranking complete. Updated {updated_count} body parts.")
        print(f"[INFO] JSON output: {template_json}")
        print(f"[INFO] CSV output: {output_csv}\n")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Body ranking failed: {e}")
        return False