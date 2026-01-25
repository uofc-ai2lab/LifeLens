import json
import shutil
from pathlib import Path
from config.video_settings import IMG_CLASSIFICATION_DIR, IMAGE_SAVE_DIR, VIDEO_OUTPUT

def categorize_img():
    prediction_path = Path(IMG_CLASSIFICATION_DIR) / "injury_predictions.json"

    save_dir = Path(IMAGE_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    preds = json.load(open(prediction_path))
    known_bp = json.load(open(VIDEO_OUTPUT))

    # best prediction per body part
    best = {}
    
    for prediction in preds["predictions"]:
        body_part = prediction["body_part"]
        if body_part not in best or prediction["injury_prob"] > best[body_part]["injury_prob"]:
            best[body_part] = prediction
        

    for body_part, entry in known_bp.items():
        if body_part in best:
            p = best[body_part]
            src = Path(p["crop_path"])
            dst = save_dir / src.name
            shutil.copy(src, dst) # copies the image file

            entry["image_id"] = p["image_id"]
            entry["injury_pred"] = p["injury_pred"]

    json.dump(known_bp, open(VIDEO_OUTPUT, "w"), indent=2)
