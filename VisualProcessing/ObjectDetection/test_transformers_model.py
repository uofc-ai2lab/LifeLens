import sys
from pathlib import Path

from PIL import Image
from transformers import pipeline, AutoModel

# Simple experimental script to try HF pipeline usage shown on the model page.
# NOTE: This model repo is YOLO-based; classification pipeline may not yield meaningful part segmentation
# and is only used here as a connectivity / load test demonstration.

MODEL_ID = "MnLgt/yolo-human-parse"


def main():
    # Load high-level pipeline (image-classification). If the repo is not configured
    # for this task, this may raise or produce placeholder outputs.
    try:
        pipe = pipeline("image-classification", model=MODEL_ID)
        # Use a sample image from dataset if provided as arg else a hub example
        if len(sys.argv) > 1:
            img_path = Path(sys.argv[1])
            img = Image.open(img_path).convert("RGB")
            outputs = pipe(img)
        else:
            sample_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png"
            outputs = pipe(sample_url)
        print("Pipeline outputs:", outputs)
    except Exception as e:
        print("Pipeline load failed:", e)

    # Direct model load (may not have task head configured for classification)
    try:
        model = AutoModel.from_pretrained(MODEL_ID, dtype="auto")
        print("AutoModel loaded successfully.")
    except Exception as e:
        print("AutoModel load failed:", e)


if __name__ == "__main__":
    main()
