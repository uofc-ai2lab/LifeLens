from pathlib import Path
import hashlib
import sys

import db

THIS_DIR = Path(__file__).parent
TEST_DATA_INPUT = THIS_DIR / "test_data" / "input"
TEST_DATA_OUTPUT = THIS_DIR / "test_data" / "output"


def save_all_input_images():
    TEST_DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
    saved = []
    for p in TEST_DATA_INPUT.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            image_key = db.generate_image_key()
            image_id = p.name
            with open(p, "rb") as f:
                image_bytes = f.read()
            sha256 = hashlib.sha256(image_bytes).hexdigest()
            row_id = db.insert_source_image(image_key=image_key, image_id=image_id, image_bytes=image_bytes, sha256=sha256)
            saved.append((image_key, image_id, row_id))
            print(f"Saved {p.name} -> key={image_key} id={row_id}")
    return saved


def fetch_and_write(saved_items):
    for key, original_name, _ in saved_items:
        out_name = f"{key}_{original_name}"
        out_path = TEST_DATA_OUTPUT / out_name
        db.write_image_to_file(key, out_path)
        print(f"Wrote back to {out_path}")


def main():
    print("Saving images from:", TEST_DATA_INPUT)
    saved = save_all_input_images()
    if not saved:
        print("No images found in input folder.")
        return
    print("Fetching saved images and writing to output folder:", TEST_DATA_OUTPUT)
    fetch_and_write(saved)
    print("Done.")


if __name__ == "__main__":
    main()
