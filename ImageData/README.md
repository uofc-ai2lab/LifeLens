Data layout

- Put all your images under `data/images/` (you can also add subfolders; paths in CSV can be relative to this root or absolute).
- Create two CSV files: `data/train.csv` and `data/val.csv` with the following columns:
  - `image_path`: Path to the image file (relative like `images/foo.jpg` or absolute `C:/.../foo.jpg`).
  - `labels`: Semicolon-separated class names (e.g., `Bruise;Laceration`). Use exact class spellings from `experiments/config.yaml`.

Example rows:

image_path,labels
images/001.jpg,Bruise
images/002.jpg,Laceration;Bruise
images/003.jpg,Normal skin

images sources:
https://www.kaggle.com/datasets/yasinpratomo/wound-dataset?resource=download

If you're not sure how to split your 400 images, you can use the helper script `scripts/make_csv_from_folder.py` to auto-split and generate CSVs.
