# Object Detection (Human Body Part Parsing)

Purpose: Provide body-part localized crops (e.g., arm, hand, leg, foot, face) as focused inputs to the wound classification models.

Model Choice: Using a YOLO segmentation model fine-tuned for human parts (`MnLgt/yolo-human-parse`). This returns bounding boxes and segmentation masks for classes:
`[hair, face, neck, arm, hand, back, leg, foot, outfit, person, phone]`.

Image set link: https://github.com/xiaojie1017/Human-Parts
Object detection link: https://huggingface.co/MnLgt/yolo-human-parse/tree/main
Dataset Assets (Priv_personpart):
- Folder `Priv_personpart/Images/` contains source person-part images we can run directly through inference.
- Supplemental annotation assets (Pascal VOC style) reside in `Priv_personpart/Annotations/` (XML), plus `ImageSets/` splits and `Json_Annos/` conversions.
- We are intentionally keeping these annotation files even though Phase 1 only consumes raw images. They enable potential future steps:
	- Fine-tuning a lighter detection/segmentation model on a subset of part classes (e.g., hands, faces, torso refinement).
	- Converting VOC → COCO → YOLO formats (script `Priv_personpart/tools/pascal2coco.py`).
	- Auditing segmentation outputs against ground-truth boxes (IoU evaluation).
- Current pipeline ignores these annotation directories at runtime; removal is deferred for flexibility.

Pipeline Phase 1 (current):
1. Load YOLO segmentation model.
2. Run inference on an image or folder.
3. For selected part classes, the code will derive bounding rectangle from mask, expand by margin, crop, and save.
4. Produce a visualization image (original + mask overlays + box).

Future Phase: Feed each crop into wound classification model; aggregate predictions.

Quick Usage (after installing ultralytics):
```bash
python ObjectDetection/detect_body_parts.py --source ImageData/images/Wound_dataset/Abrasions --output ObjectDetection/outputs --model MnLgt/yolo-human-parse --classes arm hand leg foot face --margin 0.12 --max-images 200
```

To run on the internal dataset images (ignoring annotations):
```bash
python ObjectDetection/detect_body_parts.py --source ObjectDetection/Priv_personpart/Images --output ObjectDetection/outputs --model MnLgt/yolo-human-parse --classes face hand arm leg foot neck head --alpha-png --max-images 2000
```

Large Datasets:
- Use `--max-images` (default 2000) to limit processing for spot checks.
- For full runs, set `--max-images -1` (or any <=0) to process all images.

Evaluation (box-level mAP vs VOC annotations):
- The `Priv_personpart/Annotations/` directory contains Pascal VOC XML boxes (e.g., face, hand, person).
- Use `eval_voc_mAP.py` to compare model predicted boxes to these GT boxes and report AP per class and mAP.

Examples:
```bash
python ObjectDetection/eval_voc_mAP.py --images ObjectDetection/Priv_personpart/Images --ann ObjectDetection/Priv_personpart/Annotations --model MnLgt/yolo-human-parse --classes face hand --iou 0.5 --conf 0.25 --max-images 1000
```
Notes:
- This evaluates detection quality (boxes). Ground-truth masks are not provided in VOC XML, so segmentation IoU (mask-level) is not computed here.
- If you omit `--classes`, the script evaluates on the overlap of GT labels and model class names.

Outputs:
- `outputs/vis/<image_stem>.jpg` annotated visualization.
- `outputs/crops/<image_stem>/<part_class>_<idx>.jpg` cropped part regions.
