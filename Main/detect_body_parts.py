import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from huggingface_hub import snapshot_download, hf_hub_download
import dill  # required for some legacy checkpoints
import torch  # needed for classification device selection

# classification import is done lazily inside pipeline to keep loose coupling


# NOTE: Torso placeholder
# We currently keep the trunk region aggregated as 'torso'. If downstream
# requirements emerge that need finer granularity (chest / abdomen / pelvis),
# we will revisit and either (a) implement a heuristic decomposition or (b)
# fine-tune a segmentation model with dedicated annotations. For now, using
# 'torso' keeps the pipeline simpler.
PART_DEFAULT = ["face", "arm", "hand", "leg", "foot", "neck", "torso", "head"]  # default classes (head is composite; torso may be refined later)

# possibly calculate the chest / abdomen / pelvis regions from torso masks in future

def load_model(model_path: str):
    """Load YOLO model with HF direct file preference.

    Resolution order:
    1. Local file path.
    2. HF repo string (owner/repo): try specific checkpoint filenames via hf_hub_download.
       Preferred list: epoch 114 then 125. If none succeed, snapshot and pick segmentation ('seg' in name) else largest.
    3. On failure, fallback to official 'yolov8n-seg.pt'.
    4. Otherwise delegate to YOLO for built-in models.
    """
    p = Path(model_path)
    if p.exists():
        print(f"[loader] Local weight found: {p}")
        return YOLO(str(p))
    if '/' in model_path:
        repo_id = model_path.strip()
        preferred = [
            'checkpoint-114/yolo-human-parse-epoch-114.pt',
            'checkpoint-125/yolo-human-parse-epoch-125.pt'
        ]
        for fname in preferred:
            try:
                print(f"[loader] Trying hf_hub_download: {repo_id} :: {fname}")
                weight_path = hf_hub_download(repo_id=repo_id, filename=fname)
                print(f"[loader] Using downloaded checkpoint: {weight_path}")
                return YOLO(weight_path)
            except Exception as e:
                print(f"[loader] Failed direct fetch {fname}: {e}")
        try:
            print(f"[loader] Falling back to snapshot for repo: {repo_id}")
            repo_dir = Path(snapshot_download(repo_id=repo_id))
            pt_files = sorted(repo_dir.rglob('*.pt'))
            if not pt_files:
                raise FileNotFoundError("No .pt weights after snapshot.")
            seg_candidates = [f for f in pt_files if 'seg' in f.name.lower()]
            chosen = seg_candidates[0] if seg_candidates else max(pt_files, key=lambda f: f.stat().st_size)
            print(f"[loader] Using snapshot checkpoint: {chosen} ({chosen.stat().st_size/1e6:.1f}MB)")
            return YOLO(str(chosen))
        except Exception as e:
            print(f"[loader] Snapshot fallback failed: {e}")
            print("[loader] Reverting to official yolov8n-seg.pt")
            return YOLO('yolov8n-seg.pt')
    print(f"[loader] Delegating to YOLO for spec: {model_path}")
    return YOLO(model_path)


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def mask_to_bbox(mask: np.ndarray, margin: float, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Compute bounding rectangle for a binary mask and expand by margin.

    Returns (x1, y1, x2, y2) in image coordinates.
    """
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return 0, 0, 0, 0
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    pad_h = int(h * margin)
    pad_w = int(w * margin)
    H, W = image_shape
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(W - 1, x2 + pad_w)
    y2 = min(H - 1, y2 + pad_h)
    return x1, y1, x2, y2


def save_crop(image: np.ndarray, bbox: tuple[int, int, int, int], out_path: Path):
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return False
    crop = image[y1:y2+1, x1:x2+1]
    Image.fromarray(crop).save(out_path)
    return True


def save_alpha_masked(image: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int], out_path: Path):
    """Save RGBA crop where pixels outside mask are transparent.

    bbox should tightly enclose the mask (precomputed via mask_to_bbox). We re-slice both image and mask
    and build an RGBA image: RGB from original, Alpha = 255 for mask, 0 outside.
    """
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return False
    crop_img = image[y1:y2+1, x1:x2+1]
    crop_mask = mask[y1:y2+1, x1:x2+1] > 0
    if crop_mask.sum() == 0:
        return False
    rgba = np.zeros((crop_img.shape[0], crop_img.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = crop_img
    rgba[..., 3] = crop_mask.astype(np.uint8) * 255
    Image.fromarray(rgba, mode="RGBA").save(out_path)
    return True


def visualize(image: np.ndarray, boxes: List[tuple[int,int,int,int]], labels: List[str], out_path: Path):
    """Simple visualization: draw rectangles and labels. Avoid heavy deps (use PIL)."""
    # box corners will be the four coordinates: (x1, y1, x2, y2)
    img_pil = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for (x1,y1,x2,y2), label in zip(boxes, labels):
        draw.rectangle([x1,y1,x2,y2], outline="red", width=2)
        if font:
            draw.text((x1+3, y1+3), label, fill="yellow", font=font)
    img_pil.save(out_path)


def process_image(
    model,
    image_path: Path,
    output_dir: Path,
    classes_filter: List[str],
    margin: float,
    min_area: int,
    add_head: bool,
    alpha_png: bool,
    device: Optional[str] = None,
    classification_export_dir: Optional[Path] = None,
    save_annotated: bool = True,
    debug: bool = False,
    debug_print: bool = False,
):
    image = Image.open(image_path).convert("RGB")  # load as RGB
    image_np = np.array(image)

    # Run inference. If a device is provided, use Ultralytics predict() so device selection is honored.
    if device is None:
        results = model(str(image_path))
    else:
        results = model.predict(source=str(image_path), device=device, verbose=False)
    print(f"Processing {image_path.name}: {len(results)} result(s)")
    vis_boxes = []
    vis_labels = []

    crops_root = output_dir / "crops" / image_path.stem
    ensure_dir(crops_root)
    if save_annotated:
        annotated_root = output_dir / "annotated"
        ensure_dir(annotated_root)

    # Collect masks for potential composite head (hair + face + upper neck)
    composite_parts = {"hair": [], "face": [], "neck": []}

    for result in results:
        if debug:
            try:
                mask_shape = None if getattr(result, "masks", None) is None else tuple(result.masks.data.shape)
                print(f"[debug] image={image_path.name} boxes={len(result.boxes)} masks_shape={mask_shape}")
            except Exception as e:
                print(f"[debug] could not read mask shape: {e}")
        # Save annotated image (model's own rendering) with labels and masks
        if save_annotated:
            try:
                annotated_img = result.plot()  # numpy array with overlays
                Image.fromarray(annotated_img).save((output_dir / "annotated" / image_path.name))
            except Exception as e:
                if debug:
                    print(f"[debug] failed to plot annotated image: {e}")
        names_map = result.names  # index -> class name
        # Boxes
        if getattr(result, "masks", None) is None:
            print("Warning: result has no segmentation masks; skipping mask-derived crops.")
            continue
        masks = result.masks.data.cpu().numpy()  # shape: (N, H, W)
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        for idx, cls_id in enumerate(cls_ids): # index the list 
            cls_name = names_map.get(cls_id, str(cls_id))
            mask = masks[idx]
            mask_bin_all = (mask > 0.5).astype(np.uint8)
            if cls_name in composite_parts:
                composite_parts[cls_name].append(mask_bin_all)
            if classes_filter and cls_name not in classes_filter:
                continue
            mask_bin = mask_bin_all
            area = mask_bin.sum()
            if area < min_area:
                continue
            bbox = mask_to_bbox(mask_bin, margin=margin, image_shape=image_np.shape[:2])
            filename = f"{image_path.stem}_{cls_name}_{idx}.jpg"
            saved = save_crop(image_np, bbox, crops_root / filename)
            if alpha_png and saved:
                save_alpha_masked(image_np, mask_bin, bbox, crops_root / f"{image_path.stem}_{cls_name}_{idx}_alpha.png")
            if saved:
                vis_boxes.append(bbox)
                vis_labels.append(cls_name)
                if classification_export_dir is not None:
                    cls_dir = classification_export_dir / cls_name
                    ensure_dir(cls_dir)
                    try:
                        Image.open(crops_root / filename).save(cls_dir / filename)
                    except Exception as e:
                        if debug_print:
                            print(f"[debug] classification export failed for {filename}: {e}")

    # Create composite head crop if requested
    if add_head:
        head_components = []
        head_components.extend(composite_parts["hair"])  # hair masks
        head_components.extend(composite_parts["face"])  # face masks
        # Upper portion of neck (top 40%) to connect if needed
        for neck_mask in composite_parts["neck"]:
            rows = np.where(neck_mask > 0)[0]
            if rows.size:
                top_cut = rows.min() + int((rows.max() - rows.min() + 1) * 0.4)
                neck_top = neck_mask.copy()
                neck_top[top_cut:] = 0
                head_components.append(neck_top)
        if head_components:
            head_union = np.clip(sum(head_components), 0, 1)
            if head_union.sum() >= min_area:
                bbox = mask_to_bbox(head_union, margin=margin, image_shape=image_np.shape[:2])
                head_filename = f"{image_path.stem}_head_0.jpg"
                if save_crop(image_np, bbox, crops_root / head_filename):
                    vis_boxes.append(bbox)
                    vis_labels.append("head")
                    if alpha_png:
                        save_alpha_masked(image_np, head_union, bbox, crops_root / f"{image_path.stem}_head_0_alpha.png")
                    if classification_export_dir is not None:
                        cls_dir = classification_export_dir / "head"
                        ensure_dir(cls_dir)
                        try:
                            Image.open(crops_root / head_filename).save(cls_dir / head_filename)
                        except Exception as e:
                            if debug_print:
                                print(f"[debug] classification export failed for composite head: {e}")

    # Visualization
    vis_dir = output_dir / "vis"
    ensure_dir(vis_dir)
    visualize(image_np, vis_boxes, vis_labels, vis_dir / f"{image_path.stem}.jpg")


def iterate_source(
    model,
    source: Path,
    output: Path,
    classes: List[str],
    margin: float,
    min_area: int,
    add_head: bool,
    alpha_png: bool,
    max_images: int,
    device: Optional[str] = None,
    classification_export_dir: Optional[Path] = None,
    save_annotated: bool = True,
    debug: bool = False,
    debug_print: bool = False,
):
    if source.is_file():
        process_image(
            model,
            source,
            output,
            classes,
            margin,
            min_area,
            add_head,
            alpha_png,
            device=device,
            classification_export_dir=classification_export_dir,
            save_annotated=save_annotated,
            debug=debug,
            debug_print=debug_print,
        )
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        all_imgs = [p for p in sorted(source.rglob("*")) if p.suffix.lower() in exts]
        if (max_images is not None) and (max_images > 0):
            all_imgs = all_imgs[:max_images]
            if debug:
                print(f"[debug] Limiting to first {len(all_imgs)} images")
        for img_path in all_imgs:
            process_image(
                model,
                img_path,
                output,
                classes,
                margin,
                min_area,
                add_head,
                alpha_png,
                device=device,
                classification_export_dir=classification_export_dir,
                save_annotated=save_annotated,
                debug=debug,
                debug_print=debug_print,
            )


def run_detection(
    model: str = "MnLgt/yolo-human-parse",
    source: str = "VisualProcessing/ObjectDetection/Priv_personpart/ImageSamples",
    output: str = "Main/PipelineOutputs/DetectionOutput",
    classes: Optional[List[str]] = None,
    margin: float = 0.10,
    min_area: int = 250,
    device: Optional[str] = None,
    add_head: bool = True,
    debug: bool = False,
    alpha_png: bool = False,
    max_images: int = 200,
    classification_export_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run YOLO segmentation + crop extraction.

    This is the entrypoint used by `Main/main_pipeline.py`.
    """
    classes = PART_DEFAULT if classes is None else classes

    source_path = Path(source)
    output_path = Path(output)
    ensure_dir(output_path)

    # If user requests the 'head' class, interpret it as the composite head crop.
    effective_add_head = bool(add_head or ("head" in classes))
    if ("head" in classes) and (not add_head):
        print("[detect] 'head' is in classes; enabling composite head crop.")

    model_obj = load_model(model)

    export_dir_path: Optional[Path] = Path(classification_export_dir) if classification_export_dir else None
    if export_dir_path:
        ensure_dir(export_dir_path)

    iterate_source(
        model_obj,
        source_path,
        output_path,
        classes,
        margin,
        min_area,
        effective_add_head,
        alpha_png,
        max_images,
        device=device,
        classification_export_dir=export_dir_path,
        save_annotated=True,
        debug=debug,
        debug_print=debug,
    )

    summary: Dict[str, Any] = {
        "model": model,
        "source": str(source_path),
        "output": str(output_path),
        "classification_export_dir": (str(export_dir_path) if export_dir_path else None),
        "classes": classes,
    }
    print(f"[detect] Done. Outputs in {output_path}")
    return summary


def detect_and_train_classification(
    detection_export_dir: str,
    classification_epochs: int = 5,
    classification_batch_size: int = 32,
    classification_img_size: int = 224,
    classification_val_ratio: float = 0.2,
    classification_test_ratio: float = 0.0,
    classification_lr: float = 3e-4,
    classification_split_seed: Optional[int] = None,
    freeze_backbone: bool = False,
) -> Dict[str, Any]:
    """Run classification training on a directory produced by run_detection (ImageFolder layout)."""
    from VisualProcessing.Classification.ClassificationModels.simple_train_swin_tiny import train_swin_tiny
    result = train_swin_tiny(
        data_dir=detection_export_dir,
        epochs=classification_epochs,
        batch_size=classification_batch_size,
        img_size=classification_img_size,
        val_ratio=classification_val_ratio,
        test_ratio=classification_test_ratio,
        lr=classification_lr,
        split_seed=classification_split_seed,
        freeze_backbone=freeze_backbone,
    )
    return result

