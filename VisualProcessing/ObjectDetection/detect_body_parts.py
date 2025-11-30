import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from huggingface_hub import snapshot_download, hf_hub_download
import dill  # required for some legacy checkpoints


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


def process_image(model, image_path: Path, output_dir: Path, classes_filter: List[str], margin: float, min_area: int, add_head: bool, alpha_png: bool, save_annotated: bool = True):
    image = Image.open(image_path).convert("RGB")  # load as RGB
    image_np = np.array(image)
    results = model(str(image_path))  # run inference
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
        if DEBUG_PRINT:
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
                if DEBUG_PRINT:
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
            saved = save_crop(image_np, bbox, crops_root / f"{cls_name}_{idx}.jpg")
            if alpha_png and saved:
                # Use original (non-margin) tight bbox for alpha transparency? For consistency keep margin bbox.
                save_alpha_masked(image_np, mask_bin, bbox, crops_root / f"{cls_name}_{idx}_alpha.png")
            if saved:
                vis_boxes.append(bbox)
                vis_labels.append(cls_name)

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
                if save_crop(image_np, bbox, crops_root / "head_0.jpg"):
                    vis_boxes.append(bbox)
                    vis_labels.append("head")
                    if alpha_png:
                        save_alpha_masked(image_np, head_union, bbox, crops_root / "head_0_alpha.png")

    # Visualization
    vis_dir = output_dir / "vis"
    ensure_dir(vis_dir)
    visualize(image_np, vis_boxes, vis_labels, vis_dir / f"{image_path.stem}.jpg")


def iterate_source(model, source: Path, output: Path, classes: List[str], margin: float, min_area: int, add_head: bool, alpha_png: bool, max_images: int, save_annotated: bool = True):
    if source.is_file():
        process_image(model, source, output, classes, margin, min_area, add_head, alpha_png, save_annotated)
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        all_imgs = [p for p in sorted(source.rglob("*")) if p.suffix.lower() in exts]
        if (max_images is not None) and (max_images > 0):
            all_imgs = all_imgs[:max_images]
            if DEBUG_PRINT:
                print(f"[debug] Limiting to first {len(all_imgs)} images")
        for img_path in all_imgs:
            process_image(model, img_path, output, classes, margin, min_area, add_head, alpha_png, save_annotated)


def parse_args():
    p = argparse.ArgumentParser(description="Body part segmentation + crop extraction (YOLO)")
    p.add_argument("--model", type=str, default="MnLgt/yolo-human-parse", help="Model path or HF repo name")
    # Change this to match your path for imageSamples!
    p.add_argument("--source", type=str, default="ObjectDetection/Priv_personpart/Images", help="Image file or directory (default internal dataset Priv_personpart/Images)")
    p.add_argument("--output", type=str, default="ObjectDetection/outputs", help="Output root directory")
    p.add_argument("--classes", nargs="*", default=PART_DEFAULT, help="Subset of classes to crop (default face arm hand leg foot neck torso head; head is composite; torso may later split into chest/abdomen/pelvis)")
    p.add_argument("--margin", type=float, default=0.10, help="Margin fraction to expand crop bounding box")
    p.add_argument("--min-area", type=int, default=250, help="Minimum mask pixel area to keep a crop (filter tiny artifacts)")
    p.add_argument("--device", type=str, default=None, help="Force device (e.g. 'cpu', '0'). If None, ultralytics auto-selects.")
    p.add_argument("--add-head", action="store_true", help="Also generate composite 'head' crop (hair+face+upper neck)")
    p.add_argument("--debug", action="store_true", help="Print masks tensor shapes and box counts for inspection")
    p.add_argument("--alpha-png", action="store_true", help="Also save RGBA crops with transparent background (mask applied)")
    p.add_argument("--max-images", type=int, default=1000, help="Limit number of images processed (set <=0 for all)")
    return p.parse_args()


def main():
    args = parse_args()
    global DEBUG_PRINT
    DEBUG_PRINT = bool(args.debug)
    model = load_model(args.model)
    if args.device is not None:
        try:
            model.to(args.device)
        except Exception:
            print(f"Could not move model to device {args.device}, continuing on default.")
    source_path = Path(args.source)
    output_path = Path(args.output)
    ensure_dir(output_path)
    # Auto-enable composite head if user requested head in classes but did not pass --add-head
    if ("head" in args.classes) and (not args.add_head):
        args.add_head = True
    iterate_source(model, source_path, output_path, args.classes, args.margin, args.min_area, args.add_head, args.alpha_png, args.max_images, save_annotated=True)
    print(f"Done. Outputs in {output_path}")


if __name__ == "__main__":
    main()
