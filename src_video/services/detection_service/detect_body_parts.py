
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
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


def polygon_to_mask(poly: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a polygon (Nx2) into a binary mask in original image coordinates."""
    H, W = image_shape
    mask_img = Image.new("L", (W, H), 0)
    if poly is None or len(poly) == 0:
        return np.zeros((H, W), dtype=np.uint8)
    pts = [(float(x), float(y)) for x, y in poly]
    ImageDraw.Draw(mask_img).polygon(pts, outline=1, fill=1)
    return np.array(mask_img, dtype=np.uint8)


def polygon_to_bbox(poly: np.ndarray, margin: float, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Compute bounding rectangle for a polygon (Nx2) and expand by margin.

    Returns (x1, y1, x2, y2) in image coordinates.
    """
    H, W = image_shape
    if poly is None or len(poly) == 0:
        return 0, 0, 0, 0
    xs = poly[:, 0]
    ys = poly[:, 1]
    x1 = int(np.floor(xs.min()))
    x2 = int(np.ceil(xs.max()))
    y1 = int(np.floor(ys.min()))
    y2 = int(np.ceil(ys.max()))
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    pad_h = int(h * margin)
    pad_w = int(w * margin)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(W - 1, x2 + pad_w)
    y2 = min(H - 1, y2 + pad_h)
    return x1, y1, x2, y2


def polygon_area(poly: np.ndarray) -> float:
    """Compute polygon area (shoelace formula). Returns area in pixel^2."""
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def rotate_image_np(image: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate HxWxC image by 0/90/180/270 degrees."""
    if degrees not in (0, 90, 180, 270):
        raise ValueError(f"degrees must be one of 0/90/180/270, got {degrees}")
    if degrees == 0:
        return image
    if degrees == 90:
        return np.rot90(image, k=1)
    if degrees == 180:
        return np.rot90(image, k=2)
    return np.rot90(image, k=3)


def choose_best_rotation(
    model,
    image_np: np.ndarray,
    device: Optional[str],
    candidate_degrees: List[int],
) -> int:
    """Pick a rotation that yields the strongest segmentation signal.

    Uses sum of polygon areas across all masks as a simple heuristic.
    """
    best_deg = 0
    best_score = -1.0

    for deg in candidate_degrees:
        rotated = rotate_image_np(image_np, deg)
        predict_kwargs = {"source": rotated, "verbose": False}
        if device is not None:
            predict_kwargs["device"] = device
        results = model.predict(**predict_kwargs)

        score = 0.0
        for result in results:
            if getattr(result, "masks", None) is None or getattr(result.masks, "xy", None) is None:
                continue
            for poly_pts in result.masks.xy:
                if poly_pts is None:
                    continue
                poly = np.array(poly_pts, dtype=np.float32)
                score += polygon_area(poly)

        if score > best_score:
            best_score = score
            best_deg = deg

    return best_deg


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
    auto_orient: bool = True,
    rotate_degrees: int = 0,
    auto_rotate_subject: bool = False,
    classification_export_dir: Optional[Path] = None,
    save_annotated: bool = True,
    debug: bool = False,
    debug_print: bool = False,
):
    image_pil = Image.open(image_path)
    if auto_orient:
        # Respect EXIF orientation tags (common for phone/camera images).
        image_pil = ImageOps.exif_transpose(image_pil)
    image_pil = image_pil.convert("RGB")

    if rotate_degrees not in (0, 90, 180, 270):
        raise ValueError(f"rotate_degrees must be one of 0/90/180/270, got {rotate_degrees}")
    if rotate_degrees:
        # Apply a deterministic rotation when the camera is physically mounted upside down
        # or when images are saved without correct EXIF orientation.
        image_pil = image_pil.rotate(rotate_degrees, expand=True)

    image_np = np.array(image_pil)

    # If the camera is fixed but the *person* may appear upside-down (e.g., head-of-bed camera),
    # optionally try 0 vs 180 and pick the rotation that produces the strongest segmentation.
    if auto_rotate_subject:
        best_deg = choose_best_rotation(model, image_np, device=device, candidate_degrees=[0, 180])
        if best_deg != 0:
            if debug:
                print(f"[detect] auto_rotate_subject: using {best_deg} degrees for {image_path.name}")
            image_np = rotate_image_np(image_np, best_deg)

    # Run inference on the transformed image (not the original file), otherwise EXIF/rotation fixes
    # won't be reflected in the model output.
    predict_kwargs = {"source": image_np, "verbose": False}
    if device is not None:
        predict_kwargs["device"] = device
    results = model.predict(**predict_kwargs)
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
        # Prefer polygons in original image coordinates so crops/vis align with result.plot().
        has_polys = getattr(result.masks, "xy", None) is not None
        masks = result.masks.data.cpu().numpy()  # may be resized/model-space
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        for idx, cls_id in enumerate(cls_ids): # index the list 
            cls_name = names_map.get(cls_id, str(cls_id))
            # Build a binary mask + bbox in original image resolution.
            if has_polys and idx < len(result.masks.xy) and result.masks.xy[idx] is not None:
                poly = np.array(result.masks.xy[idx], dtype=np.float32)
                mask_bin_all = polygon_to_mask(poly, image_np.shape[:2])
                bbox = polygon_to_bbox(poly, margin=margin, image_shape=image_np.shape[:2])
            else:
                # Fallback: threshold tensor mask. If it doesn't match image resolution, scale bbox and resize mask.
                mask = masks[idx]
                mask_bin_all = (mask > 0.5).astype(np.uint8)
                if mask_bin_all.shape != image_np.shape[:2]:
                    mh, mw = mask_bin_all.shape
                    ih, iw = image_np.shape[:2]

                    x1m, y1m, x2m, y2m = mask_to_bbox(mask_bin_all, margin=margin, image_shape=(mh, mw))
                    sx = iw / float(mw)
                    sy = ih / float(mh)
                    x1 = int(round(x1m * sx))
                    x2 = int(round(x2m * sx))
                    y1 = int(round(y1m * sy))
                    y2 = int(round(y2m * sy))
                    x1 = max(0, min(iw - 1, x1))
                    x2 = max(0, min(iw - 1, x2))
                    y1 = max(0, min(ih - 1, y1))
                    y2 = max(0, min(ih - 1, y2))
                    bbox = (x1, y1, x2, y2)

                    mask_img = Image.fromarray(mask_bin_all * 255)
                    mask_img = mask_img.resize((iw, ih), resample=Image.NEAREST)
                    mask_bin_all = (np.array(mask_img) > 0).astype(np.uint8)
                else:
                    bbox = mask_to_bbox(mask_bin_all, margin=margin, image_shape=image_np.shape[:2])
            if cls_name in composite_parts:
                composite_parts[cls_name].append(mask_bin_all)
            if classes_filter and cls_name not in classes_filter:
                continue
            mask_bin = mask_bin_all
            area = mask_bin.sum()
            if area < min_area:
                continue
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
    auto_orient: bool = True,
    rotate_degrees: int = 0,
    auto_rotate_subject: bool = False,
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
            auto_orient=auto_orient,
            rotate_degrees=rotate_degrees,
            auto_rotate_subject=auto_rotate_subject,
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
                auto_orient=auto_orient,
                rotate_degrees=rotate_degrees,
                auto_rotate_subject=auto_rotate_subject,
                classification_export_dir=classification_export_dir,
                save_annotated=save_annotated,
                debug=debug,
                debug_print=debug_print,
            )


def run_detection(
    model: str = "MnLgt/yolo-human-parse",
    source: str = "data/video/source_files",
    output: str = "data/video/output_files/DetectionOutput",
    classes: Optional[List[str]] = None,
    margin: float = 0.10,
    min_area: int = 250,
    device: Optional[str] = None,
    add_head: bool = True,
    debug: bool = False,
    alpha_png: bool = False,
    max_images: int = 200,
    auto_orient: bool = True,
    rotate_degrees: int = 0,
    auto_rotate_subject: bool = False,
    classification_export_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run YOLO segmentation + crop extraction.

    This is the detection entrypoint used by the `src_video` pipeline.
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
        auto_orient=auto_orient,
        rotate_degrees=rotate_degrees,
        auto_rotate_subject=auto_rotate_subject,
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
