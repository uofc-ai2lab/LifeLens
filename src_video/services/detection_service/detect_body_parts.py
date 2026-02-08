
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO
from huggingface_hub import snapshot_download, hf_hub_download
import dill  # required for some legacy checkpoints
import torch  # needed for classification device selection


# NOTE: Torso placeholder
# We currently keep the trunk region aggregated as 'torso'. If downstream
# requirements emerge that need finer granularity (chest / abdomen / pelvis),
# we will revisit and either (a) implement a heuristic decomposition or (b)
# fine-tune a segmentation model with dedicated annotations. For now, using
# 'torso' keeps the pipeline simpler.
PART_DEFAULT = ["face", "arm", "hand", "leg", "foot", "neck", "torso", "head"]


_SIDEABLE_PARTS = {"arm", "hand", "leg", "foot"}
_MIDLINE_PARTS = {"torso", "head", "face", "neck"}


def _bbox_center_x(bbox: tuple[int, int, int, int]) -> float:
    x1, _, x2, _ = bbox
    return (float(x1) + float(x2)) * 0.5


def _bbox_area(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    w = max(0.0, float(x2) - float(x1))
    h = max(0.0, float(y2) - float(y1))
    return w * h


def _compute_body_midline_x(results, image_width: int) -> float:
    """Estimate body midline in image coordinates.

    Heuristic: weighted average of center-x for torso/head/face/neck detections.
    This yields *image-left/right* (camera POV), not guaranteed anatomical left/right.
    """
    weighted_sum = 0.0
    weight = 0.0

    for result in results:
        if getattr(result, "boxes", None) is None or getattr(result.boxes, "cls", None) is None:
            continue
        names_map = getattr(result, "names", {})
        try:
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            xyxy_boxes = result.boxes.xyxy.cpu().numpy()
        except Exception:
            continue

        for idx, cls_id in enumerate(cls_ids):
            if idx >= len(xyxy_boxes):
                continue
            cls_name = names_map.get(int(cls_id), str(cls_id))
            if cls_name not in _MIDLINE_PARTS:
                continue
            x1, y1, x2, y2 = xyxy_boxes[idx].tolist()
            bbox = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
            area = _bbox_area(bbox)
            if area <= 1.0:
                continue
            weighted_sum += _bbox_center_x(bbox) * area
            weight += area

    if weight > 0.0:
        return float(weighted_sum / weight)
    return float(image_width) * 0.5


def _label_with_side(cls_name: str, bbox: tuple[int, int, int, int], midline_x: float) -> str:
    """Return a label like 'left-arm' / 'right-arm' for side-able parts."""
    if cls_name not in _SIDEABLE_PARTS:
        return cls_name
    side = "left" if _bbox_center_x(bbox) < float(midline_x) else "right"
    # Use '-' so downstream filename parsing (split on '_') keeps this as one token.
    return f"{side}-{cls_name}"


def load_model(model_path: str):
    """Load YOLO model with HF direct file preference."""
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


def _clip_bbox(bbox: tuple[int, int, int, int], image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    H, W = image_shape
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    return x1, y1, x2, y2


def _expand_bbox_with_margin(
    bbox: tuple[int, int, int, int],
    margin: float,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    H, W = image_shape
    if x2 <= x1 or y2 <= y1:
        return 0, 0, 0, 0
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    pad_h = int(h * margin)
    pad_w = int(w * margin)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(W - 1, x2 + pad_w)
    y2 = min(H - 1, y2 + pad_h)
    return x1, y1, x2, y2


def _resize_mask_to_image(mask_bin: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to the given (H,W) using nearest-neighbor."""
    H, W = image_shape
    if mask_bin.shape == (H, W):
        return mask_bin
    mask_img = Image.fromarray((mask_bin > 0).astype(np.uint8) * 255)
    mask_img = mask_img.resize((W, H), resample=Image.NEAREST)
    return (np.array(mask_img) > 0).astype(np.uint8)


def rotate_image_np(image: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate HxWxC image by 0/90/180/270 degrees.

    Returns a contiguous array (some rotations can create negative-stride views).
    """
    if degrees not in (0, 90, 180, 270):
        raise ValueError(f"degrees must be one of 0/90/180/270, got {degrees}")
    if degrees == 0:
        return np.ascontiguousarray(image)
    if degrees == 90:
        return np.ascontiguousarray(np.rot90(image, k=1))
    if degrees == 180:
        return np.ascontiguousarray(np.rot90(image, k=2))
    return np.ascontiguousarray(np.rot90(image, k=3))


def _prediction_score(results) -> float:
    """Heuristic score for choosing the best rotation.

    Uses sum of box confidences. If boxes are missing, falls back to 0.
    """
    score = 0.0
    try:
        for r in results:
            if getattr(r, "boxes", None) is None or getattr(r.boxes, "conf", None) is None:
                continue
            conf = r.boxes.conf
            # conf can be a tensor
            try:
                score += float(conf.sum().item())
            except Exception:
                score += float(np.sum(conf))
    except Exception:
        return 0.0
    return float(score)


def choose_best_rotation(model, image_np: np.ndarray, device: Optional[str], candidate_degrees: List[int]) -> int:
    """Pick the rotation that yields the strongest detection signal."""
    best_deg = 0
    best_score = -1.0
    for deg in candidate_degrees:
        rotated = rotate_image_np(image_np, deg)
        predict_kwargs = {"source": rotated, "verbose": False}
        if device is not None:
            predict_kwargs["device"] = device
        results = model.predict(**predict_kwargs)
        score = _prediction_score(results)
        if score > best_score:
            best_score = score
            best_deg = deg
    return best_deg


def mask_to_bbox(mask: np.ndarray, margin: float, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Compute bounding rectangle for a binary mask and expand by margin."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return 0, 0, 0, 0
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    return _expand_bbox_with_margin((x1, y1, x2, y2), margin=margin, image_shape=image_shape)


def save_crop(image: np.ndarray, bbox: tuple[int, int, int, int], out_path: Path) -> bool:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return False
    crop = image[y1:y2 + 1, x1:x2 + 1]
    Image.fromarray(crop).save(out_path)
    return True


def save_alpha_masked(image: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int], out_path: Path) -> bool:
    """Save RGBA crop where pixels outside mask are transparent."""
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return False
    crop_img = image[y1:y2 + 1, x1:x2 + 1]
    crop_mask = mask[y1:y2 + 1, x1:x2 + 1] > 0
    if crop_mask.sum() == 0:
        return False
    rgba = np.zeros((crop_img.shape[0], crop_img.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = crop_img
    rgba[..., 3] = crop_mask.astype(np.uint8) * 255
    Image.fromarray(rgba, mode="RGBA").save(out_path)
    return True


def visualize(image: np.ndarray, boxes: List[tuple[int, int, int, int]], labels: List[str], out_path: Path):
    """Simple visualization: draw rectangles and labels."""
    img_pil = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        if font:
            draw.text((x1 + 3, y1 + 3), label, fill="yellow", font=font)
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
    auto_rotate_subject: bool = True,
    classification_export_dir: Optional[Path] = None,
    save_annotated: bool = True,
    side_labels: bool = True,
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
        # Deterministic rotation for fixed camera setups.
        image_pil = image_pil.rotate(rotate_degrees, expand=True)

    image_np = np.ascontiguousarray(np.array(image_pil))

    # Auto-fix upside-down subjects: try 0 vs 180 and pick the best.
    if auto_rotate_subject:
        best_deg = choose_best_rotation(model, image_np, device=device, candidate_degrees=[0, 180])
        if best_deg != 0 and debug:
            print(f"[detect] auto_rotate_subject: using {best_deg} degrees for {image_path.name}")
        image_np = rotate_image_np(image_np, best_deg)

    # Key for crop/annotated alignment: run YOLO on the same pixels we're cropping.
    predict_kwargs = {"source": image_np, "verbose": False}
    if device is not None:
        predict_kwargs["device"] = device
    results = model.predict(**predict_kwargs)

    midline_x = None
    if side_labels:
        try:
            midline_x = _compute_body_midline_x(results, image_width=int(image_np.shape[1]))
        except Exception:
            midline_x = float(image_np.shape[1]) * 0.5

    print(f"Processing {image_path.name}: {len(results)} result(s)")
    vis_boxes: List[tuple[int, int, int, int]] = []
    vis_labels: List[str] = []

    crops_root = output_dir / "crops" / image_path.stem
    ensure_dir(crops_root)
    if save_annotated:
        annotated_root = output_dir / "annotated"
        ensure_dir(annotated_root)

    composite_parts: Dict[str, List[np.ndarray]] = {"hair": [], "face": [], "neck": []}

    for result in results:
        if save_annotated:
            try:
                annotated_img = result.plot()
                Image.fromarray(annotated_img).save((output_dir / "annotated" / image_path.name))
            except Exception as e:
                if debug:
                    print(f"[debug] failed to plot annotated image: {e}")

        names_map = result.names
        if getattr(result, "boxes", None) is None or getattr(result.boxes, "cls", None) is None:
            continue

        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        xyxy_boxes = None
        try:
            xyxy_boxes = result.boxes.xyxy.cpu().numpy()
        except Exception:
            xyxy_boxes = None

        masks = None
        if getattr(result, "masks", None) is not None and getattr(result.masks, "data", None) is not None:
            try:
                masks = result.masks.data.cpu().numpy()
            except Exception:
                masks = None

        for idx, cls_id in enumerate(cls_ids):
            cls_name = names_map.get(int(cls_id), str(cls_id))

            # Mask (used for min_area + alpha export + composite head). Resize to image-space.
            mask_bin = None
            if masks is not None and idx < len(masks):
                mask_bin_model = (masks[idx] > 0.5).astype(np.uint8)
                mask_bin = _resize_mask_to_image(mask_bin_model, image_np.shape[:2])
                if cls_name in composite_parts:
                    composite_parts[cls_name].append(mask_bin)

            if classes_filter and cls_name not in classes_filter:
                continue

            if mask_bin is not None:
                if int(mask_bin.sum()) < int(min_area):
                    continue

            # Primary crop region: YOLO box (matches result.plot overlays).
            bbox = (0, 0, 0, 0)
            if xyxy_boxes is not None and idx < len(xyxy_boxes):
                x1, y1, x2, y2 = xyxy_boxes[idx].tolist()
                bbox = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
                bbox = _expand_bbox_with_margin(bbox, margin=margin, image_shape=image_np.shape[:2])
                bbox = _clip_bbox(bbox, image_shape=image_np.shape[:2])

            # Fallback: mask-derived bbox if box is missing/degenerate.
            if (bbox[2] <= bbox[0] or bbox[3] <= bbox[1]) and mask_bin is not None:
                bbox = mask_to_bbox(mask_bin, margin=margin, image_shape=image_np.shape[:2])

            export_cls_name = cls_name
            if side_labels and (midline_x is not None):
                export_cls_name = _label_with_side(cls_name, bbox, midline_x)

            filename = f"{image_path.stem}_{export_cls_name}_{idx}.jpg"
            saved = save_crop(image_np, bbox, crops_root / filename)
            if alpha_png and saved and mask_bin is not None:
                save_alpha_masked(
                    image_np,
                    mask_bin,
                    bbox,
                    crops_root / f"{image_path.stem}_{export_cls_name}_{idx}_alpha.png",
                )
            if saved:
                vis_boxes.append(bbox)
                vis_labels.append(export_cls_name)
                if classification_export_dir is not None:
                    cls_dir = classification_export_dir / export_cls_name
                    ensure_dir(cls_dir)
                    try:
                        Image.open(crops_root / filename).save(cls_dir / filename)
                    except Exception as e:
                        if debug_print:
                            print(f"[debug] classification export failed for {filename}: {e}")

    if add_head:
        head_components: List[np.ndarray] = []
        head_components.extend(composite_parts["hair"])
        head_components.extend(composite_parts["face"])
        for neck_mask in composite_parts["neck"]:
            rows = np.where(neck_mask > 0)[0]
            if rows.size:
                top_cut = int(rows.min() + (rows.max() - rows.min() + 1) * 0.4)
                neck_top = neck_mask.copy()
                neck_top[top_cut:] = 0
                head_components.append(neck_top)
        if head_components:
            head_union = np.clip(sum(head_components), 0, 1).astype(np.uint8)
            if int(head_union.sum()) >= int(min_area):
                bbox = mask_to_bbox(head_union, margin=margin, image_shape=image_np.shape[:2])
                head_filename = f"{image_path.stem}_head_0.jpg"
                if save_crop(image_np, bbox, crops_root / head_filename):
                    vis_boxes.append(bbox)
                    vis_labels.append("head")
                    if alpha_png:
                        save_alpha_masked(
                            image_np,
                            head_union,
                            bbox,
                            crops_root / f"{image_path.stem}_head_0_alpha.png",
                        )
                    if classification_export_dir is not None:
                        cls_dir = classification_export_dir / "head"
                        ensure_dir(cls_dir)
                        try:
                            Image.open(crops_root / head_filename).save(cls_dir / head_filename)
                        except Exception as e:
                            if debug_print:
                                print(f"[debug] classification export failed for composite head: {e}")

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
    auto_rotate_subject: bool = True,
    classification_export_dir: Optional[Path] = None,
    save_annotated: bool = True,
    side_labels: bool = True,
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
            side_labels=side_labels,
            debug=debug,
            debug_print=debug_print,
        )
        return

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
            side_labels=side_labels,
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
    auto_rotate_subject: bool = True,
    classification_export_dir: Optional[str] = None,
    side_labels: bool = True,
) -> Dict[str, Any]:
    """Run YOLO segmentation + crop extraction.

    Crop behavior:
    - Primary: YOLO `boxes.xyxy` (matches `result.plot()` rectangles)
    - Fallback: mask-derived bbox (when a box is missing/degenerate)

    Rotation behavior:
    - `auto_orient`: applies EXIF orientation fix before inference.
    - `rotate_degrees`: deterministic 0/90/180/270 rotation for fixed camera setups.
    - `auto_rotate_subject`: tries 0 vs 180 and picks the best detection signal (helps upside-down photos).
    """
    classes = PART_DEFAULT if classes is None else classes

    source_path = Path(source)
    output_path = Path(output)
    ensure_dir(output_path)

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
        side_labels=side_labels,
        debug=debug,
        debug_print=debug,
    )

    summary: Dict[str, Any] = {
        "model": model,
        "source": str(source_path),
        "output": str(output_path),
        "classification_export_dir": (str(export_dir_path) if export_dir_path else None),
        "classes": classes,
        "auto_orient": auto_orient,
        "rotate_degrees": rotate_degrees,
        "auto_rotate_subject": auto_rotate_subject,
        "side_labels": bool(side_labels),
    }
    print(f"[detect] Done. Outputs in {output_path}")
    return summary