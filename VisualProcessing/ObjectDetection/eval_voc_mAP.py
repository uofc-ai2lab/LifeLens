import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import numpy as np
from ultralytics import YOLO


def parse_voc_xml(xml_path: Path) -> List[Tuple[str, np.ndarray]]:
    objs: List[Tuple[str, np.ndarray]] = []
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name").strip()
            bb = obj.find("bndbox")
            xmin = int(float(bb.findtext("xmin")))
            ymin = int(float(bb.findtext("ymin")))
            xmax = int(float(bb.findtext("xmax")))
            ymax = int(float(bb.findtext("ymax")))
            objs.append((name, np.array([xmin, ymin, xmax, ymax], dtype=np.float32)))
    except Exception:
        pass
    return objs


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def average_precision(rec: np.ndarray, prec: np.ndarray) -> float:
    # Standard integral AP
    # Ensure sentinel endpoints
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def evaluate(images_dir: Path, ann_dir: Path, model_name: str, classes: List[str], iou_thr: float, conf_thr: float, max_images: int, device: str | None):
    model = YOLO(model_name)
    if device is not None:
        try:
            model.to(device)
        except Exception:
            print(f"Could not move model to device {device}, continuing on default.")

    # Collect stems that have both image and annotation
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = [p for p in sorted(images_dir.rglob("*")) if p.suffix.lower() in exts]
    if max_images and max_images > 0:
        img_paths = img_paths[:max_images]
    ann_map: Dict[str, Path] = {}
    for xmlp in ann_dir.glob("*.xml"):
        ann_map[xmlp.stem] = xmlp

    # Determine evaluation classes ONCE.
    # If user provided --classes, we use it. Otherwise we evaluate the overlap between
    # (a) VOC annotation labels present in ann_dir and (b) model label names.
    provided_classes = list(classes) if classes else None

    # Gather all GT labels from annotations
    gt_names_all: set[str] = set()
    for xmlp in ann_dir.glob("*.xml"):
        for name, _box in parse_voc_xml(xmlp):
            gt_names_all.add(name)

    # Get model label names without running inference if possible
    names_map = getattr(model, "names", None)
    if names_map is None:
        names_map = getattr(getattr(model, "model", None), "names", None)

    if names_map is None:
        # Fallback: run one inference to obtain names map
        for img_path in img_paths:
            stem = img_path.stem
            if ann_map.get(stem) is None:
                continue
            results = model(str(img_path))
            if results:
                names_map = results[0].names
                break

    if names_map is None:
        raise RuntimeError("Could not determine model class names.")

    if provided_classes is None:
        model_names = set(names_map.values()) if isinstance(names_map, dict) else set(names_map)
        classes = sorted(list(gt_names_all & model_names))
    else:
        classes = provided_classes

    if not classes:
        print("No classes to evaluate (empty overlap between GT labels and model labels).")
        return

    # Accumulators per class
    preds_by_class: Dict[str, List[Tuple[str, float, np.ndarray]]] = {}
    gts_by_class: Dict[str, Dict[str, List[np.ndarray]]] = {}

    processed = 0
    for img_path in img_paths:
        stem = img_path.stem
        xml_path = ann_map.get(stem)
        if not xml_path:
            continue
        gts = parse_voc_xml(xml_path)
        if not gts:
            continue
        results = model(str(img_path))
        if len(results) == 0:
            continue
        r = results[0]
        names_map = r.names

        # Prepare GT storage per class
        for cname in classes:
            gts_by_class.setdefault(cname, {})
            gts_by_class[cname].setdefault(stem, [])
        for cname, box in gts:
            if cname in classes:
                gts_by_class[cname][stem].append(box.astype(np.float32))

        # Predictions: use model-predicted boxes (xyxy), confidences, and class ids
        if getattr(r, "boxes", None) is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else None
        cls_ids = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, "cls") else None
        confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else None
        if xyxy is None or cls_ids is None or confs is None:
            continue
        for box, cid, conf in zip(xyxy, cls_ids, confs):
            if conf < conf_thr:
                continue
            cname = names_map.get(int(cid), str(int(cid)))
            if cname not in classes:
                continue
            preds_by_class.setdefault(cname, []).append((stem, float(conf), box.astype(np.float32)))

        processed += 1
        if processed % 50 == 0:
            print(f"Processed {processed} images…")

    # Compute AP per class
    ap_per_class: Dict[str, float] = {}
    summary_lines = []
    for cname in classes:
        preds = preds_by_class.get(cname, [])
        gt_for_class = gts_by_class.get(cname, {})
        npos = sum(len(v) for v in gt_for_class.values())
        if npos == 0:
            ap_per_class[cname] = float("nan")
            summary_lines.append(f"{cname:>10}: no GT boxes, AP=nan")
            continue
        # sort predictions by confidence desc
        preds = sorted(preds, key=lambda x: -x[1])
        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)
        matched: Dict[str, List[bool]] = {img_id: [False] * len(gt_for_class.get(img_id, [])) for img_id in gt_for_class.keys()}

        for i, (img_id, conf, pbox) in enumerate(preds):
            gts_img = gt_for_class.get(img_id, [])
            if not gts_img:
                fp[i] = 1.0
                continue
            ious = np.array([iou_xyxy(pbox, g) for g in gts_img], dtype=np.float32)
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx]) if ious.size else 0.0
            if best_iou >= iou_thr and not matched[img_id][best_idx]:
                tp[i] = 1.0
                matched[img_id][best_idx] = True
            else:
                fp[i] = 1.0

        fp_cum = np.cumsum(fp)
        tp_cum = np.cumsum(tp)
        rec = tp_cum / max(npos, 1)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        ap = average_precision(rec, prec)
        ap_per_class[cname] = ap
        summary_lines.append(f"{cname:>10}: AP@{iou_thr:.2f} = {ap:.4f} (GT={npos}, P={len(preds)})")

    valid_aps = [v for v in ap_per_class.values() if not np.isnan(v)]
    mAP = float(np.mean(valid_aps)) if valid_aps else float("nan")
    print("\nEvaluation summary (VOC box-level):")
    for line in summary_lines:
        print(line)
    print(f"\n mAP@{iou_thr:.2f} over {len(valid_aps)} classes = {mAP:.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLO predictions vs VOC XML boxes (box-level AP/mAP)")
    p.add_argument("--images", type=str, default="ObjectDetection/Priv_personpart/Images", help="Images directory")
    p.add_argument("--ann", type=str, default="ObjectDetection/Priv_personpart/Annotations", help="VOC Annotations directory (XML)")
    p.add_argument("--model", type=str, default="MnLgt/yolo-human-parse", help="Model path or HF repo name")
    p.add_argument("--classes", nargs="*", default=None, help="Limit to these classes (default: overlap between GT and model)")
    p.add_argument("--iou", type=float, default=0.50, help="IoU threshold for TP (VOC AP)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions")
    p.add_argument("--max-images", type=int, default=1000, help="Limit number of images evaluated (<=0 for all)")
    p.add_argument("--device", type=str, default=None, help="Device id or 'cpu'")
    return p.parse_args()


def main():
    args = parse_args()
    images_dir = Path(args.images)
    ann_dir = Path(args.ann)
    max_images = None if args.max_images is None else int(args.max_images)
    evaluate(images_dir, ann_dir, args.model, args.classes, float(args.iou), float(args.conf), max_images, args.device)


if __name__ == "__main__":
    main()
