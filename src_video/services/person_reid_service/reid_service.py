from __future__ import annotations

"""
reid_service.py  —  Multi-person body ReID using YOLO ONNX + ResNet50
======================================================================
No pycuda / MobileFaceNet / YuNet required.

Detector
--------
YOLOv8n (person class only) exported to ONNX.
Export command (one-time, on any machine with ultralytics installed):

    pip install ultralytics
    python -c "
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    model.export(format='onnx', imgsz=640, opset=12, simplify=True)
    # outputs yolov8n.onnx — copy to your reid service directory
    "

The ONNX file runs with onnxruntime-gpu (already in your environment).
No ultralytics needed at inference time — pure ORT + NumPy.

Architecture
------------
PRIMARY PATH  — per-person body crop matching
  1. YOLO detects all persons in the frame (class 0).
  2. Each person bbox is cropped from the frame.
  3. ResNet50 embeds each crop → 2048-dim vector.
  4. Cosine similarity computed per-person vs enrolled reference.
  5. Snapshot triggered ONLY if at least one person exceeds threshold.
     If persons detected but none match → return None.

FALLBACK PATH — center body crop
  Used ONLY when YOLO detects zero persons in the frame.
  Higher threshold, LOW confidence flag.

Key improvements vs. current_reid_service.py
---------------------------------------------
1. Per-person bbox loop replaces the single full-frame crop.
2. "Persons detected but no match" returns None immediately.
   A stranger in frame cannot trigger a snapshot.
3. YOLO produces tight, person-only bboxes — no body expand heuristics.
4. Enrollment averages ENROLL_N_FRAMES crops for a stable reference.
5. Threshold raised to 0.72 (body crops need higher bar than original 0.65).

Threshold guidance
------------------
Body-crop cosine similarity is sensitive to clothing and lighting.
Recommended starting points:
  > 0.76  →  high confidence same person
  0.72–0.76 → probable match
  < 0.72  →  treat as different person

To calibrate: enable debug logging, walk in front of the camera alone,
note your similarity scores, then add a second person and note theirs.
Pick a threshold in the gap.

False positive mitigations
--------------------------
- YOLO gives per-person isolation: a stranger is never blended with you.
- Raised threshold (0.72 vs 0.65).
- Enrollment averages 8 frames, not 5.
- Body fallback (no detections) uses a stricter threshold (0.77).
- SNAPSHOT_ON_LOW_CONFIDENCE in main_video.py skips fallback snapshots.
- PERSON_MIN_AREA_PX filters tiny/distant detections.
"""

import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np

from config.logger import Logger

log = Logger("[video][reid]")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL PATHS
# ═══════════════════════════════════════════════════════════════════════════

_DIR = Path(__file__).parent
_DEFAULT_YOLO_PATH = str(_DIR / "yolov8n.onnx")
_DEFAULT_REID_PATH = str(_DIR / "resnet50_market1501_aicity156.onnx")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# ── YOLO person detector ──────────────────────────────────────────────────
YOLO_INPUT_SIZE       = 640          # square input; YOLOv8n default
YOLO_CONF_THRESH      = 0.45         # minimum detection confidence
YOLO_NMS_IOU_THRESH   = 0.45         # NMS IoU threshold
YOLO_PERSON_CLASS     = 0            # COCO class 0 = person
PERSON_MIN_AREA_PX    = 3000         # ignore detections smaller than this
                                     # (pixels²) — filters out distant bystanders
                                     # tune based on your camera distance

# ── ResNet50 body ReID ────────────────────────────────────────────────────
REID_INPUT_SIZE = (128, 256)          # (W, H) Market1501 canonical
REID_MEAN       = np.array([0.485, 0.456, 0.406], dtype=np.float32)
REID_STD        = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Matching thresholds ───────────────────────────────────────────────────
BODY_SIMILARITY_THRESH      = 0.72   # primary path — YOLO bbox → body crop
BODY_HIGH_CONF_THRESH       = 0.76   # above this → high confidence
BODY_FALLBACK_THRESH        = 0.77   # fallback path (no YOLO detections)

# ── Enrollment ────────────────────────────────────────────────────────────
ENROLL_N_FRAMES  = 8
REID_COOLDOWN_S  = 5.0


# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN ENTITIES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ReIDEvent:
    timestamp:   float
    similarity:  float
    bbox:        tuple[int, int, int, int]   # x, y, w, h in pixel coords
    frame:       Optional[np.ndarray] = None
    match_type:  str = "body_crop"           # "body_crop" | "body_fallback"
    confidence:  str = "high"


@dataclass
class EnrollmentState:
    embedding:         Optional[np.ndarray] = None
    enrolled_at:       float = 0.0
    embeddings_buffer: list  = field(default_factory=list)

    @property
    def is_enrolled(self) -> bool:
        return self.embedding is not None

    def add_candidate(self, emb: np.ndarray) -> bool:
        self.embeddings_buffer.append(emb)
        n = len(self.embeddings_buffer)
        if n >= ENROLL_N_FRAMES:
            stack          = np.stack(self.embeddings_buffer, axis=0)
            mean           = stack.mean(axis=0)
            self.embedding = mean / (np.linalg.norm(mean) + 1e-6)
            self.enrolled_at = time.time()
            self.embeddings_buffer.clear()
            log.success(f"Enrolled: averaged {ENROLL_N_FRAMES} body embeddings")
            return True
        log.info(f"Enrollment: {n}/{ENROLL_N_FRAMES} frames buffered")
        return False

    def reset(self):
        self.embedding = None
        self.enrolled_at = 0.0
        self.embeddings_buffer.clear()
        log.info("Enrollment state reset")


# ═══════════════════════════════════════════════════════════════════════════
# YOLO PERSON DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class YOLOPersonDetector:
    """
    YOLOv8n (ONNX) person detector.

    Runs pure ORT inference — no ultralytics at runtime.
    Handles its own pre/post-processing:
      - letterbox resize to 640×640
      - decode YOLOv8 output tensor (1, 84, 8400)
      - filter to class 0 (person) above conf threshold
      - NMS
      - unscale bboxes back to original frame coords

    Input  : (1, 3, 640, 640)  float32  normalised [0,1] RGB
    Output : (1, 84, 8400)     float32  [cx, cy, w, h, cls0..cls79]
    """

    def __init__(
        self,
        model_path:   str   = _DEFAULT_YOLO_PATH,
        input_size:   int   = YOLO_INPUT_SIZE,
        conf_thresh:  float = YOLO_CONF_THRESH,
        nms_thresh:   float = YOLO_NMS_IOU_THRESH,
    ):
        self._session    = None
        self._inp_name   = None
        self._failed     = False
        self._input_size = input_size
        self._conf       = conf_thresh
        self._nms        = nms_thresh

        if not Path(model_path).exists():
            log.warning(
                f"YOLO model not found: {model_path}\n"
                "  Export with:\n"
                "    pip install ultralytics\n"
                "    python -c \"from ultralytics import YOLO; "
                "YOLO('yolov8n.pt').export(format='onnx', imgsz=640, opset=12)\"\n"
                "  Falling back to center-crop body mode."
            )
            self._failed = True
            return

        try:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2
            self._session  = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
            self._inp_name = self._session.get_inputs()[0].name
            log.success(f"YOLO person detector loaded ({self._session.get_providers()[0]})")
        except Exception as e:
            log.error(f"YOLO load failed: {e} — fallback mode")
            self._failed = True

    @property
    def available(self) -> bool:
        return self._session is not None and not self._failed

    # ── Pre/post-processing ───────────────────────────────────────────────

    def _letterbox(
        self, img: np.ndarray, size: int
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Resize with padding to a square canvas.
        Returns (padded_img, scale, (pad_left, pad_top)).
        """
        h, w = img.shape[:2]
        scale = min(size / h, size / w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_left = (size - new_w) // 2
        pad_top  = (size - new_h) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        return canvas, scale, (pad_left, pad_top)

    def _decode(
        self,
        output:    np.ndarray,          # (1, 84, 8400)
        scale:     float,
        pad:       tuple[int, int],
        orig_h:    int,
        orig_w:    int,
    ) -> list[dict]:
        """
        Decode YOLOv8 output tensor to person bboxes in original frame coords.
        Returns list of {'bbox': (x,y,w,h), 'score': float}.
        """
        preds = output[0]               # (84, 8400)
        # rows: [cx, cy, w, h, cls0_score, cls1_score, ...]
        # transpose to (8400, 84) for easier indexing
        preds = preds.T                 # (8400, 84)

        # Person class score is column 4 (index = 4 + YOLO_PERSON_CLASS)
        person_scores = preds[:, 4 + YOLO_PERSON_CLASS]
        mask = person_scores >= self._conf

        if not np.any(mask):
            return []

        filtered = preds[mask]          # (N, 84)
        scores   = person_scores[mask]  # (N,)

        # cx, cy, w, h in letterboxed 640×640 space
        cx = filtered[:, 0]
        cy = filtered[:, 1]
        bw = filtered[:, 2]
        bh = filtered[:, 3]

        # Convert to x1y1x2y2 in letterboxed space
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Unscale: remove padding, divide by scale
        pad_left, pad_top = pad
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top)  / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top)  / scale

        # Clamp to original frame
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # NMS (cv2 expects x1y1wh)
        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices    = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores.tolist(),
            self._conf,
            self._nms,
        )

        results = []
        if len(indices) == 0:
            return results

        # cv2.dnn.NMSBoxes returns shape (N,1) or (N,) depending on OpenCV version
        indices = np.array(indices).flatten()

        for i in indices:
            bx = int(x1[i]);  by = int(y1[i])
            bw_ = int(x2[i] - x1[i]);  bh_ = int(y2[i] - y1[i])
            area = bw_ * bh_
            if area < PERSON_MIN_AREA_PX:
                continue
            results.append({
                "bbox":  (bx, by, bw_, bh_),
                "score": float(scores[i]),
            })

        return results

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Detect all persons in frame_bgr.

        Returns list of:
            { 'bbox': (x, y, w, h), 'score': float }
        Detections smaller than PERSON_MIN_AREA_PX are filtered out.
        Returns [] when detector unavailable or no persons found.
        """
        if not self.available:
            return []

        orig_h, orig_w = frame_bgr.shape[:2]
        letterboxed, scale, pad = self._letterbox(frame_bgr, self._input_size)

        # BGR -> RGB, normalise to [0,1], NCHW
        rgb  = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = rgb.transpose(2, 0, 1)[np.newaxis]

        try:
            output = self._session.run(None, {self._inp_name: blob})
        except Exception as e:
            if not self._failed:
                log.error(f"YOLO inference error: {e}")
                self._failed = True
            return []

        return self._decode(output[0], scale, pad, orig_h, orig_w)


# ═══════════════════════════════════════════════════════════════════════════
# BODY CROP UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def extract_center_body_crop(
    frame_bgr: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Center-of-frame crop — last-resort fallback when YOLO finds nothing."""
    fh, fw  = frame_bgr.shape[:2]
    crop_w  = int(fw * 0.6)
    crop_h  = int(fh * 0.8)
    x       = (fw - crop_w) // 2
    y       = int(fh * 0.1)
    return frame_bgr[y:y + crop_h, x:x + crop_w], (x, y, crop_w, crop_h)


# ═══════════════════════════════════════════════════════════════════════════
# RESNET50 EMBEDDER  (unchanged model)
# ═══════════════════════════════════════════════════════════════════════════

class ResNet50ReIDEmbedder:
    """
    resnet50_market1501_aicity156 via ONNX Runtime.
    Input  : (1, 3, 256, 128)  float32  ImageNet-normalised RGB
    Output : (1, 2048)         float32  L2-normalised feature vector
    """

    def __init__(self, model_path: str = _DEFAULT_REID_PATH):
        self._session  = None
        self._inp_name = None
        self._failed   = False

        if not Path(model_path).exists():
            log.warning(f"ReID model not found: {model_path}")
            self._failed = True
            return

        try:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 2
            self._session  = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
            self._inp_name = self._session.get_inputs()[0].name
            log.success(f"ResNet50 embedder loaded ({self._session.get_providers()[0]})")
        except Exception as e:
            log.error(f"ResNet50 load failed: {e}")
            self._failed = True

    @property
    def available(self) -> bool:
        return self._session is not None and not self._failed

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        if not self.available or crop_bgr is None or crop_bgr.size == 0:
            return np.zeros(2048, dtype=np.float32)
        resized = cv2.resize(crop_bgr, REID_INPUT_SIZE)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normed  = (rgb - REID_MEAN) / REID_STD
        blob    = normed.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        emb     = self._session.run(None, {self._inp_name: blob})[0][0]
        return emb / (np.linalg.norm(emb) + 1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ═══════════════════════════════════════════════════════════════════════════
# REID SERVICE
# ═══════════════════════════════════════════════════════════════════════════

class ReIDService:
    """
    Multi-person body ReID using YOLO + ResNet50.

    ENROLLMENT  (while AprilTag visible)
    ─────────────────────────────────────
    1. YOLO detects all persons in frame.
    2. The LARGEST bbox is selected (closest person, assumed to be the
       user standing in front of the marker).
    3. That person's crop is embedded by ResNet50.
    4. Steps 1-3 repeat for ENROLL_N_FRAMES, then embeddings are
       averaged into a single stable reference vector.
    If YOLO detects nothing, center-crop fallback is used.

    TRACKING  (every frame after enrollment)
    ─────────────────────────────────────────
    1. YOLO detects ALL persons in the frame.
    2. For EACH person bbox, ResNet50 embeds the crop individually.
    3. The best cosine similarity across all persons is found.
    4. Triggered ONLY if best_sim >= BODY_SIMILARITY_THRESH.
       If persons are detected but none match → return None.
       A stranger in frame cannot trigger a snapshot.
    5. If YOLO finds nobody → center-crop fallback at stricter threshold.
    """

    def __init__(
        self,
        person_detector:   YOLOPersonDetector,
        embedder:          ResNet50ReIDEmbedder,
        similarity_thresh: float = BODY_SIMILARITY_THRESH,
        fallback_thresh:   float = BODY_FALLBACK_THRESH,
        reid_cooldown_s:   float = REID_COOLDOWN_S,
        on_reid_callback:  Optional[Callable[[ReIDEvent], None]] = None,
    ):
        self._detector    = person_detector
        self._embedder    = embedder
        self._thresh      = similarity_thresh
        self._fb_thresh   = fallback_thresh
        self._cooldown    = reid_cooldown_s
        self._callback    = on_reid_callback

        self._state       = EnrollmentState()
        self._last_reid   = 0.0
        self._lock        = threading.Lock()

        self._last_persons: list[dict] = []
        self._last_event:   Optional[ReIDEvent] = None

    # ── Enrollment ──────────────────────────────────────────────────────────

    def enroll_from_frame(self, frame_bgr: np.ndarray) -> bool:
        persons = self._detector.detect(frame_bgr)

        if persons:
            # Enroll from the LARGEST person — closest to camera
            primary = max(persons, key=lambda p: p["bbox"][2] * p["bbox"][3])
            x, y, w, h = primary["bbox"]
            fh, fw = frame_bgr.shape[:2]
            crop = frame_bgr[max(y, 0):min(y+h, fh), max(x, 0):min(x+w, fw)]

            if crop.size == 0:
                log.warning("Enrollment: person crop empty — skipping frame")
                return False

            emb = self._embedder.embed(crop)
            with self._lock:
                return self._state.add_candidate(emb)

        # No person detected → center-crop fallback
        log.info("Enrollment: no person detected by YOLO — using center-crop fallback")
        crop, _ = extract_center_body_crop(frame_bgr)
        if crop.size == 0:
            return False

        emb = self._embedder.embed(crop)
        with self._lock:
            return self._state.add_candidate(emb)

    def reset_enrollment(self):
        with self._lock:
            self._state.reset()
        self._last_persons = []
        self._last_event   = None

    # ── Per-frame tracking ────────────────────────────────────────────────────

    def process_frame(
        self,
        frame_bgr:     np.ndarray,
        person_bboxes: Optional[list[tuple[int, int, int, int]]] = None,
    ) -> Optional[ReIDEvent]:
        """
        Multi-person tracking pass.

        person_bboxes: optional external detections (e.g. from run_detection).
                       If provided, YOLO is skipped and these bboxes are used.
                       Useful if run_detection already runs on each frame.
                       If None (default), YOLO runs internally.
        """
        with self._lock:
            if not self._state.is_enrolled:
                return None
            ref_emb = self._state.embedding

        fh, fw = frame_bgr.shape[:2]

        # Use external bboxes if provided, otherwise run YOLO
        if person_bboxes is not None:
            persons = [
                {"bbox": bbox, "score": 1.0}
                for bbox in person_bboxes
            ]
        else:
            persons = self._detector.detect(frame_bgr)

        self._last_persons = persons

        # ── Primary path: per-person embedding loop ───────────────────────────
        if persons:
            best_sim  = -1.0
            best_bbox = None

            for person in persons:
                x, y, w, h = person["bbox"]
                crop = frame_bgr[max(y, 0):min(y+h, fh), max(x, 0):min(x+w, fw)]
                if crop.size == 0:
                    continue

                emb = self._embedder.embed(crop)
                sim = cosine_similarity(ref_emb, emb)

                # log.debug(
                #     f"Person {person['bbox']} "
                #     f"det={person['score']:.2f} sim={sim:.3f}"
                # )

                if sim > best_sim:
                    best_sim  = sim
                    best_bbox = person["bbox"]
            
            log.debug(
                    f"Person {best_bbox} "
                    f"det={person['score']:.2f} sim={best_sim:.3f}"
                )

            # Persons detected but none match → enrolled person NOT here
            if best_sim >= self._thresh:
                confidence = "high" if best_sim >= BODY_HIGH_CONF_THRESH else "low"
                return self._maybe_emit(
                    frame_bgr, best_sim, best_bbox,
                    match_type="body_crop", confidence=confidence,
                )
            else:
                log.debug(
                    f"Persons detected, no match "
                    f"(best={best_sim:.3f}<{self._thresh}) "
                    f"enrolled person NOT in frame"
                )
                return None

        # ── Fallback: YOLO found nobody ───────────────────────────────────────
        crop, bbox = extract_center_body_crop(frame_bgr)
        if crop.size == 0:
            return None

        emb = self._embedder.embed(crop)
        sim = cosine_similarity(ref_emb, emb)

        if sim >= self._fb_thresh:
            return self._maybe_emit(
                frame_bgr, sim, bbox,
                match_type="body_fallback", confidence="low",
            )

        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _maybe_emit(
        self,
        frame_bgr:  np.ndarray,
        similarity: float,
        bbox:       tuple,
        match_type: str,
        confidence: str,
    ) -> Optional[ReIDEvent]:
        now = time.time()
        if (now - self._last_reid) < self._cooldown:
            return None

        self._last_reid = now
        event = ReIDEvent(
            timestamp  = now,
            similarity = similarity,
            bbox       = bbox,
            frame      = frame_bgr.copy(),
            match_type = match_type,
            confidence = confidence,
        )
        self._last_event = event
        log.success(
            f"ReID [{match_type}] sim={similarity:.3f} "
            f"confidence={confidence} bbox={bbox}"
        )
        if self._callback:
            self._callback(event)
        return event

    # ── Debug overlay ─────────────────────────────────────────────────────────

    def draw_debug(
        self,
        frame_bgr:     np.ndarray,
        event:         Optional[ReIDEvent] = None,
        person_bboxes: Optional[list[tuple[int, int, int, int]]] = None,
    ):
        """
        Blue box  — detected person, no match
        Green box — matched person (enrolled)
        Top-right — enrollment status
        Top-left  — ReID score when triggered
        """
        active_event = event or self._last_event

        for person in self._last_persons:
            x, y, w, h = person["bbox"]
            matched = (
                active_event is not None
                and active_event.match_type == "body_crop"
                and person["bbox"] == active_event.bbox
            )
            color = (0, 220, 0) if matched else (220, 100, 0)
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame_bgr,
                f"{person['score']:.2f}",
                (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
            )

        with self._lock:
            buffered = len(self._state.embeddings_buffer)
            enrolled = self._state.is_enrolled

        status = f"ENROLLED" if enrolled else f"ENROLLING {buffered}/{ENROLL_N_FRAMES}"
        color  = (0, 220, 0) if enrolled else (0, 165, 255)
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        (tw, _), bl = cv2.getTextSize(status, font, fs, th)
        m = 10
        status_x = max(m, frame_bgr.shape[1] - tw - m)
        status_y = max(25, m + bl)
        cv2.putText(
            frame_bgr, status,
            (status_x, status_y),
            font, fs, color, th,
        )

        if active_event:
            label = (
                f"MATCH [{active_event.match_type}] "
                f"{active_event.similarity:.2f} [{active_event.confidence}]"
            )
            label_font, label_fs, label_th = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            (lw, lh), lbl_bl = cv2.getTextSize(label, label_font, label_fs, label_th)
            label_x = max(m, frame_bgr.shape[1] - lw - m)
            label_y = status_y + lh + lbl_bl + 8
            cv2.putText(
                frame_bgr, label,
                (label_x, label_y), label_font, label_fs, (0, 220, 0), label_th,
            )


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def build_reid_service(
    yolo_path:         str   = _DEFAULT_YOLO_PATH,
    embedder_path:     str   = _DEFAULT_REID_PATH,
    use_trt:           bool  = False,
    similarity_thresh: float = BODY_SIMILARITY_THRESH,
    on_reid_callback:  Optional[Callable[[ReIDEvent], None]] = None,
) -> ReIDService:
    """
    Factory for the body-detector ReID service.

    Args:
        yolo_path:         Path to yolov8n.onnx
        embedder_path:     Path to resnet50_market1501_aicity156.onnx
        use_trt:           Reserved for future TRT path
        similarity_thresh: Cosine similarity gate for body crops
        on_reid_callback:  Called with ReIDEvent when enrolled person detected
    """
    detector = YOLOPersonDetector(yolo_path)
    embedder = ResNet50ReIDEmbedder(embedder_path)

    return ReIDService(
        person_detector    = detector,
        embedder           = embedder,
        similarity_thresh  = similarity_thresh,
        on_reid_callback   = on_reid_callback,
    )
