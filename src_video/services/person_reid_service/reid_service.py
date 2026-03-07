from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config.logger import Logger

log = Logger("[video][reid]")

# ═══════════════════════════════════════════════════════════════════════════
# MODEL PATHS  (same directory as this script)
# ═══════════════════════════════════════════════════════════════════════════

_DIR = Path(__file__).parent
_DEFAULT_YUNET_PATH = str(_DIR / "face_detection_yunet_2023mar.onnx")
_DEFAULT_REID_PATH  = str(_DIR / "resnet50_market1501_aicity156.onnx")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# YuNet — enrollment gating
YUNET_INPUT_SIZE   = (320, 320)   # (W, H) — reduce to (160,160) for more speed
YUNET_SCORE_THRESH = 0.6
YUNET_NMS_THRESH   = 0.3
YUNET_TOP_K        = 5000

# resnet50_market1501 — body ReID
REID_INPUT_SIZE = (128, 256)      # (W, H) canonical input for market1501 models
REID_MEAN       = np.array([0.485, 0.456, 0.406], dtype=np.float32)
REID_STD        = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Matching
SIMILARITY_THRESH = 0.65          # cosine similarity — body ReID needs slightly
                                  # higher threshold than face (more variation)
ENROLL_N_FRAMES   = 5             # average N body embeddings during enrollment
REID_COOLDOWN_S   = 2.0           # seconds between consecutive ReID triggers

# Body crop expansion — when we have a face bbox from YuNet during enrollment,
# expand it downward to approximate a full-body region for the ReID embedder.
# These are rough multipliers; tune based on your camera distance.
BODY_EXPAND_TOP    = 0.4          # expand upward  (add headroom above face)
BODY_EXPAND_BOTTOM = 4.5          # expand downward (torso + legs below face)
BODY_EXPAND_SIDES  = 0.8          # expand left/right

# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN ENTITIES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ReIDEvent:
    """Emitted when the primary person is re-identified."""
    timestamp:  float
    similarity: float
    bbox:       tuple[int, int, int, int]   # x, y, w, h in pixel coords
    frame:      Optional[np.ndarray] = None


@dataclass
class EnrollmentState:
    """Mutable state for the stored primary-person body embedding."""
    embedding:         Optional[np.ndarray] = None
    enrolled_at:       float = 0.0
    embeddings_buffer: list  = field(default_factory=list)

    @property
    def is_enrolled(self) -> bool:
        return self.embedding is not None

    def add_candidate(self, emb: np.ndarray) -> bool:
        """
        Buffer one candidate embedding.
        Finalises enrollment by averaging after ENROLL_N_FRAMES calls.
        Returns True when enrollment is complete.
        """
        self.embeddings_buffer.append(emb)
        if len(self.embeddings_buffer) >= ENROLL_N_FRAMES:
            stack = np.stack(self.embeddings_buffer, axis=0)
            mean  = stack.mean(axis=0)
            self.embedding   = mean / (np.linalg.norm(mean) + 1e-6)
            self.enrolled_at = time.time()
            self.embeddings_buffer.clear()
            log.success(
                f"Primary person enrolled "
                f"(averaged {ENROLL_N_FRAMES} body embeddings)"
            )
            return True
        log.info(
            f"Enrollment: {len(self.embeddings_buffer)}/{ENROLL_N_FRAMES} frames buffered"
        )
        return False

    def reset(self):
        self.embedding = None
        self.enrolled_at = 0.0
        self.embeddings_buffer.clear()
        log.info("Enrollment state reset")


# ═══════════════════════════════════════════════════════════════════════════
# YUNET FACE DETECTOR  —  enrollment gatekeeper
# ═══════════════════════════════════════════════════════════════════════════

class YuNetFaceDetector:
    """
    Thin wrapper around cv2.FaceDetectorYN (YuNet).
    Used only during enrollment to confirm the correct person is in frame.
    Runs on OpenCV's CUDA DNN backend — no additional installs needed.
    """

    def __init__(
        self,
        model_path:   str   = _DEFAULT_YUNET_PATH,
        input_size:   tuple = YUNET_INPUT_SIZE,
        score_thresh: float = YUNET_SCORE_THRESH,
        nms_thresh:   float = YUNET_NMS_THRESH,
        top_k:        int   = YUNET_TOP_K,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}\n"
                "Download:\n"
                "  wget https://github.com/opencv/opencv_zoo/raw/main/models/"
                "face_detection_yunet/face_detection_yunet_2023mar.onnx"
            )

        self._detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            input_size,
            score_thresh,
            nms_thresh,
            top_k,
            cv2.dnn.DNN_BACKEND_CUDA,
            cv2.dnn.DNN_TARGET_CUDA,
        )
        self._input_size = input_size
        log.success("YuNet face detector loaded (CUDA backend)")

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Returns list of dicts:
            { 'bbox': (x, y, w, h), 'score': float, 'landmarks': np.ndarray(5,2) }
        """
        h, w = frame_bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame_bgr)

        results = []
        if faces is None:
            return results

        for face in faces:
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            score = float(face[-1])
            lm    = face[4:14].reshape(5, 2)
            results.append({
                "bbox":      (x, y, fw, fh),
                "score":     score,
                "landmarks": lm,
            })

        return results


# ═══════════════════════════════════════════════════════════════════════════
# BODY CROP UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def expand_face_to_body(
    frame_bgr: np.ndarray,
    face_bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Expands a face bounding box to approximate a full-body region.
    Used during enrollment when we only have a face detection.
    Returns (body_crop, body_bbox).

    The expansion multipliers in BODY_EXPAND_* should be tuned to your
    camera height and typical subject distance.
    """
    fh, fw = frame_bgr.shape[:2]
    x, y, w, h = face_bbox

    top    = int(y - h * BODY_EXPAND_TOP)
    bottom = int(y + h * BODY_EXPAND_BOTTOM)
    left   = int(x - w * BODY_EXPAND_SIDES)
    right  = int(x + w + w * BODY_EXPAND_SIDES)

    # Clamp to frame boundaries
    top    = max(top,    0)
    bottom = min(bottom, fh)
    left   = max(left,   0)
    right  = min(right,  fw)

    crop = frame_bgr[top:bottom, left:right]
    return crop, (left, top, right - left, bottom - top)


def preprocess_body_crop(crop: np.ndarray) -> np.ndarray:
    """
    Preprocess a body crop for resnet50_market1501:
      - Resize to 128x256 (W x H)
      - Convert BGR to RGB
      - Normalise with ImageNet mean/std
      - Return NCHW float32 blob
    """
    resized = cv2.resize(crop, REID_INPUT_SIZE)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normed  = (rgb - REID_MEAN) / REID_STD
    blob    = normed.transpose(2, 0, 1)[np.newaxis]
    return blob.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# BODY REID EMBEDDER  —  resnet50_market1501_aicity156
# ═══════════════════════════════════════════════════════════════════════════

class ResNet50ReIDEmbedder:
    """
    resnet50_market1501_aicity156 via ONNX Runtime with CUDA EP.

    Why this model?
    - Trained on Market-1501 + AIC156 — purpose-built for person ReID
    - 2048-dim embedding captures whole-body appearance features
    - Works from any angle, even when face is not visible
    - ONNX export from NVIDIA TAO Toolkit — TensorRT-compatible
    - ~15ms per crop on Orin Nano with ORT-GPU

    Input  : (1, 3, 256, 128)  float32  ImageNet-normalised RGB
    Output : (1, 2048)         float32  L2-normalised feature vector
    """

    def __init__(self, model_path: str = _DEFAULT_REID_PATH):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime-gpu is required.\n"
                "Install on Jetson:\n"
                "  sudo pip install onnxruntime-gpu "
                "--index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-deps"
            )

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"ReID model not found: {model_path}\n"
                "Download resnet50_market1501_aicity156.onnx from NVIDIA NGC:\n"
                "  https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/reidentificationnet"
            )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2

        self._session  = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=providers,
        )
        self._inp_name = self._session.get_inputs()[0].name
        log.success(
            f"ResNet50 ReID embedder loaded  "
            f"({self._session.get_providers()[0]})"
        )

    def embed(self, body_crop_bgr: np.ndarray) -> np.ndarray:
        """
        Args:
            body_crop_bgr: BGR body crop, any size (resized internally).
        Returns:
            L2-normalised 2048-dim float32 embedding.
        """
        if body_crop_bgr.size == 0:
            log.warning("Empty body crop passed to embedder — skipping")
            return np.zeros(2048, dtype=np.float32)

        blob = preprocess_body_crop(body_crop_bgr)
        emb  = self._session.run(None, {self._inp_name: blob})[0][0]
        return emb / (np.linalg.norm(emb) + 1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised embeddings -> [-1, 1]."""
    return float(np.dot(a, b))


# ═══════════════════════════════════════════════════════════════════════════
# REID SERVICE
# ═══════════════════════════════════════════════════════════════════════════

class ReIDService:
    """
    Stateful dual-model ReID service.

    Enrollment phase  (AprilTag visible)
    -------------------------------------
    1. YuNet confirms a face is present in frame (right person check).
    2. Face bbox is expanded to a full-body crop.
    3. ResNet50 embeds the body crop.
    4. Steps 1-3 repeat for ENROLL_N_FRAMES, then embeddings are averaged
       into a single stable reference vector.

    Tracking phase  (marker-free)
    ------------------------------
    1. Your existing run_detection produces person bboxes each frame.
       Pass those bboxes + the frame into process_frame().
    2. Each detected person is embedded by ResNet50.
    3. Cosine similarity against reference triggers on_reid_callback
       when a match is found above threshold.

    If no external bboxes are provided, process_frame() falls back to
    using the full frame as a single body crop.

    Thread-safe: enroll_from_frame() and process_frame() can be called
    from the main video loop thread.
    """

    def __init__(
        self,
        face_detector:     YuNetFaceDetector,
        body_embedder:     ResNet50ReIDEmbedder,
        similarity_thresh: float = SIMILARITY_THRESH,
        reid_cooldown_s:   float = REID_COOLDOWN_S,
        on_reid_callback         = None,
    ):
        self._face_detector = face_detector
        self._body_embedder = body_embedder
        self._thresh        = similarity_thresh
        self._cooldown      = reid_cooldown_s
        self._callback      = on_reid_callback

        self._state         = EnrollmentState()
        self._last_reid     = 0.0
        self._lock          = threading.Lock()

    # ── Public API ──────────────────────────────────────────────────────────

    def enroll_from_frame(self, frame_bgr: np.ndarray) -> bool:
        """
        Called when an AprilTag is detected.

        1. YuNet checks for a face — skips frame if none found.
        2. Largest face bbox is expanded to a full-body crop.
        3. ResNet50 embeds the body crop and buffers it.
        4. Returns True when ENROLL_N_FRAMES have been buffered and
           enrollment is finalised.
        """
        faces = self._face_detector.detect(frame_bgr)
        if not faces:
            log.warning("Enrollment: no face detected — skipping frame")
            return False

        # Use the largest detected face as the primary subject
        face = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
        body_crop, _ = expand_face_to_body(frame_bgr, face["bbox"])

        if body_crop.size == 0:
            log.warning("Enrollment: body crop is empty — skipping frame")
            return False

        emb = self._body_embedder.embed(body_crop)

        with self._lock:
            finalised = self._state.add_candidate(emb)

        return finalised

    def reset_enrollment(self):
        """Reset stored embedding — next AprilTag detection re-enrolls."""
        with self._lock:
            self._state.reset()

    def process_frame(
        self,
        frame_bgr:     np.ndarray,
        person_bboxes: Optional[list[tuple[int, int, int, int]]] = None,
    ) -> Optional[ReIDEvent]:
        """
        Match detected persons against the enrolled reference.

        Args:
            frame_bgr:     Full camera frame (BGR).
            person_bboxes: Optional list of (x, y, w, h) bboxes from
                           run_detection. If provided, each person crop
                           is embedded individually for accurate matching.
                           If None, the full frame is used as a single crop.

        Returns:
            ReIDEvent if primary person matched, else None.
        """
        with self._lock:
            if not self._state.is_enrolled:
                return None
            ref_emb = self._state.embedding

        # Build list of (crop, bbox) pairs to compare
        candidates = []

        if person_bboxes:
            fh, fw = frame_bgr.shape[:2]
            for (x, y, w, h) in person_bboxes:
                x1 = max(x, 0);       y1 = max(y, 0)
                x2 = min(x + w, fw);  y2 = min(y + h, fh)
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    candidates.append((crop, (x, y, w, h)))
        else:
            # Fallback: full frame as single crop
            candidates.append((
                frame_bgr,
                (0, 0, frame_bgr.shape[1], frame_bgr.shape[0])
            ))

        if not candidates:
            return None

        best_sim  = -1.0
        best_bbox = None

        for crop, bbox in candidates:
            emb = self._body_embedder.embed(crop)
            sim = cosine_similarity(ref_emb, emb)
            if sim > best_sim:
                best_sim  = sim
                best_bbox = bbox

        if best_sim >= self._thresh:
            now = time.time()
            if (now - self._last_reid) < self._cooldown:
                return None     # within cooldown window

            self._last_reid = now
            event = ReIDEvent(
                timestamp  = now,
                similarity = best_sim,
                bbox       = best_bbox,
                frame      = frame_bgr.copy(),
            )
            log.success(f"ReID match  sim={best_sim:.3f}  bbox={best_bbox}")

            if self._callback:
                self._callback(event)

            return event

        return None

    def draw_debug(
        self,
        frame_bgr:     np.ndarray,
        event:         Optional[ReIDEvent] = None,
        person_bboxes: Optional[list[tuple[int, int, int, int]]] = None,
    ):
        """
        Draw bounding boxes and ReID status onto the frame for monitoring.
        - Grey box  : detected person, no match
        - Green box : matched primary person
        - 'REID 0.XX' text on match
        - Enrollment progress bar at top of frame
        """
        boxes = person_bboxes or []

        for (x, y, w, h) in boxes:
            color = (
                (0, 255, 0)
                if (event and (x, y, w, h) == event.bbox)
                else (200, 200, 200)
            )
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)

        if event:
            cv2.putText(
                frame_bgr,
                f"REID {event.similarity:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        # Enrollment status indicator
        with self._lock:
            buffered = len(self._state.embeddings_buffer)
            enrolled = self._state.is_enrolled

        status = "ENROLLED" if enrolled else f"ENROLLING {buffered}/{ENROLL_N_FRAMES}"
        color  = (0, 255, 0) if enrolled else (0, 165, 255)
        cv2.putText(
            frame_bgr,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def build_reid_service(
    yunet_path:        str   = _DEFAULT_YUNET_PATH,
    embedder_path:     str   = _DEFAULT_REID_PATH,
    use_trt:           bool  = False,   # reserved for future TRT path
    similarity_thresh: float = SIMILARITY_THRESH,
    on_reid_callback         = None,
) -> ReIDService:
    """
    Convenience factory — interface identical to v1 so main_video_with_reid.py
    requires no changes.
    use_trt is accepted for forward compatibility but currently unused.
    """
    face_detector = YuNetFaceDetector(yunet_path)
    body_embedder = ResNet50ReIDEmbedder(embedder_path)

    return ReIDService(
        face_detector      = face_detector,
        body_embedder      = body_embedder,
        similarity_thresh  = similarity_thresh,
        on_reid_callback   = on_reid_callback,
    )
