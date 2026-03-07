"""
Person Re-Identification (ReID) Service
========================================
Runs on NVIDIA Jetson Orin Nano.

Pipeline
--------
1.  AprilTag detected  →  enroll primary person (capture face embedding).
2.  Continuous stream  →  detect faces, compare embeddings.
3.  Match above threshold  →  emit ReID event (triggers snapshot in main_video).

Models
------
- Face detection  : YuNet  (OpenCV DNN, INT8-ready, ~2 MB)
- Face embedding  : MobileFaceNet via ONNX / TensorRT  (~4 MB)

Both are lightweight, INT8/FP16 TensorRT-convertible, and well-suited for
the Orin Nano's 1024-core Ampere GPU + 2x DLA cores.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Optional TensorRT path ──────────────────────────────────────────────────
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit          # noqa: F401  initialises CUDA context
    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False

from config.logger import Logger

log = Logger("[video][reid]")

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Paths – override via environment or settings dict
_DIR = Path(__file__).parent
_DEFAULT_YUNET_MODEL   = str(_DIR / "face_detection_yunet_2023mar.onnx")
_DEFAULT_EMBEDDER_ONNX = str(_DIR / "mobilefacenet.onnx")
_DEFAULT_EMBEDDER_TRT  = str(_DIR / "mobilefacenet_fp16.engine")

# Detection
YUNET_INPUT_SIZE  = (320, 320)   # (W, H) – reduce to (160,160) for more speed
YUNET_SCORE_THRESH = 0.6
YUNET_NMS_THRESH   = 0.3
YUNET_TOP_K        = 5000

# Embedding / matching
EMBED_INPUT_SIZE  = (112, 112)   # MobileFaceNet canonical input
SIMILARITY_THRESH = 0.55         # cosine similarity  (0→no match, 1→identical)
ENROLL_N_FRAMES   = 5            # average N embeddings during enrolment
REID_COOLDOWN_S   = 2.0          # seconds between consecutive ReID triggers

# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN ENTITIES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ReIDEvent:
    """Emitted when the primary person is re-identified."""
    timestamp: float
    similarity: float
    bbox: tuple[int, int, int, int]   # x, y, w, h  in pixel coords
    frame: Optional[np.ndarray] = None


@dataclass
class EnrollmentState:
    """Mutable state for the primary-person embedding."""
    embedding: Optional[np.ndarray] = None
    enrolled_at: float = 0.0
    embeddings_buffer: list = field(default_factory=list)

    @property
    def is_enrolled(self) -> bool:
        return self.embedding is not None

    def add_candidate(self, emb: np.ndarray) -> bool:
        """Buffer one candidate embedding; finalise after ENROLL_N_FRAMES."""
        self.embeddings_buffer.append(emb)
        if len(self.embeddings_buffer) >= ENROLL_N_FRAMES:
            stack = np.stack(self.embeddings_buffer, axis=0)
            mean  = stack.mean(axis=0)
            self.embedding    = mean / (np.linalg.norm(mean) + 1e-6)
            self.enrolled_at  = time.time()
            self.embeddings_buffer.clear()
            log.success(f"Primary person enrolled (avg of {ENROLL_N_FRAMES} embeddings)")
            return True
        return False

    def reset(self):
        self.embedding = None
        self.enrolled_at = 0.0
        self.embeddings_buffer.clear()
        log.info("Enrollment state reset")


# ═══════════════════════════════════════════════════════════════════════════
# FACE DETECTOR  –  YuNet (OpenCV DNN)
# ═══════════════════════════════════════════════════════════════════════════

class YuNetFaceDetector:
    """
    Thin wrapper around cv2.FaceDetectorYN (YuNet).

    Why YuNet?
    - Built into OpenCV ≥ 4.8, zero extra dependency.
    - ONNX model ≈ 2 MB, runs at ~60 FPS on Jetson Orin Nano at 320×320.
    - Outputs 5-point landmarks (eyes, nose, mouth) used for alignment.
    - Supports cv2 DNN CUDA backend  →  GPU accelerated out of the box.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_YUNET_MODEL,
        input_size: tuple[int, int] = YUNET_INPUT_SIZE,
        score_thresh: float = YUNET_SCORE_THRESH,
        nms_thresh: float   = YUNET_NMS_THRESH,
        top_k: int          = YUNET_TOP_K,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}\n"
                "Download: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet"
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
          { 'bbox': (x,y,w,h), 'score': float, 'landmarks': np.ndarray(5,2) }
        """
        h, w = frame_bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame_bgr)

        results = []
        if faces is None:
            return results

        for face in faces:
            # YuNet row: [x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y,
            #             rm_x, rm_y, lm_x, lm_y, score]
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            score = float(face[-1])
            lm    = face[4:14].reshape(5, 2)
            results.append({"bbox": (x, y, fw, fh), "score": score, "landmarks": lm})

        return results


# ═══════════════════════════════════════════════════════════════════════════
# FACE ALIGNMENT  (5-point similarity transform → 112×112)
# ═══════════════════════════════════════════════════════════════════════════

# Standard 112×112 reference landmarks (ArcFace canonical)
_REF_LANDMARKS_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(frame_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Warp-align a detected face to 112×112 using a 5-point similarity transform.
    Aligned crops drastically improve embedding quality.
    """
    src_pts = landmarks.astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src_pts, _REF_LANDMARKS_112, method=cv2.LMEDS)
    if M is None:
        # Fallback: simple crop + resize
        x, y, w, h = _landmarks_to_bbox(landmarks)
        crop = frame_bgr[max(y, 0): y + h, max(x, 0): x + w]
        return cv2.resize(crop, EMBED_INPUT_SIZE)
    aligned = cv2.warpAffine(frame_bgr, M, EMBED_INPUT_SIZE, flags=cv2.INTER_LINEAR)
    return aligned


def _landmarks_to_bbox(lm: np.ndarray) -> tuple[int, int, int, int]:
    x_min, y_min = int(lm[:, 0].min()), int(lm[:, 1].min())
    x_max, y_max = int(lm[:, 0].max()), int(lm[:, 1].max())
    margin = max(x_max - x_min, y_max - y_min) // 2
    return x_min - margin, y_min - margin, (x_max - x_min) + 2 * margin, (y_max - y_min) + 2 * margin


# ═══════════════════════════════════════════════════════════════════════════
# FACE EMBEDDER  –  MobileFaceNet  (ONNX or TensorRT)
# ═══════════════════════════════════════════════════════════════════════════

class ONNXFaceEmbedder:
    """
    MobileFaceNet via ONNX Runtime with CUDA execution provider.

    Why MobileFaceNet?
    - 4 MB model, designed for mobile / edge.
    - 512-dim embedding, competitive accuracy on LFW (~99.5 %).
    - ONNX export available; trivially convertible to TensorRT FP16.
    - ~5 ms per face on Orin Nano with ORT-GPU.
    """

    def __init__(self, model_path: str = _DEFAULT_EMBEDDER_ONNX):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime-gpu is required: pip install onnxruntime-gpu")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Embedder model not found: {model_path}\n"
                "Download: https://github.com/cavalleria/cavaface.pytorch  or\n"
                "          https://github.com/deepinsight/insightface (buffalo_sc package)"
            )

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2      # keep CPU threads low on Jetson

        self._session  = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        self._inp_name = self._session.get_inputs()[0].name
        log.success(f"MobileFaceNet ONNX embedder loaded  ({self._session.get_providers()[0]})")

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """
        Args:
            aligned_bgr: 112×112 BGR uint8 face crop (already aligned).
        Returns:
            L2-normalised 512-dim float32 embedding.
        """
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0                     # ArcFace normalisation
        blob = rgb.transpose(2, 0, 1)[np.newaxis]        # (1,3,112,112)
        emb  = self._session.run(None, {self._inp_name: blob})[0][0]
        return emb / (np.linalg.norm(emb) + 1e-6)


class TRTFaceEmbedder:
    """
    MobileFaceNet via TensorRT FP16 engine.
    Provides ~2× speedup over ONNX Runtime on the Orin Nano.
    Use convert_to_tensorrt() below to build the engine once.
    """

    def __init__(self, engine_path: str = _DEFAULT_EMBEDDER_TRT):
        if not _TRT_AVAILABLE:
            raise RuntimeError("TensorRT / pycuda not installed")
        if not Path(engine_path).exists():
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()
        self._stream  = cuda.Stream()

        # Allocate pinned host memory + device memory
        binding = self._engine.get_binding_shape(0)
        size_in  = int(np.prod(binding)) * np.dtype(np.float32).itemsize
        binding_out = self._engine.get_binding_shape(1)
        size_out = int(np.prod(binding_out)) * np.dtype(np.float32).itemsize

        self._h_in  = cuda.pagelocked_empty(int(np.prod(binding)),  np.float32)
        self._h_out = cuda.pagelocked_empty(int(np.prod(binding_out)), np.float32)
        self._d_in  = cuda.mem_alloc(size_in)
        self._d_out = cuda.mem_alloc(size_out)
        log.success("MobileFaceNet TensorRT FP16 engine loaded")

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        blob = rgb.transpose(2, 0, 1).ravel()         # (3×112×112,)
        np.copyto(self._h_in, blob)

        cuda.memcpy_htod_async(self._d_in, self._h_in, self._stream)
        self._context.execute_async_v2(
            bindings=[int(self._d_in), int(self._d_out)],
            stream_handle=self._stream.handle,
        )
        cuda.memcpy_dtoh_async(self._h_out, self._d_out, self._stream)
        self._stream.synchronize()

        emb = self._h_out.copy()
        return emb / (np.linalg.norm(emb) + 1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised embeddings → [-1, 1]."""
    return float(np.dot(a, b))


# ═══════════════════════════════════════════════════════════════════════════
# REID SERVICE
# ═══════════════════════════════════════════════════════════════════════════

class ReIDService:
    """
    Stateful service that:
      - Enrolls the primary person on AprilTag detection.
      - Continuously matches faces in the live stream.
      - Calls `on_reid_callback(ReIDEvent)` when a match is found.

    Thread-safe: enroll() and process_frame() can be called from the
    main video loop; the callback is invoked on the same thread.
    """

    def __init__(
        self,
        detector: YuNetFaceDetector,
        embedder: ONNXFaceEmbedder | TRTFaceEmbedder,
        similarity_thresh: float = SIMILARITY_THRESH,
        reid_cooldown_s: float   = REID_COOLDOWN_S,
        on_reid_callback=None,
    ):
        self._detector   = detector
        self._embedder   = embedder
        self._thresh     = similarity_thresh
        self._cooldown   = reid_cooldown_s
        self._callback   = on_reid_callback

        self._state      = EnrollmentState()
        self._last_reid  = 0.0
        self._lock       = threading.Lock()

    # ── Public API ──────────────────────────────────────────────────────────

    def enroll_from_frame(self, frame_bgr: np.ndarray) -> bool:
        """
        Called when an AprilTag is detected.
        Detects the largest face in the frame and buffers its embedding.
        Returns True when enrolment is finalised (after ENROLL_N_FRAMES calls).
        """
        faces = self._detector.detect(frame_bgr)
        if not faces:
            log.warning("Enrolment: no face detected in frame")
            return False

        # Choose the largest face (most likely the person carrying the tag)
        face   = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
        aligned = align_face(frame_bgr, face["landmarks"])
        emb    = self._embedder.embed(aligned)

        with self._lock:
            finalised = self._state.add_candidate(emb)
        return finalised

    def reset_enrollment(self):
        with self._lock:
            self._state.reset()

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[ReIDEvent]:
        """
        Run face detection + matching on one frame.
        Returns a ReIDEvent if the primary person is found, else None.
        """
        with self._lock:
            if not self._state.is_enrolled:
                return None
            ref_emb = self._state.embedding

        faces = self._detector.detect(frame_bgr)
        if not faces:
            return None

        best_sim  = -1.0
        best_face = None

        for face in faces:
            aligned = align_face(frame_bgr, face["landmarks"])
            emb     = self._embedder.embed(aligned)
            sim     = cosine_similarity(ref_emb, emb)
            if sim > best_sim:
                best_sim  = sim
                best_face = face

        if best_sim >= self._thresh:
            now = time.time()
            if (now - self._last_reid) < self._cooldown:
                return None          # suppress within cooldown window

            self._last_reid = now
            event = ReIDEvent(
                timestamp  = now,
                similarity = best_sim,
                bbox       = best_face["bbox"],
                frame      = frame_bgr.copy(),
            )
            log.success(f"ReID match  sim={best_sim:.3f}  bbox={best_face['bbox']}")

            if self._callback:
                self._callback(event)

            return event

        return None

    def draw_debug(self, frame_bgr: np.ndarray, event: Optional[ReIDEvent] = None):
        """Overlay detection results on frame (for development / monitoring)."""
        faces = self._detector.detect(frame_bgr)
        for face in faces:
            x, y, w, h = face["bbox"]
            color = (0, 255, 0) if (event and face["bbox"] == event.bbox) else (200, 200, 200)
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


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def build_reid_service(
    yunet_path:   str  = _DEFAULT_YUNET_MODEL,
    embedder_path: str = _DEFAULT_EMBEDDER_ONNX,
    use_trt: bool      = False,
    similarity_thresh: float = SIMILARITY_THRESH,
    on_reid_callback         = None,
) -> ReIDService:
    """
    Convenience factory.  Pass use_trt=True once you have a built engine.
    Falls back to ONNX if TRT is unavailable.
    """
    detector = YuNetFaceDetector(yunet_path)

    if use_trt and _TRT_AVAILABLE:
        trt_path = embedder_path.replace(".onnx", "_fp16.engine")
        embedder = TRTFaceEmbedder(trt_path)
    else:
        embedder = ONNXFaceEmbedder(embedder_path)

    return ReIDService(
        detector           = detector,
        embedder           = embedder,
        similarity_thresh  = similarity_thresh,
        on_reid_callback   = on_reid_callback,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TENSORRT CONVERSION UTILITY
# ═══════════════════════════════════════════════════════════════════════════

def convert_to_tensorrt(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_gb: int = 1,
):
    """
    One-time offline conversion of the MobileFaceNet ONNX model to a
    TensorRT FP16 engine.  Run this on the Jetson before deployment.

    Usage:
        python -c "from reid_service import convert_to_tensorrt; \\
                   convert_to_tensorrt('models/mobilefacenet.onnx', \\
                                       'models/mobilefacenet_fp16.engine')"
    """
    if not _TRT_AVAILABLE:
        raise RuntimeError("TensorRT not available")

    logger  = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log.error(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        log.info("TRT build: FP16 enabled")

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TRT engine build failed")

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    log.success(f"TRT engine saved → {engine_path}")
