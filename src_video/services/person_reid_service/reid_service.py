from __future__ import annotations

import time
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import torchreid


REID_MODEL_NAME       = "osnet_x0_25"          # Lightest OSNet — good for Jetson Orin
REID_INPUT_SIZE       = (256, 128)
ENROLLMENT_FRAMES     = 5                       # Frames to average for enrollment embedding
PATIENT_MATCH_THRESH  = 0.70                    # Cosine similarity threshold for patient match
PERSON_MATCH_THRESH   = 0.65                    # Slightly looser for general session re-association


def load_reid_model(device: str = "cuda:0") -> torch.nn.Module:
    """
    Load a lightweight OSNet x0.25 model for embedding extraction.
    Downloads pretrained weights automatically on first run.
    """
    model = torchreid.models.build_model(
        name=REID_MODEL_NAME,
        num_classes=1000,
        pretrained=True,
    )
    model = model.to(device)
    model.eval()
    return model


def extract_embedding(
    model: torch.nn.Module,
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    device: str = "cuda:0",
) -> Optional[torch.Tensor]:
    """
    Crop person from frame using bbox, preprocess, and extract L2-normalised embedding.

    Args:
        model:  Loaded OSNet model.
        frame:  Full BGR frame from camera.
        bbox:   (x1, y1, x2, y2) bounding box.
        device: Torch device string.

    Returns:
        1-D normalised embedding tensor, or None if crop is invalid.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Guard against out-of-bounds crops
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # Preprocess: resize → RGB → normalise → tensor
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, (REID_INPUT_SIZE[1], REID_INPUT_SIZE[0]))

    tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(tensor)

    # L2 normalise so cosine similarity == dot product
    embedding = F.normalize(embedding, p=2, dim=1).squeeze(0).cpu()
    return embedding


def average_embeddings(embeddings: list[torch.Tensor]) -> torch.Tensor:
    """Average a list of embeddings and re-normalise."""
    stacked = torch.stack(embeddings, dim=0)
    mean_emb = stacked.mean(dim=0)
    return F.normalize(mean_emb, p=2, dim=0)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D normalised tensors."""
    return float(torch.dot(a, b).clamp(-1.0, 1.0))


# ─────────────────────────────────────────────
# Enrollment Buffer
# ─────────────────────────────────────────────

class EnrollmentBuffer:
    """
    Accumulates embeddings over N frames after AprilTag detection
    to produce a robust averaged enrollment embedding.
    """

    def __init__(self, target_frames: int = ENROLLMENT_FRAMES):
        self.target_frames = target_frames
        self._buffer: list[torch.Tensor] = []
        self.enrolled_embedding: Optional[torch.Tensor] = None
        self.is_complete = False

    def add(self, embedding: torch.Tensor) -> bool:
        """
        Add an embedding frame. Returns True when enrollment is complete.
        """
        if self.is_complete:
            return True

        self._buffer.append(embedding)

        if len(self._buffer) >= self.target_frames:
            self.enrolled_embedding = average_embeddings(self._buffer)
            self.is_complete = True
            return True

        return False

    @property
    def progress(self) -> str:
        return f"{len(self._buffer)}/{self.target_frames}"

    def reset(self):
        self._buffer.clear()
        self.enrolled_embedding = None
        self.is_complete = False


# ─────────────────────────────────────────────
# Session Person Registry
# ─────────────────────────────────────────────

class SessionRegistry:
    """
    Maintains a session-scoped registry of all persons seen during the stream.

    Each person gets a stable session_id (e.g. P001, P002...) that persists
    even if DeepOCSORT assigns a new track ID on re-entry.

    Flow:
      - New track ID seen → extract embedding → check against registry
      - Match found       → re-associate with existing session_id
      - No match          → register as new person, assign new session_id
    """

    def __init__(self, match_threshold: float = PERSON_MATCH_THRESH):
        self.threshold = match_threshold
        self._registry: dict[str, dict] = {}          # session_id → {embedding, track_id, last_seen, ...}
        self._track_to_session: dict[int, str] = {}   # current track_id → session_id
        self._counter = 0

    def _new_session_id(self) -> str:
        self._counter += 1
        return f"P{self._counter:03d}"

    def get_session_id(self, track_id: int) -> Optional[str]:
        """Return the session_id for a currently active track, if known."""
        return self._track_to_session.get(track_id)

    def register_or_match(
        self,
        track_id: int,
        embedding: torch.Tensor,
        is_patient: bool = False,
        forced_session_id: Optional[str] = None,
    ) -> str:
        """
        Given a new track_id and its embedding, either:
          - Re-associate to an existing session_id via Re-ID matching, or
          - Create a new session entry.

        Args:
            track_id:          DeepOCSORT track ID.
            embedding:         Extracted OSNet embedding.
            is_patient:        Flag to mark this as the enrolled patient.
            forced_session_id: If provided, force-assign this session_id
                               (used when patient is first enrolled via AprilTag).

        Returns:
            The session_id assigned to this track.
        """
        # Already mapped
        if track_id in self._track_to_session:
            sid = self._track_to_session[track_id]
            self._registry[sid]["last_seen"] = time.time()
            self._registry[sid]["track_id"] = track_id
            return sid

        # Try to match against existing registry entries
        best_sid   = None
        best_score = -1.0

        for sid, entry in self._registry.items():
            score = cosine_similarity(embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_sid = sid

        if forced_session_id is not None:
            # AprilTag enrollment: force-assign regardless of match
            sid = forced_session_id
            if sid not in self._registry:
                self._registry[sid] = {}
            self._registry[sid].update({
                "embedding":  embedding,
                "track_id":   track_id,
                "last_seen":  time.time(),
                "is_patient": is_patient,
            })
            self._track_to_session[track_id] = sid
            return sid

        if best_sid is not None and best_score >= self.threshold:
            # Re-association: returning person
            self._registry[best_sid]["track_id"] = track_id
            self._registry[best_sid]["last_seen"] = time.time()
            self._registry[best_sid]["embedding"] = average_embeddings(
                [self._registry[best_sid]["embedding"], embedding]
            )  # Online update: blend old + new embedding
            self._track_to_session[track_id] = best_sid
            return best_sid

        # New person
        sid = self._new_session_id()
        self._registry[sid] = {
            "embedding":  embedding,
            "track_id":   track_id,
            "last_seen":  time.time(),
            "is_patient": is_patient,
        }
        self._track_to_session[track_id] = sid
        return sid

    def remove_track(self, track_id: int):
        """Called when a track is lost — unmaps track_id but keeps session entry."""
        self._track_to_session.pop(track_id, None)

    def get_patient_session_id(self) -> Optional[str]:
        """Return the session_id of the enrolled patient, if any."""
        for sid, entry in self._registry.items():
            if entry.get("is_patient"):
                return sid
        return None

    def get_patient_embedding(self) -> Optional[torch.Tensor]:
        """Return the current embedding for the enrolled patient."""
        for entry in self._registry.values():
            if entry.get("is_patient"):
                return entry["embedding"]
        return None

    def is_patient_track(self, track_id: int) -> bool:
        """Return True if a given track_id maps to the patient's session."""
        sid = self._track_to_session.get(track_id)
        if sid is None:
            return False
        return self._registry.get(sid, {}).get("is_patient", False)

    def all_sessions(self) -> dict:
        """Return a snapshot of the full session registry."""
        return {
            sid: {
                "track_id":   e["track_id"],
                "last_seen":  e["last_seen"],
                "is_patient": e.get("is_patient", False),
            }
            for sid, e in self._registry.items()
        }

    def sync_active_tracks(self, active_track_ids: set[int]):
        """
        Remove stale track_id → session mappings for tracks no longer reported
        by the tracker. Call once per frame after tracker.update().
        """
        stale = [tid for tid in self._track_to_session if tid not in active_track_ids]
        for tid in stale:
            self.remove_track(tid)

