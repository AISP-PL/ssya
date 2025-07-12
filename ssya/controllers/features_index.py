from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ssya.helpers.metrics import cosine_similarity
from ssya.models.detection import Detection  # type: ignore

logger = logging.getLogger(__name__)


class FeatureIndex:
    """Persistent RAM index: list of (image_idx, det_idx, embedding)."""

    def __init__(self, entries: list[dict[str, Any]] | None = None):
        """Initialize with existing entries or empty."""
        if entries is not None:
            self.entries = entries
        else:
            self.entries: list[dict[str, Any]] = []

    def add(self, image_idx: int, det_idx: int, emb: np.ndarray):
        self.entries.append({"image_idx": image_idx, "det_idx": det_idx, "emb": emb})

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self.entries, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_features(self, detections: list[Detection]) -> list[Detection]:
        """Update detections list with embeddings from the index."""
        for det in detections:
            if det.embedding is None:
                for e in self.entries:
                    if e["image_idx"] == det.image_idx and e["det_idx"] == det.class_id:
                        det.embedding = e["emb"]
                        break

        return detections

    @classmethod
    def load(cls, path: Path) -> FeatureIndex:
        with open(path, "rb") as f:
            entries = pickle.load(f)

        return cls(entries)

    def get_similar_images(self, ref_emb: np.ndarray, thresh: float) -> set[int]:
        """Find images with at least one detection above the threshold."""
        imgs: set[int] = set()
        for e in self.entries:
            if cosine_similarity(ref_emb, e["emb"]) >= thresh:
                imgs.add(e["image_idx"])

        return imgs
