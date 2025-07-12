import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from yaya_tools.helpers.dataset import load_directory_images_annotatations

from ssya.controllers.features_index import FeatureIndex
from ssya.controllers.sam2_wrapper import Sam2Runner
from ssya.models.detection import Detection

logger = logging.getLogger(__name__)


class DatasetManager:
    """Loads dataset, detections, builds/loads feature index."""

    def __init__(self, root: Path):
        self.root = root
        ann_map = load_directory_images_annotatations(str(root))
        self.images: list[str] = list(ann_map.keys())
        self.ann_map = ann_map
        self.detections: dict[str, list[Detection]] = {}
        for img_idx, img_path in enumerate(self.images):
            if not ann_map[img_path]:
                self.detections[img_path] = []
                continue
            with open(root / ann_map[img_path]) as f:
                lines = [l.split() for l in f]
            self.detections[img_path] = [
                Detection(int(cls), (float(xc), float(yc), float(w), float(h)), img_idx) for cls, xc, yc, w, h in lines
            ]
        logger.info("Dataset: %d images (%d with annotations)", len(self.images), len(self.detections))

        # Build or load feature index ---------------------------------
        self.index_path = root / "features.pickle"
        if self.index_path.exists():
            logger.info("Loading cached features …")
            self.fidx = FeatureIndex.load(self.index_path)
        else:
            self.fidx = FeatureIndex()
            self._build_index()
            self.fidx.save(self.index_path)

        # Detections : Update with embeddings from the index
        for img_path, dets in self.detections.items():
            self.detections[img_path] = self.fidx.get_features(dets)

    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        sam = Sam2Runner()
        logger.info("Building feature index (SAM2)…")
        for img_idx, img_path in enumerate(tqdm(self.images, desc="Images")):
            img = cv2.imread(str(self.root / img_path))
            if img is None:
                continue
            for det_idx, det in enumerate(self.detections[img_path]):
                mask, emb = sam.mask_and_embed(img, det.bbox_pixels(img.shape[1], img.shape[0]))
                det.embedding = emb
                self.fidx.add(img_idx, det_idx, emb)

    # ------------------------------------------------------------------

    # Convenience helpers used by GUI ----------------------------------
    def image(self, idx: int) -> np.ndarray:
        """Get image at index `idx`."""
        return cv2.imread(str(self.root / self.images[idx]))

    def image_detections(self, idx: int) -> list[Detection]:
        """Get detections for the image at index `idx`."""
        return self.detections[self.images[idx]]

    def image_count(self) -> int:
        """Get the number of images in the dataset."""
        return len(self.images)
