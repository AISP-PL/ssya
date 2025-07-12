#!/usr/bin/env python
"""
Improved SSYA prototype.

Changes vs. previous version
----------------------------
1. **Offline feature index** — at start‑up every detection gets a SAM2
   embedding. Index (list of dicts) is cached to *features.pickle* inside
   the dataset directory. Re‑use if present.
2. **tqdm progress bars** while building the index.
3. **Two vertical lists** on the right:
   * top → list of image filenames
   * bottom → list of detections in currently selected image
4. **Filtering** – "Find similar" now hides any *images* that do not
   contain at least one detection above the threshold.
5. **Clear filter** button resets view to full dataset.

This is still a **single file** ready to run: `python ssya_gui_rewrite.py -i /path/to/dataset`.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2  # type: ignore
import numpy as np  # type: ignore
import requests
import torch
import torch.nn.functional as F
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm.auto import tqdm

# --- External helpers from yaya_tools ---------------------------------------
from yaya_tools.helpers.dataset import (
    load_directory_images_annotatations,
)  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def gem_pooling(features: torch.Tensor, mask: torch.Tensor, p: float = 3.0):
    """
    GeM pooling z maską: features (B, C, H, W), mask (B, 1, H, W) – bool/int.
    Zwraca (B, C)
    """
    eps = 1e-6
    masked = features * mask  # (B, C, H, W)
    pooled = F.avg_pool2d(masked.clamp(min=eps).pow(p), kernel_size=masked.shape[-2:])  # (B, C, 1, 1)
    pooled = pooled.pow(1.0 / p).squeeze(-1).squeeze(-1)
    # uwzględnij liczbę aktywnych pikseli
    denom = mask.flatten(2).sum(-1).clamp(min=1e-6)  # (B,1)
    pooled = pooled / denom
    return F.normalize(pooled, dim=-1)


# ----------------------------------------------------------------------------
# 1. Basic dataclasses
# ----------------------------------------------------------------------------


@dataclass
class Detection:
    class_id: int
    bbox_norm: tuple[float, float, float, float]
    image_idx: int  # index in DatasetManager.images
    embedding: np.ndarray | None = None

    def bbox_pixels(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        xc, yc, w, h = self.bbox_norm
        w_px, h_px = int(w * img_w), int(h * img_h)
        x1 = int((xc * img_w) - w_px / 2)
        y1 = int((yc * img_h) - h_px / 2)
        return x1, y1, w_px, h_px


# ----------------------------------------------------------------------------
# 2. SAM2 wrapper
# ----------------------------------------------------------------------------


class Sam2Runner:
    """Light wrapper that exposes mask + embedding for a bbox."""

    _instance = None  # singleton for reuse

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model()
        return cls._instance

    # ------------------------------------------------------------------

    def _init_model(self):
        model_path = "zoo/sam2_tiny.pth"
        if not os.path.exists(model_path):
            url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading SAM2 weights …")
            with requests.get(url, stream=True) as r, open(model_path, "wb") as f:
                for chunk in r.iter_content(1 << 14):
                    f.write(chunk)
        cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu"
        model = build_sam2(cfg, model_path).to(device).eval()
        self._predictor = SAM2ImagePredictor(model)
        self.device = device

    # ------------------------------------------------------------------

    def mask_and_embed(self, img_bgr: np.ndarray, box_px: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
        img_rgb = img_bgr[:, :, ::-1].copy()
        self._predictor.set_image(img_rgb)  # ← tutaj SAM2 wylicza embedding

        # ---------- segmentacja ----------
        masks, _, _ = self._predictor.predict(
            box=np.array([box_px[0], box_px[1], box_px[0] + box_px[2], box_px[1] + box_px[3]]),
            multimask_output=False,
            return_logits=False,
        )
        mask_hr = masks[0]  # (H, W) bool

        # ---------- mapa cech ----------
        feat_container = getattr(self._predictor, "_features", None)
        if feat_container is None:
            raise RuntimeError("Brak _features w predictorze — sprawdź wersję biblioteki")

        # słownik → weź 'image_embed'
        feat_map = feat_container.get("image_embed", None) if isinstance(feat_container, dict) else feat_container

        if feat_map is None or not torch.is_tensor(feat_map):
            raise RuntimeError("Nie znalazłem tensora z mapą cech w _features")

        B, C, h, w = feat_map.shape

        B, C, h, w = feat_map.shape

        # dopasuj maskę do rozdzielczości cech
        mask_lr = cv2.resize(mask_hr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        mask_t = torch.from_numpy(mask_lr).to(feat_map.device).view(1, 1, h, w)

        emb_t = gem_pooling(feat_map, mask_t, p=3.0)  # z poprzedniej odpowiedzi
        emb = emb_t.cpu().numpy()[0]  # (C,)

        return mask_hr, emb


# ----------------------------------------------------------------------------
# 3. Dataset + FeatureIndex
# ----------------------------------------------------------------------------


class FeatureIndex:
    """Persistent RAM index: list of (image_idx, det_idx, embedding)."""

    def __init__(self):
        self.entries: list[dict[str, Any]] = []

    def add(self, image_idx: int, det_idx: int, emb: np.ndarray):
        self.entries.append({"image_idx": image_idx, "det_idx": det_idx, "emb": emb})

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump([{**e, "emb": e["emb"].astype(np.float32)} for e in self.entries], f)

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
            raw = pickle.load(f)
        fi = cls()
        for e in raw:
            fi.entries.append({**e, "emb": e["emb"]})
        return fi

    # ------------------------------------------------------------------

    def similar_images(self, ref_emb: np.ndarray, thresh: float) -> set[int]:
        """Find images with at least one detection above the threshold."""
        imgs: set[int] = set()
        for e in self.entries:
            if cosine_similarity(ref_emb, e["emb"]) >= thresh:
                imgs.add(e["image_idx"])

        return imgs


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


# ----------------------------------------------------------------------------
# 4. GUI widgets
# ----------------------------------------------------------------------------


class ImageViewer(QWidget):
    """Widget to display images with detections and masks."""

    def __init__(self):
        super().__init__()
        self.lbl = QLabel(alignment=Qt.AlignCenter)
        self.lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl.setScaledContents(True)
        # Fixed with and height set, 1920 - 200 , 1080 - 200
        self.lbl.setFixedWidth(1620)
        self.lbl.setFixedHeight(880)

        QVBoxLayout(self).addWidget(self.lbl)

    def show_image(
        self,
        img_bgr: np.ndarray,
        dets: list[Detection],
        masks: list[np.ndarray],
        selected_detection: Detection | None = None,
        sim_threshold: float = 0.5,
    ):
        """Display image with detections and masks."""
        if img_bgr is None:
            return

        disp = img_bgr.copy()
        h, w, _ = disp.shape

        for m in masks:
            disp[m > 0] = (disp[m > 0] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)

        for d in dets:
            x, y, bw, bh = d.bbox_pixels(w, h)
            cv2.rectangle(disp, (x, y), (x + bw, y + bh), (255, 0, 0), 1)
            cv2.putText(disp, str(d.class_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        if selected_detection is not None:
            for det in dets:
                sim = cosine_similarity(selected_detection.embedding, det.embedding)
                x, y, bw, bh = det.bbox_pixels(w, h)
                cv2.putText(disp, f"sim={sim:.2f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                if sim < sim_threshold:
                    continue

                cv2.rectangle(disp, (x, y), (x + bw, y + bh), (0, 255, 0), thickness=2)

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, QImage.Format_RGB888)
        self.lbl.setPixmap(QPixmap.fromImage(qimg))


# ----------------------------------------------------------------------------
# 5. Main Window
# ----------------------------------------------------------------------------


class MainWindow(QWidget):
    def __init__(self, dm: DatasetManager):
        super().__init__()
        self.dm = dm
        self.setWindowTitle("SSYA – embeddings cache & filtering")

        # widgets ------------------------------------------------------
        self.viewer = ImageViewer()

        self.files_list = QListWidget()
        self.files_list.addItems(dm.images)
        self.dets_list = QListWidget()

        self.slider = QSlider(Qt.Horizontal, minimum=0, maximum=100, value=50)
        btn_similar = QPushButton("🔍 Find similar")
        btn_clear = QPushButton("❌ Clear filter")

        # layout -------------------------------------------------------
        side = QVBoxLayout()
        side.addWidget(QLabel("Images"))
        side.addWidget(self.files_list)
        side.addWidget(QLabel("Detections"))
        side.addWidget(self.dets_list)
        side.addWidget(QLabel("Threshold"))
        side.addWidget(self.slider)
        side.addWidget(btn_similar)
        side.addWidget(btn_clear)
        side.addStretch()

        splitter = QSplitter()
        splitter.addWidget(self.viewer)
        side_widget = QWidget()
        side_widget.setLayout(side)
        splitter.addWidget(side_widget)
        splitter.setSizes([800, 300])

        QVBoxLayout(self).addWidget(splitter)

        # state --------------------------------------------------------
        self.cur_img_idx = 0
        self.selected_mask: list[np.ndarray] = []
        self.selected_detection: Detection | None = None

        # signals ------------------------------------------------------
        self.files_list.currentRowChanged.connect(self.on_file_select)
        self.dets_list.currentRowChanged.connect(self.on_det_select)
        btn_similar.clicked.connect(self.on_find_similar)
        btn_clear.clicked.connect(self.on_clear_filter)

        self.display_image(0)

    # ------------------------------------------------------------------

    def display_image(self, idx: int) -> None:
        """Display image and its detections."""
        self.cur_img_idx = idx
        img = self.dm.image(idx)
        dets = self.dm.image_detections(idx)
        self.dets_list.clear()
        for i, d in enumerate(dets):
            self.dets_list.addItem(f"#{i} cls={d.class_id}")
        self.selected_mask = []
        self.viewer.show_image(img, dets, self.selected_mask, selected_detection=self.selected_detection)

    # ------------------------------------------------------------------

    def on_file_select(self, row: int):
        if row >= 0:
            self.display_image(row)

    # ------------------------------------------------------------------

    def on_det_select(self, row: int):
        if row < 0:
            return
        det = self.dm.image_detections(self.cur_img_idx)[row]
        img = self.dm.image(self.cur_img_idx)
        if det.embedding is None:
            sam = Sam2Runner()
            mask, emb = sam.mask_and_embed(img, det.bbox_pixels(img.shape[1], img.shape[0]))
            det.embedding = emb
        else:
            # build fake mask for viz only
            x, y, bw, bh = det.bbox_pixels(img.shape[1], img.shape[0])
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask[y : y + bh, x : x + bw] = 1
        self.selected_mask = [mask]
        self.viewer.show_image(
            img, self.dm.image_detections(self.cur_img_idx), self.selected_mask, selected_detection=det
        )

    # ------------------------------------------------------------------

    def on_find_similar(self) -> None:
        """Find images with at least one detection above the threshold."""
        row = self.dets_list.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Select", "Select a detection first")
            return

        det = self.dm.image_detections(self.cur_img_idx)[row]
        if det.embedding is None:
            QMessageBox.warning(self, "No embedding", "Embedding missing – click detection again")
            return

        self.selected_detection = det

        thresh = self.slider.value() / 100.0
        keep = self.dm.fidx.similar_images(det.embedding, thresh)

        self.files_list.clear()
        self.files_list.addItems([self.dm.images[i] for i in sorted(keep)])

        if keep:
            self.files_list.setCurrentRow(0)

    # ------------------------------------------------------------------

    def on_clear_filter(self):
        self.files_list.clear()
        self.files_list.addItems(self.dm.images)
        self.files_list.setCurrentRow(self.cur_img_idx)


# ----------------------------------------------------------------------------
# 6. Utils
# ----------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ----------------------------------------------------------------------------
# 7. CLI entry
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset_path", type=Path)
    args = ap.parse_args()

    app = QApplication(sys.argv)

    ds_path = args.dataset_path
    if ds_path is None:
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec_():
            sel = dlg.selectedFiles()
            if sel:
                ds_path = Path(sel[0])
    if ds_path is None:
        sys.exit("Dataset path missing")

    dm = DatasetManager(ds_path)
    win = MainWindow(dm)
    win.resize(1400, 800)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
