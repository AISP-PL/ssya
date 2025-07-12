#!/usr/bin/env python
"""
Prototype GUI for browsing a dataset of images annotated in YOLO‑format, with basic
integration hooks for Segment‑Anything‑Model‑2 (SAM2) to create masks & embeddings,
plus a simple similarity search workflow.

The goal is to deliver a **single‑file** proof‑of‑concept that the team can extend.
It deliberately keeps heavy lifting (e.g. CUDA SAM2 inference, FAISS indexing)
behind clearly‑marked TODO stubs so the GUI remains responsive even without the
full ML stack installed.

Run it either via CLI:
    python main.py -i /path/to/dataset
or simply:
    python main.py
and pick a folder from the file‑dialog.

Requirements (matching the provided pyproject):
  * Python ≥ 3.11
  * PyQt5
  * OpenCV‑Python
  * NumPy (Faiss optional)
  * yaya‑tools (already on the path via pyproject)
  * sam2‑python (future) — optional, otherwise falls back to a stub

Dataset layout expected by ``yaya_tools``:
   images: *.jpg / *.png alongside *.txt (YOLO v5)

Author: AISP / ChatGPT prototype – 2025‑07‑12
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore
import requests
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- External helpers supplied by yaya_tools ---------------------------------
from yaya_tools.helpers.dataset import (
    get_images_annotated,
    load_directory_images_annotatations,
)  # type: ignore

logger = logging.getLogger("ssya_gui")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# -----------------------------------------------------------------------------
# 1. Simple datamodels
# -----------------------------------------------------------------------------


@dataclass
class Detection:
    """Single YOLO bounding‑box + (lazy) embedding placeholder."""

    class_id: int
    bbox_norm: tuple[float, float, float, float]  # x_center, y_center, w, h in [0,1]
    image_path: Path
    embedding: np.ndarray | None = None  # computed lazily by SAM2

    # Pixel bbox helper (lazily computed per resolution)
    def bbox_pixels(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        xc, yc, w, h = self.bbox_norm
        w_px, h_px = int(w * img_w), int(h * img_h)
        x1 = int((xc * img_w) - w_px / 2)
        y1 = int((yc * img_h) - h_px / 2)
        return x1, y1, w_px, h_px


# -----------------------------------------------------------------------------
# 2. SAM2 minimal wrapper (stub‑friendly)
# -----------------------------------------------------------------------------


class Sam2Runner:
    """Wraps Segment‑Anything‑Model‑2. Falls back to a fast stub if missing."""

    def __init__(self) -> None:
        """Initialize the SAM2 predictor if available, otherwise use a stub."""
        model_path = "zoo/sam2_tiny.pth"  # Default model path
        # Check: Model not exists, download from URL to zoo
        if not os.path.exists(model_path):
            url_download = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
            # Ensure target directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            logger.info("Downloading SAM2 model from %s to %s", url_download, model_path)
            resp = requests.get(url_download, allow_redirects=True)
            resp.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(resp.content)

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        config_path = "configs/sam2.1/sam2.1_hiera_t.yaml"  # plik *.yaml
        model = build_sam2(config_path, model_path).to("cuda").eval()

        # SAM2ImagePredictor expects a model instance
        ckpt = Path(model_path)
        logger.info("Loading SAM2 from %s", ckpt)
        self._predictor = SAM2ImagePredictor(model)

    def mask_and_embed(
        self, image_bgr: np.ndarray, bbox_px: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        image_rgb = image_bgr[:, :, ::-1].copy()
        self._predictor.set_image(image_rgb)

        x, y, w, h = bbox_px
        masks, scores, logits = self._predictor.predict(
            box=np.array([x, y, x + w, y + h]),
            multimask_output=False,
            return_logits=True,
        )

        mask_hr = masks[0]  # pełna rozdzielczość
        logit_map = logits[0]  # 256×256

        mask_lr = cv2.resize(
            mask_hr.astype(np.uint8),
            logit_map.shape[::-1],  # (W,H)
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        emb = np.array([logit_map[mask_lr].mean()], dtype=np.float32)
        return mask_hr, emb


# -----------------------------------------------------------------------------
# 3. Dataset utilities
# -----------------------------------------------------------------------------


class DatasetManager:
    """Loads the dataset via yaya_tools and exposes detections & images."""

    def __init__(self, root: Path):
        self.root = root
        if not root.exists():
            raise FileNotFoundError(root)

        self._images_ann: dict[str, str | None] = load_directory_images_annotatations(str(root))
        self._images = list(self._images_ann.keys())
        self._annotated: list[str] = get_images_annotated(self._images_ann)

        # Preload size cache to avoid repeat I/O
        self._size_cache: dict[str, tuple[int, int]] = {}
        logger.info("Dataset loaded: %d images, %d with annotations", len(self._images), len(self._annotated))

        # Pre‑parse detections for each annotated image
        self._detections: dict[str, list[Detection]] = {}
        for img_path in self._annotated:
            self._detections[img_path] = self._parse_yolo_annotations(img_path)

    # ---------------------------------------------------------------------

    def _parse_yolo_annotations(self, img_path: str) -> list[Detection]:
        """Parse YOLO annotations for a given image path."""
        ann_path = self._images_ann[img_path]
        if not ann_path:
            return []

        detections: list[Detection] = []
        with open(self.root / ann_path, encoding="utf-8") as f:
            for line in f:
                class_id_s, xc_s, yc_s, w_s, h_s = line.strip().split()
                detections.append(
                    Detection(
                        class_id=int(class_id_s),
                        bbox_norm=(float(xc_s), float(yc_s), float(w_s), float(h_s)),
                        image_path=Path(img_path),
                    )
                )
        return detections

    # ---------------------------------------------------------------------

    def image(self, idx: int) -> np.ndarray:
        """Load an image by index. Raises ValueError if the image cannot be loaded."""
        path = self._images[idx]
        img = cv2.imread(self.root / path)
        if img is None:
            raise ValueError(f"Could not load image: {path}")

        return img

    def detections(self, idx: int) -> list[Detection]:
        path = self._images[idx]
        return self._detections.get(path, [])

    def image_count(self) -> int:
        return len(self._images)


# -----------------------------------------------------------------------------
# 4. GUI widgets
# -----------------------------------------------------------------------------


class ImageViewer(QWidget):
    """Displays image with bounding boxes & optional masks."""

    def __init__(self) -> None:
        super().__init__()

        # QLabel przejmie całe dostępne miejsce i sam będzie skalował pixmapę
        self.label = QLabel(alignment=Qt.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setScaledContents(True)  # <-- kluczowe
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

        # Maximum width and height : Set 1920x1080 - 200
        self.setMaximumSize(1920 - 200, 1080 - 200)

        self._current_img: np.ndarray | None = None
        self._current_masks: list[np.ndarray] = []

    # ------------------------------------------------------------------

    def set_image(
        self,
        img_bgr: np.ndarray,
        detections: list[Detection],
        masks: list[np.ndarray] | None = None,
    ) -> None:
        """Render image + boxes (+ masks if provided)."""

        display = img_bgr.copy()
        h, w, _ = display.shape

        # Draw masks first (semi‑transparent blue).
        if masks:
            for mask in masks:
                display[mask > 0] = (display[mask > 0] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)

        # Draw boxes (green).
        for det in detections:
            x, y, bw, bh = det.bbox_pixels(w, h)
            cv2.rectangle(display, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(display, str(det.class_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Convert to Qt pixmap
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))


# -----------------------------------------------------------------------------
# 5. Main Window
# -----------------------------------------------------------------------------


class MainWindow(QWidget):
    """Top‑level application managing dataset navigation & similarity workflow."""

    def __init__(self, dataset: DatasetManager):
        super().__init__()
        self.setWindowTitle("SSYA – Similarity Search with YOLO & SAM2 (prototype)")
        self.dataset = dataset
        self.sam2 = Sam2Runner()

        # Nav controls
        self.prev_btn = QPushButton("← Prev")
        self.next_btn = QPushButton("Next →")
        self.info_label = QLabel()

        # Detection list
        self.det_list = QListWidget()
        self.det_list.setFixedWidth(180)

        # Threshold slider
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)

        # Action buttons
        self.find_similar_btn = QPushButton("🔍  Znajdź podobne")
        self.name_objects_btn = QPushButton("🏷️  Nazwij obiekty")

        # Image viewer
        self.viewer = ImageViewer()

        # --- Layout -----------------------------------------------------
        side_layout = QVBoxLayout()
        side_layout.addWidget(self.det_list)
        side_layout.addWidget(QLabel("Similarity threshold"))
        side_layout.addWidget(self.threshold_slider)
        side_layout.addWidget(self.find_similar_btn)
        side_layout.addWidget(self.name_objects_btn)
        side_layout.addStretch()

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.info_label)
        nav_layout.addWidget(self.next_btn)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.viewer)
        main_layout.addLayout(side_layout)

        root_layout = QVBoxLayout(self)
        root_layout.addLayout(main_layout)
        root_layout.addLayout(nav_layout)

        # --- State ------------------------------------------------------
        self.cur_idx = 0
        self._masks: dict[int, list[np.ndarray]] = {}  # per image idx

        # --- Signals ----------------------------------------------------
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn.clicked.connect(lambda: self._step(1))
        self.det_list.currentRowChanged.connect(self._on_detection_clicked)
        self.find_similar_btn.clicked.connect(self._find_similar)

        # Init display
        self._refresh()

    # ------------------------------------------------------------------

    def _step(self, delta: int) -> None:
        self.cur_idx = (self.cur_idx + delta) % self.dataset.image_count()
        self._refresh()

    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        img = self.dataset.image(self.cur_idx)
        dets = self.dataset.detections(self.cur_idx)
        masks = self._masks.get(self.cur_idx)
        self.viewer.set_image(img, dets, masks)

        # Update list widget
        self.det_list.clear()
        for i, d in enumerate(dets):
            self.det_list.addItem(f"#{i}  cls={d.class_id}")
        self.info_label.setText(f"Image {self.cur_idx + 1}/{self.dataset.image_count()}")

    # ------------------------------------------------------------------

    def _on_detection_clicked(self, row: int) -> None:
        if row < 0:
            return
        det = self.dataset.detections(self.cur_idx)[row]
        img = self.dataset.image(self.cur_idx)
        mask, embedding = self.sam2.mask_and_embed(img, det.bbox_pixels(img.shape[1], img.shape[0]))
        det.embedding = embedding
        self._masks.setdefault(self.cur_idx, []).append(mask)
        self.viewer.set_image(img, self.dataset.detections(self.cur_idx), self._masks[self.cur_idx])

    # ------------------------------------------------------------------

    def _find_similar(self) -> None:
        """Very naive similarity search across all cached embeddings."""
        # Gather reference embedding (latest clicked)
        ref_det: Detection | None = None
        if self.det_list.currentRow() >= 0:
            ref_det = self.dataset.detections(self.cur_idx)[self.det_list.currentRow()]
        if ref_det is None or ref_det.embedding is None:
            QMessageBox.warning(self, "Brak embeddingu", "Najpierw wybierz detekcję i wygeneruj maskę.")
            return

        threshold = self.threshold_slider.value() / 100.0
        similar: list[tuple[int, int, float]] = []  # (img_idx, det_idx, score)

        for img_idx in range(self.dataset.image_count()):
            for det_idx, det in enumerate(self.dataset.detections(img_idx)):
                if det.embedding is None:
                    continue  # skip un‑computed
                score = cosine_similarity(ref_det.embedding, det.embedding)
                if score >= threshold:
                    similar.append((img_idx, det_idx, score))

        similar.sort(key=lambda t: t[2], reverse=True)
        if not similar:
            QMessageBox.information(self, "Nic nie znaleziono", "Brak podobnych obiektów powyżej progu.")
            return

        # Jump to first similar result (after current)
        img_idx, det_idx, _ = similar[0]
        self.cur_idx = img_idx
        self._refresh()
        self.det_list.setCurrentRow(det_idx)


# -----------------------------------------------------------------------------
# 6. Utility functions
# -----------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# -----------------------------------------------------------------------------
# 7. Entry point
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSYA dataset browser (prototype)")
    parser.add_argument(
        "-i",
        "--dataset_path",
        type=Path,
        help="Path to the dataset root folder (images + YOLO txts).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Qt app must be created before DatasetManager if we show a file‑dialog
    app = QApplication(sys.argv)

    dataset_path: Path | None = args.dataset_path
    if dataset_path is None:
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        if dlg.exec_():
            selected = dlg.selectedFiles()
            if selected:
                dataset_path = Path(selected[0])
    if dataset_path is None:
        logger.error("No dataset path provided. Exiting.")
        sys.exit(1)

    try:
        dataset = DatasetManager(dataset_path)
    except Exception as e:
        QMessageBox.critical(None, "Dataset load error", str(e))
        sys.exit(1)

    # Check : Dataset is empty
    if dataset.image_count() == 0:
        QMessageBox.warning(None, "Empty dataset", "The selected dataset contains no images.")
        sys.exit(1)

    win = MainWindow(dataset)
    win.resize(1200, 800)
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
