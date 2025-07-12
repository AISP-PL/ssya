from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np  # type: ignore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ssya.controllers.dataset_manager import DatasetManager
from ssya.controllers.sam2_wrapper import Sam2Runner
from ssya.models.detection import Detection
from ssya.widgets.image_viewer import ImageViewer  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


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
        self.viewer.show_image(
            img,
            dets,
            self.selected_mask,
            selected_detection=self.selected_detection,
            sim_threshold=self.slider.value() / 100.0,
        )

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
            img,
            self.dm.image_detections(self.cur_img_idx),
            self.selected_mask,
            selected_detection=det,
            sim_threshold=self.slider.value() / 100.0,
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

    def on_clear_filter(self):
        self.files_list.clear()
        self.files_list.addItems(self.dm.images)
        self.files_list.setCurrentRow(self.cur_img_idx)


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
