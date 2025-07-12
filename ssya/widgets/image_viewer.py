from __future__ import annotations

import logging

import cv2  # type: ignore
import numpy as np  # type: ignore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ssya.helpers.metrics import cosine_similarity
from ssya.models.detection import Detection  # type: ignore

logger = logging.getLogger(__name__)


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
                cv2.putText(disp, f"{sim:.2f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if sim < sim_threshold:
                    continue

                cv2.rectangle(disp, (x, y), (x + bw, y + bh), (0, 255, 0), thickness=2)

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, QImage.Format_RGB888)
        self.lbl.setPixmap(QPixmap.fromImage(qimg))
