from dataclasses import dataclass

import numpy as np


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
