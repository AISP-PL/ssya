from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore
import requests
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

logger = logging.getLogger(__name__)


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
