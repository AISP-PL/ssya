import os
import threading

import cv2
import faiss
import numpy as np
import requests
import torch
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from segment_anything import SamPredictor, sam_model_registry
from yaya_tools.helpers.dataset import load_directory_images_annotatations


class MainWindow(QMainWindow):
    def __init__(self, dataset_path: str):
        super().__init__()
        self.dataset_path = dataset_path
        # SAM and similarity state
        checkpoint = os.path.join(self.dataset_path, "sam_vit_h.pth")
        # If checkpoint missing, download from official source
        if not os.path.exists(checkpoint):
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            with open(checkpoint, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        self.sam_model = sam_model_registry["vit_h"](checkpoint=checkpoint)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_model.to(device)
        self.sam_predictor = SamPredictor(self.sam_model)
        self.selected_embedding = None
        self.db_entries = []  # list of dicts with keys: image_idx, bbox, embedding
        self.similar_scores = []  # list of (entry idx, score)
        # Load images and annotations
        self.all_images = load_directory_images_annotatations(dataset_path)
        self.image_paths = list(self.all_images.keys())
        self.current_index = 0
        # UI setup
        self.init_ui()
        self.update_display()

    def init_ui(self):
        # Widgets
        self.image_label = QLabel(alignment=Qt.AlignCenter)
        self.det_list = QListWidget()
        self.det_list.itemClicked.connect(self.on_detection_selected)
        self.prev_btn = QPushButton("< Poprzednie")
        self.next_btn = QPushButton("Następne >")
        self.find_btn = QPushButton("Znajdź podobne")
        self.find_btn.clicked.connect(self.find_similar)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.valueChanged.connect(self.apply_threshold)
        self.name_btn = QPushButton("Nazwij obiekty")
        self.name_btn.clicked.connect(self.rename_objects)
        # Connections
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        # Layouts
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("Próg:"))
        info_layout.addWidget(self.threshold_slider)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.find_btn)
        btn_layout.addWidget(self.name_btn)
        side_layout = QVBoxLayout()
        side_layout.addWidget(self.det_list)
        side_layout.addLayout(info_layout)
        side_layout.addLayout(btn_layout)
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label, 3)
        main_layout.addLayout(side_layout, 1)
        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)
        self.setWindowTitle("SSYA - Przeglądarka anotacji")

    def update_display(self):
        # Load image
        img_path = self.image_paths[self.current_index]
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        # Draw bounding boxes from YOLO .txt
        ann_path = self.all_images[img_path]
        if ann_path and os.path.exists(ann_path):
            with open(ann_path) as f:
                for line in f:
                    cls, x_c, y_c, bw, bh = line.strip().split()
                    # Convert to pixel coords
                    xc, yc, bwf, bhf = map(float, (x_c, y_c, bw, bh))
                    x1 = int((xc - bwf / 2) * w)
                    y1 = int((yc - bhf / 2) * h)
                    x2 = int((xc + bwf / 2) * w)
                    y2 = int((yc + bhf / 2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Convert to QImage
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img)
        self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # Update list
        self.det_list.clear()
        if ann_path and os.path.exists(ann_path):
            with open(ann_path) as f:
                for line in f:
                    self.det_list.addItem(line.strip())
        # Info: image X of Y
        self.statusBar().showMessage(
            f"Zdjęcie {self.current_index + 1} z {len(self.image_paths)}: {os.path.basename(img_path)}"
        )

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.update_display()

    def on_detection_selected(self, item):
        # Parse selected detection and get mask & embedding
        ann = item.text().split()
        _, x_c, y_c, bw, bh = map(float, ann)
        img = cv2.imread(self.image_paths[self.current_index])
        h, w = img.shape[:2]
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)
        # SAM prediction
        self.sam_predictor.set_image(img)
        masks, _, _ = self.sam_predictor.predict(box=np.array([x1, y1, x2, y2]), multimask_output=False)
        mask = masks[0]
        # compute embedding: mean of image_embeddings over mask
        emb = self.sam_predictor.image_embeddings[:, mask].mean(dim=1).cpu().numpy()
        self.selected_embedding = emb
        # visualize mask overlay
        overlay = img.copy()
        overlay[mask] = (0, 0, 255)
        alpha = 0.5
        img_overlay = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        qt_img = QImage(cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB).data, w, h, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def find_similar(self):
        # compute embeddings for entire dataset in background
        def worker():
            self.db_entries = []
            for idx, img_path in enumerate(self.image_paths):
                img = cv2.imread(img_path)
                ann_path = self.all_images[img_path]
                if not ann_path:
                    continue
                with open(ann_path) as f:
                    for line in f:
                        cls, x_c, y_c, bw, bh = map(float, line.strip().split())
                        h, w = img.shape[:2]
                        x1 = int((x_c - bw / 2) * w)
                        y1 = int((y_c - bh / 2) * h)
                        x2 = int((x_c + bw / 2) * w)
                        y2 = int((y_c + bh / 2) * h)
                        self.sam_predictor.set_image(img)
                        masks, _, _ = self.sam_predictor.predict(box=np.array([x1, y1, x2, y2]), multimask_output=False)
                        mask = masks[0]
                        emb = self.sam_predictor.image_embeddings[:, mask].mean(dim=1).cpu().numpy()
                        self.db_entries.append({"image_idx": idx, "bbox": (x1, y1, x2, y2), "emb": emb})
            # build FAISS index
            dim = self.db_entries[0]["emb"].shape[0]
            index = faiss.IndexFlatL2(dim)
            embs = np.stack([e["emb"] for e in self.db_entries])
            index.add(embs)
            distances, indices = index.search(self.selected_embedding.reshape(1, -1), len(self.db_entries))
            # store similarity scores
            self.similar_scores = [(idx_val, float(distances[0][j])) for j, idx_val in enumerate(indices[0])]
            # apply threshold filter
            self.apply_threshold()

        threading.Thread(target=worker, daemon=True).start()

    def apply_threshold(self):
        if not self.similar_scores:
            return
        thresh = self.threshold_slider.value() / 100
        filtered_idxs = {self.db_entries[i]["image_idx"] for i, dist in self.similar_scores if dist <= (1 - thresh)}
        self.image_paths = [p for i, p in enumerate(self.image_paths) if i in filtered_idxs]
        self.current_index = 0
        self.update_display()

    def rename_objects(self):
        # Dialog to rename classes in current image
        ann_path = self.all_images[self.image_paths[self.current_index]]
        if not ann_path:
            return
        dialog = QDialog(self)
        form = QFormLayout(dialog)
        edits = []
        with open(ann_path) as f:
            lines = [line.strip().split() for line in f]
        for vals in lines:
            cls, *coords = vals
            edit = QLineEdit(cls)
            form.addRow(f"Old class {cls}:", edit)
            edits.append(edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        form.addRow(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dialog_accepted = dialog.exec_() == QDialog.Accepted
        if dialog_accepted:
            with open(ann_path, "w") as f:
                for edit, vals in zip(edits, lines, strict=True):
                    f.write(f"{edit.text()} {' '.join(vals[1:])}\n")
        dialog.deleteLater()
