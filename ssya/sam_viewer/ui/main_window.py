"""Main window UI for SAM Viewer application."""

import logging
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QSlider, QSplitter, QGroupBox, QScrollArea,
    QMessageBox, QProgressBar, QStatusBar, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen

from ..modules.yolo_parser import YOLOParser, YOLODetection
from ..modules.image_navigator import ImageNavigator

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main window for SAM Viewer application."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize main window.
        
        Args:
            dataset_path: Path to dataset directory
        """
        super().__init__()
        
        self.dataset_path = Path(dataset_path)
        self.yolo_parser: Optional[YOLOParser] = None
        self.image_navigator: Optional[ImageNavigator] = None
        self.selected_detection: Optional[int] = None
        
        # Initialize UI
        self.init_ui()
        
        # Load dataset
        self.load_dataset()
        
        # Setup keyboard shortcuts
        self.setup_shortcuts()
    
    def init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("SAM Viewer - YOLO Annotation Viewer with SAM2 Integration")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel (image display)
        self.create_image_panel(splitter)
        
        # Right panel (controls and detection list)
        self.create_control_panel(splitter)
        
        # Set splitter proportions
        splitter.setSizes([1000, 400])
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def create_image_panel(self, parent):
        """Create image display panel."""
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        
        # Image info label
        self.image_info_label = QLabel("No image loaded")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        self.image_info_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        image_layout.addWidget(self.image_info_label)
        
        # Image display area with scroll
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignCenter)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        self.image_label.mousePressEvent = self.image_click_event
        
        scroll_area.setWidget(self.image_label)
        image_layout.addWidget(scroll_area)
        
        # Navigation controls
        nav_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("← Previous")
        self.prev_button.clicked.connect(self.previous_image)
        self.prev_button.setEnabled(False)
        nav_layout.addWidget(self.prev_button)
        
        nav_layout.addStretch()
        
        self.next_button = QPushButton("Next →")
        self.next_button.clicked.connect(self.next_image)
        self.next_button.setEnabled(False)
        nav_layout.addWidget(self.next_button)
        
        image_layout.addLayout(nav_layout)
        
        parent.addWidget(image_widget)
    
    def create_control_panel(self, parent):
        """Create control panel with detection list and SAM controls."""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # Detection list group
        detection_group = QGroupBox("Detections")
        detection_layout = QVBoxLayout(detection_group)
        
        self.detection_list = QListWidget()
        self.detection_list.itemClicked.connect(self.detection_selected)
        detection_layout.addWidget(self.detection_list)
        
        control_layout.addWidget(detection_group)
        
        # SAM controls group
        sam_group = QGroupBox("SAM Controls")
        sam_layout = QVBoxLayout(sam_group)
        
        self.find_similar_button = QPushButton("Find Similar Objects")
        self.find_similar_button.clicked.connect(self.find_similar_objects)
        self.find_similar_button.setEnabled(False)
        sam_layout.addWidget(self.find_similar_button)
        
        # Threshold slider
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Similarity Threshold:"))
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(70)
        self.threshold_slider.valueChanged.connect(self.threshold_changed)
        threshold_layout.addWidget(self.threshold_slider)
        
        self.threshold_label = QLabel("0.70")
        threshold_layout.addWidget(self.threshold_label)
        
        sam_layout.addLayout(threshold_layout)
        
        self.apply_threshold_button = QPushButton("Apply Threshold Filter")
        self.apply_threshold_button.clicked.connect(self.apply_threshold_filter)
        self.apply_threshold_button.setEnabled(False)
        sam_layout.addWidget(self.apply_threshold_button)
        
        self.name_objects_button = QPushButton("Name Object Group")
        self.name_objects_button.clicked.connect(self.name_objects)
        self.name_objects_button.setEnabled(False)
        sam_layout.addWidget(self.name_objects_button)
        
        control_layout.addWidget(sam_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)
        
        # Status info
        self.status_info = QLabel("Select a detection to start")
        self.status_info.setWordWrap(True)
        self.status_info.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        control_layout.addWidget(self.status_info)
        
        control_layout.addStretch()
        
        parent.addWidget(control_widget)
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Navigation shortcuts will be handled by button clicks for now
        pass
    
    def load_dataset(self):
        """Load dataset and initialize parsers."""
        try:
            # Initialize YOLO parser
            classes_file = self.dataset_path / "classes.txt"
            self.yolo_parser = YOLOParser(classes_file)
            
            # Load annotations
            labels_dir = self.dataset_path / "labels"
            images_dir = self.dataset_path / "images"
            
            annotations = self.yolo_parser.load_dataset_annotations(labels_dir, images_dir)
            
            # Initialize image navigator
            self.image_navigator = ImageNavigator(images_dir, annotations)
            
            # Update UI
            self.update_image_display()
            self.update_detection_list()
            self.update_navigation_buttons()
            
            # Update status
            total_images = self.image_navigator.total_images
            total_detections = sum(len(dets) for dets in annotations.values())
            self.status_bar.showMessage(f"Loaded {total_images} images with {total_detections} detections")
            
            logger.info(f"Dataset loaded successfully: {total_images} images, {total_detections} detections")
            
        except Exception as e:
            error_msg = f"Failed to load dataset: {e}"
            logger.error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            self.status_bar.showMessage("Failed to load dataset")
    
    def update_image_display(self):
        """Update image display with current image and detections."""
        if not self.image_navigator:
            return
        
        # Update image info
        current_num, total, image_name = self.image_navigator.get_image_info()
        self.image_info_label.setText(f"Image {current_num} of {total}: {image_name}")
        
        # Load and display image
        image = self.image_navigator.load_current_image()
        if image is not None:
            # Draw detections on image
            image_with_detections = self.image_navigator.draw_detections(
                image, self.yolo_parser.classes, self.selected_detection
            )
            
            # Convert to Qt pixmap
            pixmap = self.cv_image_to_pixmap(image_with_detections)
            self.image_label.setPixmap(pixmap)
            self.image_label.resize(pixmap.size())
        else:
            self.image_label.setText("Failed to load image")
    
    def cv_image_to_pixmap(self, cv_image: np.ndarray) -> QPixmap:
        """
        Convert OpenCV image to Qt pixmap.
        
        Args:
            cv_image: OpenCV image (BGR format)
            
        Returns:
            Qt pixmap
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Create Qt image
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to pixmap
        return QPixmap.fromImage(qt_image)
    
    def update_detection_list(self):
        """Update detection list widget."""
        self.detection_list.clear()
        
        if not self.image_navigator:
            return
        
        detections = self.image_navigator.current_detections
        
        for i, detection in enumerate(detections):
            class_name = self.yolo_parser.get_class_name(detection.class_id)
            
            # Format detection info
            item_text = (
                f"Detection {i+1}: {class_name} (ID: {detection.class_id})\n"
                f"Center: ({detection.x_center:.3f}, {detection.y_center:.3f})\n"
                f"Size: {detection.width:.3f} × {detection.height:.3f}"
            )
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, i)  # Store detection index
            self.detection_list.addItem(item)
    
    def update_navigation_buttons(self):
        """Update navigation button states."""
        if not self.image_navigator:
            return
        
        total_images = self.image_navigator.total_images
        current_index = self.image_navigator.current_index
        
        self.prev_button.setEnabled(current_index > 0)
        self.next_button.setEnabled(current_index < total_images - 1)
    
    def previous_image(self):
        """Navigate to previous image."""
        if self.image_navigator and self.image_navigator.previous_image():
            self.selected_detection = None
            self.update_image_display()
            self.update_detection_list()
            self.update_navigation_buttons()
            self.update_sam_controls()
    
    def next_image(self):
        """Navigate to next image."""
        if self.image_navigator and self.image_navigator.next_image():
            self.selected_detection = None
            self.update_image_display()
            self.update_detection_list()
            self.update_navigation_buttons()
            self.update_sam_controls()
    
    def detection_selected(self, item: QListWidgetItem):
        """Handle detection selection from list."""
        detection_index = item.data(Qt.UserRole)
        self.selected_detection = detection_index
        
        # Update image display to highlight selected detection
        self.update_image_display()
        
        # Update SAM controls
        self.update_sam_controls()
        
        # Update status
        detection = self.image_navigator.current_detections[detection_index]
        class_name = self.yolo_parser.get_class_name(detection.class_id)
        self.status_info.setText(f"Selected: {class_name} (Detection {detection_index + 1})")
    
    def image_click_event(self, event):
        """Handle mouse click on image to select detection."""
        if not self.image_navigator or not self.image_navigator.current_detections:
            return
        
        # Get click coordinates relative to image
        x = event.pos().x()
        y = event.pos().y()
        
        # Get image dimensions
        pixmap = self.image_label.pixmap()
        if not pixmap:
            return
        
        img_width, img_height = self.image_navigator.get_image_dimensions()
        if img_width == 0 or img_height == 0:
            return
        
        # Convert click to image coordinates
        click_x = (x / pixmap.width()) * img_width
        click_y = (y / pixmap.height()) * img_height
        
        # Find clicked detection
        detections = self.image_navigator.current_detections
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.to_bbox(img_width, img_height)
            
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                self.selected_detection = i
                
                # Select in list
                self.detection_list.setCurrentRow(i)
                
                # Update display
                self.update_image_display()
                self.update_sam_controls()
                
                # Update status
                class_name = self.yolo_parser.get_class_name(detection.class_id)
                self.status_info.setText(f"Selected: {class_name} (Detection {i + 1})")
                break
    
    def update_sam_controls(self):
        """Update SAM control button states."""
        has_selection = self.selected_detection is not None
        
        self.find_similar_button.setEnabled(has_selection)
        self.name_objects_button.setEnabled(has_selection)
    
    def threshold_changed(self, value):
        """Handle threshold slider change."""
        threshold = value / 100.0
        self.threshold_label.setText(f"{threshold:.2f}")
    
    def find_similar_objects(self):
        """Find similar objects using SAM2."""
        if self.selected_detection is None:
            return
        
        # TODO: Implement SAM2 integration
        self.status_info.setText("SAM2 integration coming soon...")
        QMessageBox.information(self, "Info", "SAM2 integration will be implemented in the next phase.")
    
    def apply_threshold_filter(self):
        """Apply similarity threshold filter."""
        # TODO: Implement threshold filtering
        self.status_info.setText("Threshold filtering coming soon...")
        QMessageBox.information(self, "Info", "Threshold filtering will be implemented after SAM2 integration.")
    
    def name_objects(self):
        """Open dialog to name object group."""
        # TODO: Implement object naming
        self.status_info.setText("Object naming coming soon...")
        QMessageBox.information(self, "Info", "Object naming functionality will be implemented in the final phase.")