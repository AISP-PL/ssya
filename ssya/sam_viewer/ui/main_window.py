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
    QMessageBox, QProgressBar, QStatusBar, QFrame,
    QProgressDialog, QInputDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen

from ..modules.yolo_parser import YOLOParser, YOLODetection
from ..modules.image_navigator import ImageNavigator
from ..modules.sam_interface import SAMInterface
from ..modules.feature_matcher import FeatureMatcher, SimilarityResult

logger = logging.getLogger(__name__)


class FeatureExtractionWorker(QThread):
    """Worker thread for feature extraction."""
    
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(bool)  # success
    
    def __init__(self, feature_matcher: FeatureMatcher, image_navigator: ImageNavigator):
        super().__init__()
        self.feature_matcher = feature_matcher
        self.image_navigator = image_navigator
    
    def run(self):
        """Run feature extraction in background."""
        try:
            def progress_callback(current, total):
                self.progress.emit(current, total)
            
            success = self.feature_matcher.process_dataset(
                self.image_navigator, progress_callback
            )
            self.finished.emit(success)
            
        except Exception as e:
            logger.error(f"Feature extraction worker failed: {e}")
            self.finished.emit(False)


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
        self.sam_interface: Optional[SAMInterface] = None
        self.feature_matcher: Optional[FeatureMatcher] = None
        self.selected_detection: Optional[int] = None
        self.similarity_results: List[SimilarityResult] = []
        self.filtered_images: Optional[List[str]] = None
        self.current_threshold: float = 0.7
        
        # Initialize UI
        self.init_ui()
        
        # Load dataset
        self.load_dataset()
        
        # Initialize SAM
        self.init_sam()
        
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
        
        # Similarity results group
        results_group = QGroupBox("Similar Objects")
        results_layout = QVBoxLayout(results_group)
        
        self.results_list = QListWidget()
        self.results_list.itemClicked.connect(self.similarity_result_selected)
        results_layout.addWidget(self.results_list)
        
        control_layout.addWidget(results_group)
        
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
    
    def init_sam(self):
        """Initialize SAM interface and feature matcher."""
        try:
            # Initialize SAM interface
            self.sam_interface = SAMInterface()
            
            # Initialize feature matcher with cache
            cache_dir = self.dataset_path / "cache" / "features"
            self.feature_matcher = FeatureMatcher(self.sam_interface, cache_dir)
            
            if self.sam_interface.is_loaded:
                self.status_info.setText("SAM interface ready. Select a detection to begin.")
                logger.info("SAM interface initialized successfully")
            else:
                self.status_info.setText("SAM interface failed to load. Feature extraction disabled.")
                logger.warning("SAM interface failed to load")
                
        except Exception as e:
            error_msg = f"Failed to initialize SAM: {e}"
            logger.error(error_msg)
            self.status_info.setText("SAM initialization failed.")
    
    def update_image_display(self):
        """Update image display with current image and detections."""
        if not self.image_navigator:
            return
        
        # Update image info
        current_num, total, image_name = self.image_navigator.get_image_info()
        filter_info = ""
        if self.filtered_images is not None:
            filter_info = f" (Filtered: {len(self.filtered_images)} images)"
        self.image_info_label.setText(f"Image {current_num} of {total}: {image_name}{filter_info}")
        
        # Load and display image
        image = self.image_navigator.load_current_image()
        if image is not None:
            # Draw detections on image
            image_with_detections = self.image_navigator.draw_detections(
                image, self.yolo_parser.classes, self.selected_detection
            )
            
            # If we have a selected detection and SAM mask, overlay it
            if (self.selected_detection is not None and 
                self.feature_matcher and self.sam_interface):
                
                feature = self.feature_matcher.get_detection_feature(
                    self.image_navigator.current_image_name, self.selected_detection
                )
                if feature and feature.mask is not None:
                    image_with_detections = self.overlay_sam_mask(image_with_detections, feature.mask)
            
            # Convert to Qt pixmap
            pixmap = self.cv_image_to_pixmap(image_with_detections)
            self.image_label.setPixmap(pixmap)
            self.image_label.resize(pixmap.size())
        else:
            self.image_label.setText("Failed to load image")
    
    def overlay_sam_mask(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Overlay SAM mask on image.
        
        Args:
            image: Input image (BGR format)
            mask: Binary mask
            alpha: Transparency factor
            
        Returns:
            Image with overlaid mask
        """
        # Create colored mask (blue overlay)
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = [255, 100, 0]  # Orange color for mask
        
        # Blend with original image
        result = cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)
        
        # Draw mask contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 255), 2)  # Yellow contours
        
        return result
    
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
        
        if self.filtered_images is not None:
            # Navigation within filtered results
            current_image = self.image_navigator.current_image_name
            if current_image in self.filtered_images:
                current_index = self.filtered_images.index(current_image)
                self.prev_button.setEnabled(current_index > 0)
                self.next_button.setEnabled(current_index < len(self.filtered_images) - 1)
            else:
                self.prev_button.setEnabled(False)
                self.next_button.setEnabled(False)
        else:
            # Normal navigation
            total_images = self.image_navigator.total_images
            current_index = self.image_navigator.current_index
            
            self.prev_button.setEnabled(current_index > 0)
            self.next_button.setEnabled(current_index < total_images - 1)
    
    def previous_image(self):
        """Navigate to previous image."""
        if not self.image_navigator:
            return
            
        if self.filtered_images is not None:
            # Navigate within filtered results
            current_image = self.image_navigator.current_image_name
            if current_image in self.filtered_images:
                current_index = self.filtered_images.index(current_image)
                if current_index > 0:
                    prev_image = self.filtered_images[current_index - 1]
                    self.image_navigator.find_image_by_name(prev_image)
        else:
            # Normal navigation
            self.image_navigator.previous_image()
        
        self.selected_detection = None
        self.update_image_display()
        self.update_detection_list()
        self.update_navigation_buttons()
        self.update_sam_controls()
    
    def next_image(self):
        """Navigate to next image."""
        if not self.image_navigator:
            return
            
        if self.filtered_images is not None:
            # Navigate within filtered results
            current_image = self.image_navigator.current_image_name
            if current_image in self.filtered_images:
                current_index = self.filtered_images.index(current_image)
                if current_index < len(self.filtered_images) - 1:
                    next_image = self.filtered_images[current_index + 1]
                    self.image_navigator.find_image_by_name(next_image)
        else:
            # Normal navigation
            self.image_navigator.next_image()
        
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
        has_sam = self.sam_interface and self.sam_interface.is_loaded
        has_results = len(self.similarity_results) > 0
        
        self.find_similar_button.setEnabled(has_selection and has_sam)
        self.apply_threshold_button.setEnabled(has_results)
        self.name_objects_button.setEnabled(has_results)
    
    def threshold_changed(self, value):
        """Handle threshold slider change."""
        threshold = value / 100.0
        self.threshold_label.setText(f"{threshold:.2f}")
        self.current_threshold = threshold
    
    def find_similar_objects(self):
        """Find similar objects using SAM2."""
        if self.selected_detection is None or not self.feature_matcher:
            return
        
        # Check if features have been extracted
        if not self.feature_matcher.detection_features:
            # Need to extract features first
            reply = QMessageBox.question(
                self, "Feature Extraction", 
                "Features need to be extracted from all images first. This may take a few minutes. Continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
            
            # Start feature extraction
            self.extract_features()
            return
        
        # Get reference embedding
        current_image = self.image_navigator.current_image_name
        reference_feature = self.feature_matcher.get_detection_feature(current_image, self.selected_detection)
        
        if not reference_feature:
            # Extract feature for current detection
            image = self.image_navigator.load_current_image()
            detection = self.image_navigator.current_detections[self.selected_detection]
            
            reference_feature = self.feature_matcher.extract_detection_features(
                image, current_image, detection, self.selected_detection
            )
            
            if not reference_feature:
                QMessageBox.warning(self, "Error", "Failed to extract features for selected detection.")
                return
        
        # Find similar objects
        self.similarity_results = self.feature_matcher.find_similar_objects(
            reference_feature.embedding,
            similarity_threshold=0.0,  # Show all results, filter with slider
            max_results=100
        )
        
        # Update results display
        self.update_similarity_results()
        
        # Update controls
        self.update_sam_controls()
        
        # Update status
        self.status_info.setText(f"Found {len(self.similarity_results)} similar objects")
    
    def extract_features(self):
        """Extract features from all detections."""
        if not self.feature_matcher or not self.image_navigator:
            return
        
        # Create progress dialog
        progress_dialog = QProgressDialog("Extracting features...", "Cancel", 0, 100, self)
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.show()
        
        # Create worker thread
        self.extraction_worker = FeatureExtractionWorker(self.feature_matcher, self.image_navigator)
        self.extraction_worker.progress.connect(lambda current, total: 
            progress_dialog.setValue(int(100 * current / total)))
        self.extraction_worker.finished.connect(lambda success: self.feature_extraction_finished(success, progress_dialog))
        
        # Start extraction
        self.extraction_worker.start()
    
    def feature_extraction_finished(self, success: bool, progress_dialog: QProgressDialog):
        """Handle feature extraction completion."""
        progress_dialog.close()
        
        if success:
            stats = self.feature_matcher.get_statistics()
            QMessageBox.information(
                self, "Feature Extraction Complete",
                f"Successfully extracted features from {stats['total_features']} detections "
                f"across {stats['total_images']} images."
            )
            
            # Now try finding similar objects again
            self.find_similar_objects()
        else:
            QMessageBox.critical(self, "Error", "Feature extraction failed.")
    
    def update_similarity_results(self):
        """Update similarity results list."""
        self.results_list.clear()
        
        if not self.similarity_results:
            return
        
        # Filter by threshold
        filtered_results = [
            result for result in self.similarity_results 
            if result.similarity_score >= self.current_threshold
        ]
        
        for result in filtered_results:
            class_name = self.yolo_parser.get_class_name(result.detection.class_id)
            
            item_text = (
                f"{result.image_name}\n"
                f"{class_name} (Det {result.detection_index + 1})\n"
                f"Similarity: {result.similarity_score:.3f}"
            )
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, result)
            self.results_list.addItem(item)
    
    def similarity_result_selected(self, item: QListWidgetItem):
        """Handle selection of similarity result."""
        result = item.data(Qt.UserRole)
        
        # Navigate to the image
        if self.image_navigator.find_image_by_name(result.image_name):
            self.selected_detection = result.detection_index
            
            # Update displays
            self.update_image_display()
            self.update_detection_list()
            self.update_navigation_buttons()
            
            # Select detection in list
            self.detection_list.setCurrentRow(result.detection_index)
            
            # Update status
            class_name = self.yolo_parser.get_class_name(result.detection.class_id)
            self.status_info.setText(
                f"Viewing similar object: {class_name} "
                f"(Similarity: {result.similarity_score:.3f})"
            )
    
    def apply_threshold_filter(self):
        """Apply similarity threshold filter."""
        if not self.similarity_results:
            return
        
        # Get filtered results
        filtered_results = [
            result for result in self.similarity_results 
            if result.similarity_score >= self.current_threshold
        ]
        
        if not filtered_results:
            QMessageBox.information(self, "No Results", "No objects match the current threshold.")
            return
        
        # Extract unique image names
        self.filtered_images = list(set(result.image_name for result in filtered_results))
        self.filtered_images.sort()
        
        # Navigate to first filtered image
        if self.filtered_images:
            self.image_navigator.find_image_by_name(self.filtered_images[0])
            self.selected_detection = None
            
            # Update displays
            self.update_image_display()
            self.update_detection_list()
            self.update_navigation_buttons()
            
            # Update status
            self.status_info.setText(
                f"Filtered to {len(self.filtered_images)} images "
                f"with {len(filtered_results)} similar objects"
            )
        
        # Update similarity results display
        self.update_similarity_results()
    
    def name_objects(self):
        """Open dialog to name object group."""
        if not self.similarity_results:
            return
        
        # Get current threshold filtered results
        filtered_results = [
            result for result in self.similarity_results 
            if result.similarity_score >= self.current_threshold
        ]
        
        if not filtered_results:
            QMessageBox.information(self, "No Objects", "No objects match the current threshold.")
            return
        
        # Ask for group name
        name, ok = QInputDialog.getText(
            self, "Name Object Group",
            f"Enter name for group of {len(filtered_results)} similar objects:"
        )
        
        if ok and name.strip():
            # Save metadata
            self.save_object_group_metadata(name.strip(), filtered_results)
    
    def save_object_group_metadata(self, group_name: str, results: List[SimilarityResult]):
        """Save object group metadata to JSON file."""
        try:
            # Create metadata
            metadata = {
                "group_name": group_name,
                "created_at": str(Path(__file__).stat().st_ctime),
                "threshold": self.current_threshold,
                "total_objects": len(results),
                "objects": []
            }
            
            for result in results:
                obj_data = {
                    "image_name": result.image_name,
                    "detection_index": result.detection_index,
                    "class_id": result.detection.class_id,
                    "class_name": self.yolo_parser.get_class_name(result.detection.class_id),
                    "bbox": {
                        "x_center": result.detection.x_center,
                        "y_center": result.detection.y_center,
                        "width": result.detection.width,
                        "height": result.detection.height
                    },
                    "similarity_score": result.similarity_score
                }
                metadata["objects"].append(obj_data)
            
            # Save to file
            output_dir = self.dataset_path / "output"
            output_dir.mkdir(exist_ok=True)
            
            output_file = output_dir / f"{group_name.replace(' ', '_').lower()}_group.json"
            
            with open(output_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            QMessageBox.information(
                self, "Group Saved",
                f"Object group '{group_name}' saved to:\n{output_file}"
            )
            
            logger.info(f"Saved object group '{group_name}' with {len(results)} objects to {output_file}")
            
        except Exception as e:
            error_msg = f"Failed to save object group: {e}"
            logger.error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)