"""Image navigation and loading module."""

import logging
from pathlib import Path
from typing import List, Optional, Dict
from PIL import Image
import cv2
import numpy as np

from .yolo_parser import YOLODetection

logger = logging.getLogger(__name__)


class ImageNavigator:
    """Handles image loading and navigation through dataset."""
    
    def __init__(self, images_dir: Path, annotations: Dict[str, List[YOLODetection]]):
        """
        Initialize image navigator.
        
        Args:
            images_dir: Directory containing images
            annotations: Dictionary mapping image names to detections
        """
        self.images_dir = images_dir
        self.annotations = annotations
        self.image_files = self._load_image_list()
        self.current_index = 0
        
    def _load_image_list(self) -> List[str]:
        """Load and sort list of image files."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend([f.name for f in self.images_dir.glob(f"*{ext}")])
            image_files.extend([f.name for f in self.images_dir.glob(f"*{ext.upper()}")])
        
        # Sort files naturally
        image_files.sort()
        
        # Filter to only include images that have annotations loaded
        image_files = [img for img in image_files if img in self.annotations]
        
        logger.info(f"Loaded {len(image_files)} image files for navigation")
        return image_files
    
    @property
    def total_images(self) -> int:
        """Get total number of images."""
        return len(self.image_files)
    
    @property
    def current_image_name(self) -> Optional[str]:
        """Get current image filename."""
        if 0 <= self.current_index < len(self.image_files):
            return self.image_files[self.current_index]
        return None
    
    @property
    def current_image_path(self) -> Optional[Path]:
        """Get current image full path."""
        if self.current_image_name:
            return self.images_dir / self.current_image_name
        return None
    
    @property
    def current_detections(self) -> List[YOLODetection]:
        """Get detections for current image."""
        if self.current_image_name:
            return self.annotations.get(self.current_image_name, [])
        return []
    
    def get_image_info(self) -> tuple[int, int, str]:
        """
        Get current image information.
        
        Returns:
            Tuple of (current_index + 1, total_images, image_name)
        """
        current_num = self.current_index + 1 if self.image_files else 0
        total = len(self.image_files)
        name = self.current_image_name or "No image"
        return current_num, total, name
    
    def load_current_image(self) -> Optional[np.ndarray]:
        """
        Load current image as numpy array.
        
        Returns:
            Image as BGR numpy array or None if failed
        """
        if not self.current_image_path:
            return None
        
        try:
            # Use OpenCV to load image (BGR format)
            image = cv2.imread(str(self.current_image_path))
            if image is None:
                logger.error(f"Failed to load image: {self.current_image_path}")
                return None
            
            logger.debug(f"Loaded image: {self.current_image_path} - Shape: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {self.current_image_path}: {e}")
            return None
    
    def get_image_dimensions(self) -> tuple[int, int]:
        """
        Get current image dimensions.
        
        Returns:
            Tuple of (width, height) or (0, 0) if no image
        """
        image = self.load_current_image()
        if image is not None:
            height, width = image.shape[:2]
            return width, height
        return 0, 0
    
    def next_image(self) -> bool:
        """
        Navigate to next image.
        
        Returns:
            True if navigation successful, False if at end
        """
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            logger.debug(f"Navigated to next image: {self.current_image_name}")
            return True
        return False
    
    def previous_image(self) -> bool:
        """
        Navigate to previous image.
        
        Returns:
            True if navigation successful, False if at beginning
        """
        if self.current_index > 0:
            self.current_index -= 1
            logger.debug(f"Navigated to previous image: {self.current_image_name}")
            return True
        return False
    
    def go_to_image(self, index: int) -> bool:
        """
        Navigate to specific image by index.
        
        Args:
            index: Image index (0-based)
            
        Returns:
            True if navigation successful, False if invalid index
        """
        if 0 <= index < len(self.image_files):
            self.current_index = index
            logger.debug(f"Navigated to image {index}: {self.current_image_name}")
            return True
        return False
    
    def find_image_by_name(self, image_name: str) -> bool:
        """
        Navigate to image by filename.
        
        Args:
            image_name: Name of the image file
            
        Returns:
            True if image found and navigation successful
        """
        try:
            index = self.image_files.index(image_name)
            return self.go_to_image(index)
        except ValueError:
            logger.warning(f"Image not found: {image_name}")
            return False
    
    def draw_detections(self, image: np.ndarray, class_names: List[str], 
                       selected_detection: Optional[int] = None) -> np.ndarray:
        """
        Draw YOLO detections on image.
        
        Args:
            image: Input image (BGR format)
            class_names: List of class names
            selected_detection: Index of selected detection to highlight
            
        Returns:
            Image with drawn detections
        """
        if image is None:
            return image
        
        result_image = image.copy()
        height, width = image.shape[:2]
        detections = self.current_detections
        
        for i, detection in enumerate(detections):
            # Convert YOLO to bbox coordinates
            x1, y1, x2, y2 = detection.to_bbox(width, height)
            
            # Choose color based on selection
            if i == selected_detection:
                color = (0, 255, 255)  # Yellow for selected
                thickness = 3
            else:
                color = (0, 255, 0)    # Green for normal
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
            
            # Get class name
            class_name = class_names[detection.class_id] if detection.class_id < len(class_names) else f"Class {detection.class_id}"
            
            # Draw label
            label = f"{class_name} ({detection.class_id})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background for label
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return result_image