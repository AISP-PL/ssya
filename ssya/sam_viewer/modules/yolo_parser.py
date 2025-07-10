"""YOLO annotation parser module."""

import logging
from pathlib import Path
from typing import List, NamedTuple, Dict, Optional

logger = logging.getLogger(__name__)


class YOLODetection(NamedTuple):
    """Represents a single YOLO detection."""
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    
    def to_bbox(self, img_width: int, img_height: int) -> tuple[int, int, int, int]:
        """
        Convert YOLO format to bounding box coordinates.
        
        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            Tuple of (x1, y1, x2, y2) in pixel coordinates
        """
        x_center_px = self.x_center * img_width
        y_center_px = self.y_center * img_height
        width_px = self.width * img_width
        height_px = self.height * img_height
        
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)
        
        return x1, y1, x2, y2


class YOLOParser:
    """Parser for YOLO format annotations."""
    
    def __init__(self, classes_file: Path):
        """
        Initialize YOLO parser.
        
        Args:
            classes_file: Path to classes.txt file
        """
        self.classes_file = classes_file
        self.classes = self._load_classes()
        
    def _load_classes(self) -> List[str]:
        """Load class names from classes.txt file."""
        try:
            with open(self.classes_file, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Loaded {len(classes)} classes from {self.classes_file}")
            return classes
        except Exception as e:
            logger.error(f"Failed to load classes from {self.classes_file}: {e}")
            raise
    
    def get_class_name(self, class_id: int) -> str:
        """
        Get class name by ID.
        
        Args:
            class_id: Class ID
            
        Returns:
            Class name or "Unknown" if ID is invalid
        """
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        return f"Unknown({class_id})"
    
    def parse_annotation_file(self, annotation_file: Path) -> List[YOLODetection]:
        """
        Parse a single YOLO annotation file.
        
        Args:
            annotation_file: Path to .txt annotation file
            
        Returns:
            List of YOLODetection objects
        """
        detections = []
        
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f.readlines(), 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) != 5:
                            logger.warning(
                                f"Invalid line format in {annotation_file}:{line_num} - "
                                f"expected 5 values, got {len(parts)}"
                            )
                            continue
                        
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Validate ranges
                        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and
                                0.0 <= width <= 1.0 and 0.0 <= height <= 1.0):
                            logger.warning(
                                f"Invalid coordinates in {annotation_file}:{line_num} - "
                                f"values should be between 0.0 and 1.0"
                            )
                            continue
                        
                        detection = YOLODetection(class_id, x_center, y_center, width, height)
                        detections.append(detection)
                        
                    except ValueError as e:
                        logger.warning(
                            f"Failed to parse line in {annotation_file}:{line_num} - {e}"
                        )
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to read annotation file {annotation_file}: {e}")
            raise
        
        logger.debug(f"Parsed {len(detections)} detections from {annotation_file}")
        return detections
    
    def load_dataset_annotations(self, labels_dir: Path, images_dir: Path) -> Dict[str, List[YOLODetection]]:
        """
        Load all annotations from a dataset.
        
        Args:
            labels_dir: Directory containing .txt annotation files
            images_dir: Directory containing image files
            
        Returns:
            Dictionary mapping image filenames to list of detections
        """
        annotations = {}
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} image files in {images_dir}")
        
        # Load annotations for each image
        for image_file in image_files:
            annotation_file = labels_dir / f"{image_file.stem}.txt"
            
            if annotation_file.exists():
                try:
                    detections = self.parse_annotation_file(annotation_file)
                    annotations[image_file.name] = detections
                except Exception as e:
                    logger.error(f"Failed to load annotations for {image_file.name}: {e}")
                    annotations[image_file.name] = []
            else:
                logger.debug(f"No annotation file found for {image_file.name}")
                annotations[image_file.name] = []
        
        total_detections = sum(len(detections) for detections in annotations.values())
        logger.info(f"Loaded annotations for {len(annotations)} images with {total_detections} total detections")
        
        return annotations