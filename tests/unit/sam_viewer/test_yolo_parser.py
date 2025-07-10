"""Tests for YOLO parser module."""

import pytest
import tempfile
from pathlib import Path

from ssya.sam_viewer.modules.yolo_parser import YOLOParser, YOLODetection


class TestYOLODetection:
    """Test YOLODetection class."""
    
    def test_detection_creation(self):
        """Test creating a YOLO detection."""
        detection = YOLODetection(1, 0.5, 0.6, 0.2, 0.3)
        
        assert detection.class_id == 1
        assert detection.x_center == 0.5
        assert detection.y_center == 0.6
        assert detection.width == 0.2
        assert detection.height == 0.3
    
    def test_to_bbox_conversion(self):
        """Test conversion from YOLO format to bounding box."""
        detection = YOLODetection(0, 0.5, 0.5, 0.2, 0.3)
        
        # For 640x480 image
        bbox = detection.to_bbox(640, 480)
        
        # Expected: center at (320, 240), size (128, 144)
        # So bbox should be (256, 168, 384, 312)
        assert bbox == (256, 168, 384, 312)
    
    def test_to_bbox_edge_cases(self):
        """Test edge cases for bbox conversion."""
        # Top-left corner
        detection = YOLODetection(0, 0.1, 0.1, 0.2, 0.2)
        bbox = detection.to_bbox(100, 100)
        assert bbox == (0, 0, 20, 20)
        
        # Bottom-right corner
        detection = YOLODetection(0, 0.9, 0.9, 0.2, 0.2)
        bbox = detection.to_bbox(100, 100)
        assert bbox == (80, 80, 100, 100)


class TestYOLOParser:
    """Test YOLOParser class."""
    
    @pytest.fixture
    def temp_classes_file(self):
        """Create a temporary classes file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("person\ncar\nbicycle\ndog\n")
            f.flush()
            yield Path(f.name)
        Path(f.name).unlink()
    
    @pytest.fixture
    def temp_annotation_file(self):
        """Create a temporary annotation file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("0 0.5 0.5 0.2 0.3\n")
            f.write("1 0.3 0.4 0.1 0.2\n")
            f.write("2 0.7 0.8 0.15 0.25\n")
            f.flush()
            yield Path(f.name)
        Path(f.name).unlink()
    
    def test_load_classes(self, temp_classes_file):
        """Test loading classes from file."""
        parser = YOLOParser(temp_classes_file)
        
        assert len(parser.classes) == 4
        assert parser.classes == ["person", "car", "bicycle", "dog"]
    
    def test_get_class_name(self, temp_classes_file):
        """Test getting class name by ID."""
        parser = YOLOParser(temp_classes_file)
        
        assert parser.get_class_name(0) == "person"
        assert parser.get_class_name(1) == "car"
        assert parser.get_class_name(3) == "dog"
        assert parser.get_class_name(999) == "Unknown(999)"
        assert parser.get_class_name(-1) == "Unknown(-1)"
    
    def test_parse_annotation_file(self, temp_classes_file, temp_annotation_file):
        """Test parsing annotation file."""
        parser = YOLOParser(temp_classes_file)
        detections = parser.parse_annotation_file(temp_annotation_file)
        
        assert len(detections) == 3
        
        # Check first detection
        det1 = detections[0]
        assert det1.class_id == 0
        assert det1.x_center == 0.5
        assert det1.y_center == 0.5
        assert det1.width == 0.2
        assert det1.height == 0.3
        
        # Check second detection
        det2 = detections[1]
        assert det2.class_id == 1
        assert det2.x_center == 0.3
        assert det2.y_center == 0.4
    
    def test_parse_invalid_annotation_lines(self, temp_classes_file):
        """Test parsing annotation file with invalid lines."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("0 0.5 0.5 0.2 0.3\n")  # Valid line
            f.write("invalid line\n")         # Invalid: not enough values
            f.write("0 1.5 0.5 0.2 0.3\n")   # Invalid: x_center > 1.0
            f.write("0 0.5 0.5 0.2\n")       # Invalid: missing height
            f.write("1 0.2 0.3 0.1 0.15\n")  # Valid line
            f.flush()
            
            parser = YOLOParser(temp_classes_file)
            detections = parser.parse_annotation_file(Path(f.name))
            
            # Should only parse the 2 valid lines
            assert len(detections) == 2
            assert detections[0].class_id == 0
            assert detections[1].class_id == 1
        
        Path(f.name).unlink()
    
    def test_load_dataset_annotations(self, temp_classes_file):
        """Test loading dataset annotations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            labels_dir = temp_path / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()
            
            # Create test image files
            (images_dir / "image1.jpg").touch()
            (images_dir / "image2.png").touch()
            (images_dir / "image3.jpg").touch()
            
            # Create annotation files
            with open(labels_dir / "image1.txt", 'w') as f:
                f.write("0 0.5 0.5 0.2 0.3\n")
            
            with open(labels_dir / "image2.txt", 'w') as f:
                f.write("1 0.3 0.4 0.1 0.2\n")
                f.write("2 0.7 0.8 0.15 0.25\n")
            
            # image3.jpg has no annotation file
            
            parser = YOLOParser(temp_classes_file)
            annotations = parser.load_dataset_annotations(labels_dir, images_dir)
            
            assert len(annotations) == 3
            assert len(annotations["image1.jpg"]) == 1
            assert len(annotations["image2.png"]) == 2
            assert len(annotations["image3.jpg"]) == 0  # No annotations