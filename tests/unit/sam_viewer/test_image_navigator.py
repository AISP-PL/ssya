"""Tests for image navigator module."""

import pytest
import tempfile
import numpy as np
import cv2
from pathlib import Path

from ssya.sam_viewer.modules.image_navigator import ImageNavigator
from ssya.sam_viewer.modules.yolo_parser import YOLODetection


class TestImageNavigator:
    """Test ImageNavigator class."""
    
    @pytest.fixture
    def temp_dataset(self):
        """Create a temporary dataset with images and annotations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            images_dir = temp_path / "images"
            images_dir.mkdir()
            
            # Create test images
            for i in range(3):
                image = np.zeros((100, 150, 3), dtype=np.uint8)
                image[:] = (50 + i * 50, 100, 150)  # Different colors
                cv2.imwrite(str(images_dir / f"image{i+1}.jpg"), image)
            
            # Create annotations
            annotations = {
                "image1.jpg": [
                    YOLODetection(0, 0.5, 0.5, 0.2, 0.3),
                    YOLODetection(1, 0.3, 0.7, 0.1, 0.2)
                ],
                "image2.jpg": [
                    YOLODetection(2, 0.6, 0.4, 0.25, 0.35)
                ],
                "image3.jpg": []  # No detections
            }
            
            yield images_dir, annotations
    
    def test_navigator_initialization(self, temp_dataset):
        """Test navigator initialization."""
        images_dir, annotations = temp_dataset
        navigator = ImageNavigator(images_dir, annotations)
        
        assert navigator.total_images == 3
        assert navigator.current_index == 0
        assert len(navigator.image_files) == 3
        
        # Check image files are sorted
        assert navigator.image_files == sorted(navigator.image_files)
    
    def test_current_image_properties(self, temp_dataset):
        """Test current image properties."""
        images_dir, annotations = temp_dataset
        navigator = ImageNavigator(images_dir, annotations)
        
        # Test initial state
        assert navigator.current_image_name == "image1.jpg"
        assert navigator.current_image_path == images_dir / "image1.jpg"
        assert len(navigator.current_detections) == 2
    
    def test_get_image_info(self, temp_dataset):
        """Test getting image information."""
        images_dir, annotations = temp_dataset
        navigator = ImageNavigator(images_dir, annotations)
        
        current_num, total, name = navigator.get_image_info()
        assert current_num == 1
        assert total == 3
        assert name == "image1.jpg"
    
    def test_load_current_image(self, temp_dataset):
        """Test loading current image."""
        images_dir, annotations = temp_dataset
        navigator = ImageNavigator(images_dir, annotations)
        
        image = navigator.load_current_image()
        assert image is not None
        assert image.shape == (100, 150, 3)
        assert image.dtype == np.uint8
    
    def test_get_image_dimensions(self, temp_dataset):
        """Test getting image dimensions."""
        images_dir, annotations = temp_dataset
        navigator = ImageNavigator(images_dir, annotations)
        
        width, height = navigator.get_image_dimensions()
        assert width == 150
        assert height == 100
    
    def test_navigation(self, temp_dataset):
        """Test image navigation."""
        images_dir, annotations = temp_dataset
        navigator = ImageNavigator(images_dir, annotations)
        
        # Test next navigation
        assert navigator.next_image() is True
        assert navigator.current_image_name == "image2.jpg"
        assert len(navigator.current_detections) == 1
        
        assert navigator.next_image() is True
        assert navigator.current_image_name == "image3.jpg"
        assert len(navigator.current_detections) == 0
        
        # At end, should return False
        assert navigator.next_image() is False
        assert navigator.current_image_name == "image3.jpg"
        
        # Test previous navigation
        assert navigator.previous_image() is True
        assert navigator.current_image_name == "image2.jpg"
        
        assert navigator.previous_image() is True
        assert navigator.current_image_name == "image1.jpg"
        
        # At beginning, should return False
        assert navigator.previous_image() is False
        assert navigator.current_image_name == "image1.jpg"
    
    def test_go_to_image(self, temp_dataset):
        """Test going to specific image by index."""
        images_dir, annotations = temp_dataset
        navigator = ImageNavigator(images_dir, annotations)
        
        # Test valid indices
        assert navigator.go_to_image(2) is True
        assert navigator.current_image_name == "image3.jpg"
        
        assert navigator.go_to_image(0) is True
        assert navigator.current_image_name == "image1.jpg"
        
        # Test invalid indices
        assert navigator.go_to_image(-1) is False
        assert navigator.go_to_image(3) is False
        assert navigator.current_image_name == "image1.jpg"  # Should stay at current
    
    def test_find_image_by_name(self, temp_dataset):
        """Test finding image by filename."""
        images_dir, annotations = temp_dataset
        navigator = ImageNavigator(images_dir, annotations)
        
        # Test existing image
        assert navigator.find_image_by_name("image2.jpg") is True
        assert navigator.current_image_name == "image2.jpg"
        
        # Test non-existing image
        assert navigator.find_image_by_name("nonexistent.jpg") is False
        assert navigator.current_image_name == "image2.jpg"  # Should stay at current
    
    def test_draw_detections(self, temp_dataset):
        """Test drawing detections on image."""
        images_dir, annotations = temp_dataset
        navigator = ImageNavigator(images_dir, annotations)
        
        # Load image and draw detections
        image = navigator.load_current_image()
        class_names = ["person", "car", "bicycle"]
        
        # Draw without selection
        result = navigator.draw_detections(image, class_names)
        assert result.shape == image.shape
        assert not np.array_equal(result, image)  # Should be different (boxes drawn)
        
        # Draw with selection
        result_selected = navigator.draw_detections(image, class_names, selected_detection=0)
        assert result_selected.shape == image.shape
        assert not np.array_equal(result_selected, result)  # Should be different (highlighting)
    
    def test_empty_navigator(self):
        """Test navigator with no images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            images_dir = Path(temp_dir) / "images"
            images_dir.mkdir()
            
            navigator = ImageNavigator(images_dir, {})
            
            assert navigator.total_images == 0
            assert navigator.current_image_name is None
            assert navigator.current_image_path is None
            assert navigator.current_detections == []
            
            # Navigation should fail
            assert navigator.next_image() is False
            assert navigator.previous_image() is False
            
            current_num, total, name = navigator.get_image_info()
            assert current_num == 0
            assert total == 0
            assert name == "No image"