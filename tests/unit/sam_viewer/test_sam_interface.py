"""Tests for SAM interface module."""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from ssya.sam_viewer.modules.sam_interface import SAMInterface


class TestSAMInterface:
    """Test SAMInterface class."""
    
    @pytest.fixture
    def sam_interface(self):
        """Create SAM interface instance."""
        return SAMInterface()
    
    @pytest.fixture
    def test_image(self):
        """Create test image."""
        return np.zeros((100, 150, 3), dtype=np.uint8)
    
    def test_initialization(self, sam_interface):
        """Test SAM interface initialization."""
        assert sam_interface.model_path is None
        assert sam_interface.is_loaded is True  # Mock implementation always loads
    
    def test_set_image(self, sam_interface, test_image):
        """Test setting image for processing."""
        success = sam_interface.set_image(test_image)
        assert success is True
        assert hasattr(sam_interface, 'current_image')
        assert np.array_equal(sam_interface.current_image, test_image)
    
    def test_predict_mask_from_bbox(self, sam_interface, test_image):
        """Test mask prediction from bounding box."""
        sam_interface.set_image(test_image)
        
        bbox = (20, 30, 80, 70)  # x1, y1, x2, y2
        mask, embedding, confidence = sam_interface.predict_mask(bbox)
        
        # Check outputs
        assert mask is not None
        assert mask.shape == (100, 150)  # Same as image height, width
        assert mask.dtype == np.uint8
        
        assert embedding is not None
        assert embedding.shape == (256,)
        assert embedding.dtype == np.float32
        
        assert 0.0 <= confidence <= 1.0
    
    def test_predict_mask_from_points(self, sam_interface, test_image):
        """Test mask prediction from points."""
        sam_interface.set_image(test_image)
        
        points = np.array([[50, 60], [70, 80]])
        labels = np.array([1, 1])  # Both positive points
        
        mask, embedding, confidence = sam_interface.predict_mask_from_points(points, labels)
        
        # Check outputs
        assert mask is not None
        assert mask.shape == (100, 150)
        assert embedding is not None
        assert embedding.shape == (256,)
        assert 0.0 <= confidence <= 1.0
    
    def test_extract_features_from_mask(self, sam_interface, test_image):
        """Test feature extraction from masked region."""
        # Create a simple mask
        mask = np.zeros((100, 150), dtype=np.uint8)
        cv2.rectangle(mask, (20, 30), (80, 70), 255, -1)
        
        features = sam_interface.extract_features_from_mask(test_image, mask)
        
        assert features is not None
        assert features.shape == (256,)
        assert features.dtype == np.float32
        
        # Features should be normalized
        norm = np.linalg.norm(features)
        assert abs(norm - 1.0) < 1e-6 or norm == 0.0
    
    def test_compute_similarity_cosine(self, sam_interface):
        """Test cosine similarity computation."""
        # Create test embeddings
        embedding1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embedding2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        embedding3 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        # Test orthogonal vectors (should be 0)
        similarity = sam_interface.compute_similarity(embedding1, embedding2, "cosine")
        assert abs(similarity - 0.0) < 1e-6
        
        # Test identical vectors (should be 1)
        similarity = sam_interface.compute_similarity(embedding1, embedding3, "cosine")
        assert abs(similarity - 1.0) < 1e-6
    
    def test_compute_similarity_euclidean(self, sam_interface):
        """Test Euclidean similarity computation."""
        # Create test embeddings
        embedding1 = np.array([0.0, 0.0], dtype=np.float32)
        embedding2 = np.array([1.0, 1.0], dtype=np.float32)
        embedding3 = np.array([0.0, 0.0], dtype=np.float32)
        
        # Test different vectors
        similarity = sam_interface.compute_similarity(embedding1, embedding2, "euclidean")
        assert 0.0 <= similarity <= 1.0
        
        # Test identical vectors (should be 1)
        similarity = sam_interface.compute_similarity(embedding1, embedding3, "euclidean")
        assert abs(similarity - 1.0) < 1e-3
    
    def test_predict_mask_without_image(self, sam_interface):
        """Test mask prediction without setting image first."""
        bbox = (20, 30, 80, 70)
        mask, embedding, confidence = sam_interface.predict_mask(bbox)
        
        assert mask is None
        assert embedding is None
        assert confidence == 0.0
    
    def test_predict_mask_invalid_bbox(self, sam_interface, test_image):
        """Test mask prediction with invalid bounding box."""
        sam_interface.set_image(test_image)
        
        # Bbox outside image bounds
        bbox = (200, 200, 300, 300)
        mask, embedding, confidence = sam_interface.predict_mask(bbox)
        
        # Should still work but adjust bbox to image bounds
        assert mask is not None
        assert mask.shape == (100, 150)
    
    def test_extract_features_empty_mask(self, sam_interface, test_image):
        """Test feature extraction with empty mask."""
        # Create empty mask
        mask = np.zeros((100, 150), dtype=np.uint8)
        
        features = sam_interface.extract_features_from_mask(test_image, mask)
        
        # Should still return features (padding/default values)
        assert features is not None
        assert features.shape == (256,)