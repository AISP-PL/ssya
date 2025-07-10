"""Tests for feature matcher module."""

import pytest
import tempfile
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import MagicMock, patch

from ssya.sam_viewer.modules.feature_matcher import FeatureMatcher, DetectionFeature, SimilarityResult
from ssya.sam_viewer.modules.sam_interface import SAMInterface
from ssya.sam_viewer.modules.yolo_parser import YOLODetection
from ssya.sam_viewer.modules.image_navigator import ImageNavigator


class TestDetectionFeature:
    """Test DetectionFeature class."""
    
    def test_detection_feature_creation(self):
        """Test creating a detection feature."""
        detection = YOLODetection(1, 0.5, 0.6, 0.2, 0.3)
        embedding = np.random.rand(256).astype(np.float32)
        mask = np.zeros((100, 150), dtype=np.uint8)
        
        feature = DetectionFeature(
            image_name="test.jpg",
            detection_index=0,
            detection=detection,
            embedding=embedding,
            mask=mask,
            confidence=0.85
        )
        
        assert feature.image_name == "test.jpg"
        assert feature.detection_index == 0
        assert feature.detection == detection
        assert np.array_equal(feature.embedding, embedding)
        assert np.array_equal(feature.mask, mask)
        assert feature.confidence == 0.85


class TestSimilarityResult:
    """Test SimilarityResult class."""
    
    def test_similarity_result_creation(self):
        """Test creating a similarity result."""
        detection = YOLODetection(1, 0.5, 0.6, 0.2, 0.3)
        embedding = np.random.rand(256).astype(np.float32)
        
        result = SimilarityResult(
            image_name="test.jpg",
            detection_index=0,
            detection=detection,
            similarity_score=0.75,
            embedding=embedding
        )
        
        assert result.image_name == "test.jpg"
        assert result.detection_index == 0
        assert result.detection == detection
        assert result.similarity_score == 0.75
        assert np.array_equal(result.embedding, embedding)


class TestFeatureMatcher:
    """Test FeatureMatcher class."""
    
    @pytest.fixture
    def sam_interface(self):
        """Create mock SAM interface."""
        sam = SAMInterface()
        return sam
    
    @pytest.fixture
    def feature_matcher(self, sam_interface):
        """Create feature matcher instance."""
        return FeatureMatcher(sam_interface)
    
    @pytest.fixture
    def feature_matcher_with_cache(self, sam_interface):
        """Create feature matcher with cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            yield FeatureMatcher(sam_interface, cache_dir)
    
    @pytest.fixture
    def test_image(self):
        """Create test image."""
        image = np.zeros((100, 150, 3), dtype=np.uint8)
        # Add some pattern for feature extraction
        cv2.rectangle(image, (20, 30), (80, 70), (100, 150, 200), -1)
        return image
    
    @pytest.fixture
    def test_detection(self):
        """Create test YOLO detection."""
        return YOLODetection(1, 0.5, 0.6, 0.2, 0.3)
    
    def test_initialization(self, feature_matcher, sam_interface):
        """Test feature matcher initialization."""
        assert feature_matcher.sam_interface == sam_interface
        assert feature_matcher.cache_dir is None
        assert feature_matcher.detection_features == []
    
    def test_initialization_with_cache(self, feature_matcher_with_cache):
        """Test feature matcher initialization with cache."""
        assert feature_matcher_with_cache.cache_dir is not None
        assert feature_matcher_with_cache.cache_dir.exists()
    
    def test_extract_detection_features(self, feature_matcher, test_image, test_detection):
        """Test extracting features for a single detection."""
        feature = feature_matcher.extract_detection_features(
            test_image, "test.jpg", test_detection, 0
        )
        
        assert feature is not None
        assert feature.image_name == "test.jpg"
        assert feature.detection_index == 0
        assert feature.detection == test_detection
        assert feature.embedding is not None
        assert feature.embedding.shape == (256,)
        assert feature.mask is not None
        assert feature.mask.shape == (100, 150)
        assert 0.0 <= feature.confidence <= 1.0
    
    def test_cache_operations(self, feature_matcher_with_cache, test_image, test_detection):
        """Test caching of detection features."""
        # Extract features (should save to cache)
        feature1 = feature_matcher_with_cache.extract_detection_features(
            test_image, "test.jpg", test_detection, 0
        )
        assert feature1 is not None
        
        # Extract again (should load from cache)
        feature2 = feature_matcher_with_cache.extract_detection_features(
            test_image, "test.jpg", test_detection, 0
        )
        assert feature2 is not None
        
        # Features should be the same (from cache)
        assert feature1.image_name == feature2.image_name
        assert feature1.detection_index == feature2.detection_index
        assert np.array_equal(feature1.embedding, feature2.embedding)
    
    def test_get_detection_feature(self, feature_matcher, test_image, test_detection):
        """Test getting stored detection feature."""
        # Initially should return None
        feature = feature_matcher.get_detection_feature("test.jpg", 0)
        assert feature is None
        
        # Add a feature
        extracted_feature = feature_matcher.extract_detection_features(
            test_image, "test.jpg", test_detection, 0
        )
        feature_matcher.detection_features.append(extracted_feature)
        
        # Should now find it
        feature = feature_matcher.get_detection_feature("test.jpg", 0)
        assert feature is not None
        assert feature.image_name == "test.jpg"
        assert feature.detection_index == 0
    
    def test_find_similar_objects_empty(self, feature_matcher):
        """Test finding similar objects with no features."""
        reference_embedding = np.random.rand(256).astype(np.float32)
        
        results = feature_matcher.find_similar_objects(reference_embedding)
        assert results == []
    
    def test_find_similar_objects(self, feature_matcher, test_image, test_detection):
        """Test finding similar objects."""
        # Create some test features
        features = []
        for i in range(3):
            embedding = np.random.rand(256).astype(np.float32)
            feature = DetectionFeature(
                image_name=f"image{i}.jpg",
                detection_index=0,
                detection=test_detection,
                embedding=embedding,
                confidence=0.8
            )
            features.append(feature)
        
        feature_matcher.detection_features = features
        
        # Search with first embedding as reference
        reference_embedding = features[0].embedding
        results = feature_matcher.find_similar_objects(
            reference_embedding, similarity_threshold=0.0, max_results=10
        )
        
        assert len(results) >= 1  # Should find at least the reference itself
        assert all(isinstance(r, SimilarityResult) for r in results)
        assert all(0.0 <= r.similarity_score <= 1.0 for r in results)
        
        # Results should be sorted by similarity (highest first)
        similarities = [r.similarity_score for r in results]
        assert similarities == sorted(similarities, reverse=True)
    
    def test_find_similar_objects_with_threshold(self, feature_matcher):
        """Test finding similar objects with threshold filtering."""
        # Create features with known embeddings
        feature1 = DetectionFeature(
            image_name="image1.jpg",
            detection_index=0,
            detection=YOLODetection(1, 0.5, 0.5, 0.2, 0.2),
            embedding=np.array([1.0, 0.0] + [0.0] * 254, dtype=np.float32),  # Similar to reference
            confidence=0.8
        )
        
        feature2 = DetectionFeature(
            image_name="image2.jpg",
            detection_index=0,
            detection=YOLODetection(1, 0.3, 0.3, 0.1, 0.1),
            embedding=np.array([0.0, 1.0] + [0.0] * 254, dtype=np.float32),  # Different from reference
            confidence=0.8
        )
        
        feature_matcher.detection_features = [feature1, feature2]
        
        # Reference embedding similar to feature1
        reference_embedding = np.array([1.0, 0.0] + [0.0] * 254, dtype=np.float32)
        
        # Search with high threshold
        results = feature_matcher.find_similar_objects(
            reference_embedding, similarity_threshold=0.8, max_results=10
        )
        
        # Should only find feature1 (similar to reference)
        assert len(results) >= 1
        assert results[0].image_name == "image1.jpg"
    
    def test_get_statistics_empty(self, feature_matcher):
        """Test getting statistics with no features."""
        stats = feature_matcher.get_statistics()
        
        assert stats["total_features"] == 0
        assert stats["total_images"] == 0
        assert stats["avg_confidence"] == 0.0
        assert stats["class_distribution"] == {}
    
    def test_get_statistics_with_features(self, feature_matcher):
        """Test getting statistics with features."""
        # Add some test features
        features = [
            DetectionFeature(
                image_name="image1.jpg",
                detection_index=0,
                detection=YOLODetection(0, 0.5, 0.5, 0.2, 0.2),
                embedding=np.random.rand(256).astype(np.float32),
                confidence=0.8
            ),
            DetectionFeature(
                image_name="image1.jpg",
                detection_index=1,
                detection=YOLODetection(1, 0.3, 0.3, 0.1, 0.1),
                embedding=np.random.rand(256).astype(np.float32),
                confidence=0.9
            ),
            DetectionFeature(
                image_name="image2.jpg",
                detection_index=0,
                detection=YOLODetection(0, 0.7, 0.7, 0.3, 0.3),
                embedding=np.random.rand(256).astype(np.float32),
                confidence=0.7
            )
        ]
        
        feature_matcher.detection_features = features
        
        stats = feature_matcher.get_statistics()
        
        assert stats["total_features"] == 3
        assert stats["total_images"] == 2  # image1.jpg and image2.jpg
        assert abs(stats["avg_confidence"] - 0.8) < 1e-6  # (0.8 + 0.9 + 0.7) / 3
        assert stats["class_distribution"] == {0: 2, 1: 1}  # 2 detections of class 0, 1 of class 1