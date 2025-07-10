"""Feature matching module for finding similar objects."""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, NamedTuple
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .yolo_parser import YOLODetection
from .sam_interface import SAMInterface

logger = logging.getLogger(__name__)


class DetectionFeature(NamedTuple):
    """Represents a detection with its features."""
    image_name: str
    detection_index: int
    detection: YOLODetection
    embedding: np.ndarray
    mask: Optional[np.ndarray] = None
    confidence: float = 0.0


class SimilarityResult(NamedTuple):
    """Represents a similarity search result."""
    image_name: str
    detection_index: int
    detection: YOLODetection
    similarity_score: float
    embedding: np.ndarray


class FeatureMatcher:
    """Handles feature extraction and similarity matching."""
    
    def __init__(self, sam_interface: SAMInterface, cache_dir: Optional[Path] = None):
        """
        Initialize feature matcher.
        
        Args:
            sam_interface: SAM interface for mask generation and feature extraction
            cache_dir: Directory to cache embeddings (optional)
        """
        self.sam_interface = sam_interface
        self.cache_dir = cache_dir
        self.detection_features: List[DetectionFeature] = []
        self._processing_lock = threading.Lock()
        
        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, image_name: str, detection_index: int) -> Optional[Path]:
        """Get cache file path for a detection."""
        if not self.cache_dir:
            return None
        
        # Create safe filename
        safe_name = image_name.replace(".", "_").replace("/", "_")
        cache_filename = f"{safe_name}_det{detection_index}.json"
        return self.cache_dir / cache_filename
    
    def _save_to_cache(self, detection_feature: DetectionFeature):
        """Save detection feature to cache."""
        cache_path = self._get_cache_path(detection_feature.image_name, detection_feature.detection_index)
        if not cache_path:
            return
        
        try:
            cache_data = {
                "image_name": detection_feature.image_name,
                "detection_index": detection_feature.detection_index,
                "detection": {
                    "class_id": detection_feature.detection.class_id,
                    "x_center": detection_feature.detection.x_center,
                    "y_center": detection_feature.detection.y_center,
                    "width": detection_feature.detection.width,
                    "height": detection_feature.detection.height
                },
                "embedding": detection_feature.embedding.tolist(),
                "confidence": detection_feature.confidence
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
                
            logger.debug(f"Saved feature cache: {cache_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save feature cache: {e}")
    
    def _load_from_cache(self, image_name: str, detection_index: int, detection: YOLODetection) -> Optional[DetectionFeature]:
        """Load detection feature from cache."""
        cache_path = self._get_cache_path(image_name, detection_index)
        if not cache_path or not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            # Verify detection matches
            cached_det = cache_data["detection"]
            if (cached_det["class_id"] != detection.class_id or
                abs(cached_det["x_center"] - detection.x_center) > 1e-6 or
                abs(cached_det["y_center"] - detection.y_center) > 1e-6 or
                abs(cached_det["width"] - detection.width) > 1e-6 or
                abs(cached_det["height"] - detection.height) > 1e-6):
                logger.debug(f"Cache mismatch for {image_name}:{detection_index}, regenerating")
                return None
            
            embedding = np.array(cache_data["embedding"], dtype=np.float32)
            confidence = cache_data.get("confidence", 0.0)
            
            logger.debug(f"Loaded feature cache: {cache_path}")
            
            return DetectionFeature(
                image_name=image_name,
                detection_index=detection_index,
                detection=detection,
                embedding=embedding,
                confidence=confidence
            )
            
        except Exception as e:
            logger.warning(f"Failed to load feature cache {cache_path}: {e}")
            return None
    
    def extract_detection_features(self, image: np.ndarray, image_name: str, detection: YOLODetection, 
                                 detection_index: int) -> Optional[DetectionFeature]:
        """
        Extract features for a single detection.
        
        Args:
            image: Input image (BGR format)
            image_name: Name of the image file
            detection: YOLO detection
            detection_index: Index of detection within image
            
        Returns:
            DetectionFeature object or None if failed
        """
        # Try loading from cache first
        cached_feature = self._load_from_cache(image_name, detection_index, detection)
        if cached_feature:
            return cached_feature
        
        try:
            # Set image for SAM
            if not self.sam_interface.set_image(image):
                logger.error(f"Failed to set image for SAM: {image_name}")
                return None
            
            # Convert detection to bbox
            height, width = image.shape[:2]
            bbox = detection.to_bbox(width, height)
            
            # Generate mask and embedding
            mask, embedding, confidence = self.sam_interface.predict_mask(bbox)
            
            if mask is None or embedding is None:
                logger.error(f"Failed to generate mask/embedding for {image_name}:{detection_index}")
                return None
            
            # Create detection feature
            detection_feature = DetectionFeature(
                image_name=image_name,
                detection_index=detection_index,
                detection=detection,
                embedding=embedding,
                mask=mask,
                confidence=confidence
            )
            
            # Save to cache
            self._save_to_cache(detection_feature)
            
            logger.debug(f"Extracted features for {image_name}:{detection_index}")
            return detection_feature
            
        except Exception as e:
            logger.error(f"Failed to extract features for {image_name}:{detection_index}: {e}")
            return None
    
    def process_dataset(self, image_navigator, progress_callback=None) -> bool:
        """
        Process entire dataset to extract features for all detections.
        
        Args:
            image_navigator: ImageNavigator instance
            progress_callback: Optional callback function for progress updates (current, total)
            
        Returns:
            True if successful, False otherwise
        """
        with self._processing_lock:
            self.detection_features.clear()
            
            total_images = image_navigator.total_images
            processed_images = 0
            
            logger.info(f"Starting feature extraction for {total_images} images")
            
            # Save current position
            original_index = image_navigator.current_index
            
            try:
                # Process each image
                for img_index in range(total_images):
                    image_navigator.go_to_image(img_index)
                    
                    # Load image
                    image = image_navigator.load_current_image()
                    if image is None:
                        logger.warning(f"Failed to load image at index {img_index}")
                        continue
                    
                    image_name = image_navigator.current_image_name
                    detections = image_navigator.current_detections
                    
                    logger.debug(f"Processing {image_name} with {len(detections)} detections")
                    
                    # Process each detection in this image
                    for det_index, detection in enumerate(detections):
                        feature = self.extract_detection_features(
                            image, image_name, detection, det_index
                        )
                        
                        if feature:
                            self.detection_features.append(feature)
                    
                    processed_images += 1
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(processed_images, total_images)
                
                # Restore original position
                image_navigator.go_to_image(original_index)
                
                total_features = len(self.detection_features)
                logger.info(f"Feature extraction complete: {total_features} features extracted from {processed_images} images")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to process dataset: {e}")
                # Restore original position
                image_navigator.go_to_image(original_index)
                return False
    
    def find_similar_objects(self, reference_embedding: np.ndarray, 
                           similarity_threshold: float = 0.0,
                           max_results: int = 100,
                           metric: str = "cosine") -> List[SimilarityResult]:
        """
        Find objects similar to the reference embedding.
        
        Args:
            reference_embedding: Reference feature embedding
            similarity_threshold: Minimum similarity score (0.0-1.0)
            max_results: Maximum number of results to return
            metric: Similarity metric ("cosine" or "euclidean")
            
        Returns:
            List of SimilarityResult objects sorted by similarity (highest first)
        """
        with self._processing_lock:
            if not self.detection_features:
                logger.warning("No detection features available for similarity search")
                return []
            
            logger.info(f"Searching for similar objects among {len(self.detection_features)} features")
            
            # Compute similarities
            similarities = []
            
            for feature in self.detection_features:
                similarity = self.sam_interface.compute_similarity(
                    reference_embedding, feature.embedding, metric
                )
                
                if similarity >= similarity_threshold:
                    result = SimilarityResult(
                        image_name=feature.image_name,
                        detection_index=feature.detection_index,
                        detection=feature.detection,
                        similarity_score=similarity,
                        embedding=feature.embedding
                    )
                    similarities.append(result)
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Limit results
            similarities = similarities[:max_results]
            
            logger.info(f"Found {len(similarities)} similar objects (threshold: {similarity_threshold:.3f})")
            
            return similarities
    
    def get_detection_feature(self, image_name: str, detection_index: int) -> Optional[DetectionFeature]:
        """
        Get stored feature for specific detection.
        
        Args:
            image_name: Name of the image file
            detection_index: Index of detection within image
            
        Returns:
            DetectionFeature object or None if not found
        """
        with self._processing_lock:
            for feature in self.detection_features:
                if (feature.image_name == image_name and 
                    feature.detection_index == detection_index):
                    return feature
            return None
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about processed features.
        
        Returns:
            Dictionary with statistics
        """
        with self._processing_lock:
            if not self.detection_features:
                return {
                    "total_features": 0,
                    "total_images": 0,
                    "avg_confidence": 0.0,
                    "class_distribution": {}
                }
            
            # Calculate statistics
            total_features = len(self.detection_features)
            unique_images = len(set(f.image_name for f in self.detection_features))
            avg_confidence = np.mean([f.confidence for f in self.detection_features])
            
            # Class distribution
            class_counts = {}
            for feature in self.detection_features:
                class_id = feature.detection.class_id
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
            
            return {
                "total_features": total_features,
                "total_images": unique_images,
                "avg_confidence": float(avg_confidence),
                "class_distribution": class_counts
            }