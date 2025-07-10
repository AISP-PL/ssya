# Changelog

All notable changes to the SAM Viewer project will be documented in this file.

## [1.0.0] - 2024-01-15

### Added
- **SAM Viewer Application**: Complete Qt5-based GUI application for YOLO annotation viewing with SAM2 integration
- **YOLO Parser Module**: Full support for YOLOv5 annotation format
  - Parse `.txt` annotation files with format: `class_id x_center y_center width height`
  - Load class definitions from `classes.txt`
  - Validate annotation ranges and handle errors gracefully
  - Convert YOLO format to pixel coordinates for visualization
- **Image Navigator**: Comprehensive image browsing functionality
  - Load images from directories (supports .jpg, .png, .bmp, .tiff)
  - Navigate with Previous/Next controls
  - Display "Image X of Y" information
  - Click-to-select detections on images
  - Draw bounding boxes with class labels
- **SAM Interface**: Mock SAM2 integration ready for real model replacement
  - Predict masks from bounding box prompts
  - Predict masks from point prompts
  - Extract feature embeddings from masked regions
  - Compute similarity metrics (cosine and Euclidean distance)
- **Feature Matcher**: Advanced similarity search capabilities
  - Extract features from all detections in dataset
  - Background processing with progress indication
  - Feature caching system for improved performance
  - Find similar objects across entire dataset
  - Threshold-based filtering of results
- **GUI Components**: Rich user interface with Qt5
  - Main image display with zoom and scroll
  - Detection list with click selection
  - SAM controls (Find Similar, Apply Threshold, Name Objects)
  - Similarity results browser
  - Progress bars for background operations
  - Status information and logging
- **Object Grouping**: Save and export similar object groups
  - Name groups of similar objects
  - Export metadata to JSON format
  - Include similarity scores and detection details
- **Navigation Filtering**: Advanced navigation modes
  - Normal navigation through all images
  - Filtered navigation showing only images with similar objects
  - Threshold-based filtering with real-time updates
- **Comprehensive Testing**: 40+ unit tests
  - Test coverage for all core modules
  - Mock-based testing for SAM integration
  - Error handling and edge case validation
  - Performance and memory efficiency tests

### Technical Features
- **Thread-Safe Design**: Background processing doesn't block UI
- **Memory Efficient**: Images loaded on-demand
- **Robust Error Handling**: Graceful handling of invalid annotations and missing files
- **Extensible Architecture**: Modular design for easy feature additions
- **Caching System**: Feature embeddings cached to disk for faster subsequent runs
- **Logging System**: Comprehensive logging for debugging and monitoring

### CLI Interface
- **Command Line Interface**: Run with `sam-viewer --dataset /path/to/dataset`
- **Flexible Options**: Verbose logging, log level control
- **Dataset Validation**: Automatic validation of dataset structure

### Documentation
- **Complete README**: Comprehensive documentation with usage examples
- **Code Documentation**: Extensive docstrings and comments
- **Type Hints**: Full type annotation for better development experience

## [Future Releases]

### Planned Features
- **Real SAM2 Integration**: Replace mock implementation with actual SAM2 models
- **Advanced Export Options**: Support for COCO, CVAT, Roboflow formats
- **Batch Processing**: Process multiple datasets in parallel
- **Visual Analytics**: PCA/UMAP visualization of embeddings
- **Performance Optimizations**: GPU acceleration and model optimization
- **Additional Similarity Metrics**: More sophisticated similarity measures
- **Annotation Editing**: Ability to modify and save annotations
- **Integration APIs**: REST API for integration with other tools