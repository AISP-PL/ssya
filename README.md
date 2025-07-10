# SAM Viewer - YOLO Annotation Viewer with SAM2 Integration

A powerful Qt5-based application for viewing YOLO annotations and finding similar objects using SAM2 (Segment Anything Model 2).

## Features

### ✅ Core Features (Implemented)
- **YOLO Annotation Support**: Load and display YOLOv5 format annotations
- **Interactive Image Browsing**: Navigate through datasets with arrow controls
- **Detection Visualization**: Visual overlay of bounding boxes with class labels
- **Click-to-Select**: Click on detections or select from list
- **SAM Integration**: Mock SAM2 interface ready for real model integration
- **Feature Extraction**: Extract embeddings from detected objects
- **Similarity Search**: Find similar objects across the entire dataset
- **Threshold Filtering**: Filter results by similarity threshold
- **Object Grouping**: Name and save groups of similar objects
- **Caching System**: Cache extracted features for faster subsequent runs

### 🔄 Coming Soon
- Real SAM2 model integration (currently using mock implementation)
- Advanced filtering options
- Batch processing capabilities
- Export to various formats (COCO, CVAT, etc.)

## Installation

### Requirements
- Python >= 3.11, < 3.12
- PyQt5 >= 5.15.7
- OpenCV >= 4.8.0
- NumPy >= 1.24.0
- Pillow >= 10.0.0

### Install from Source
```bash
git clone https://github.com/AISP-PL/ssya.git
cd ssya
pip install -e .
```

## Usage

### Dataset Structure
Your dataset should be organized as follows:
```
dataset/
├── images/           # Image files (.jpg, .png)
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels/           # YOLO annotation files (.txt)
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
└── classes.txt       # Class names (one per line)
```

### YOLO Annotation Format
Each `.txt` file should contain detections in YOLOv5 format:
```
class_id x_center y_center width height
```
Where all coordinates are normalized (0.0-1.0).

### Running the Application
```bash
# Using the installed script
sam-viewer --dataset /path/to/your/dataset

# Or using Python module
python -m ssya.sam_viewer.main --dataset /path/to/your/dataset

# With verbose logging
sam-viewer --dataset /path/to/your/dataset --verbose
```

### Basic Workflow
1. **Load Dataset**: Start the application with your dataset path
2. **Browse Images**: Use arrow buttons or navigate through the image list
3. **Select Detection**: Click on a bounding box or select from the detection list
4. **Find Similar**: Click "Find Similar Objects" to extract features and search
5. **Filter Results**: Use the threshold slider to filter similarity results
6. **Name Groups**: Save groups of similar objects with custom names

## Application Interface

### Main Window Components
- **Image Display**: Shows current image with YOLO bounding boxes
- **Navigation Controls**: Previous/Next buttons for browsing
- **Detection List**: Lists all detections in current image
- **SAM Controls**: Find similar, apply threshold, name objects
- **Similarity Results**: Shows found similar objects
- **Status Information**: Current operation status and statistics

### Key Features
- **Visual Feedback**: Selected detections are highlighted in yellow
- **SAM Mask Overlay**: Generated masks are overlaid on images (orange with yellow contours)
- **Filtered Navigation**: When threshold is applied, navigation is limited to matching images
- **Progress Indication**: Background feature extraction shows progress
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Technical Architecture

### Core Modules
- **`yolo_parser.py`**: Handles YOLO annotation parsing and validation
- **`image_navigator.py`**: Manages image loading and navigation
- **`sam_interface.py`**: SAM2 model interface (currently mock implementation)
- **`feature_matcher.py`**: Feature extraction and similarity search
- **`main_window.py`**: Qt5 GUI implementation

### Features
- **Thread-Safe**: Background processing doesn't block the UI
- **Caching**: Extracted features are cached for performance
- **Error Handling**: Robust error handling for invalid data
- **Memory Efficient**: Processes images on-demand
- **Extensible**: Modular design for easy feature additions

## Development

### Running Tests
```bash
# Run all tests
python -m pytest

# Run only SAM Viewer tests
python -m pytest tests/unit/sam_viewer/ -v

# Test specific module
python test_modules.py
```

### Test Coverage
- 40+ unit tests covering all core functionality
- Mock-based testing for SAM integration
- Comprehensive error case coverage
- Performance and memory tests

## Example Output

### Metadata Export
When you name an object group, it saves to `dataset/output/{group_name}_group.json`:
```json
{
  "group_name": "red_cars",
  "created_at": "2024-01-15T10:30:00",
  "threshold": 0.75,
  "total_objects": 12,
  "objects": [
    {
      "image_name": "image1.jpg",
      "detection_index": 0,
      "class_id": 1,
      "class_name": "car",
      "bbox": {
        "x_center": 0.5,
        "y_center": 0.6,
        "width": 0.2,
        "height": 0.3
      },
      "similarity_score": 0.87
    }
  ]
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Future Integration

The application is designed to easily integrate with real SAM2 models. To replace the mock implementation:

1. Install SAM2 dependencies
2. Update `sam_interface.py` to use real SAM2 models
3. Replace mock prediction methods with actual SAM2 calls

## License

This project is part of the AISP-PL organization. See LICENSE for details.

