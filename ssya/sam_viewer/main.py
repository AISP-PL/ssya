#!/usr/bin/env python3
"""
SAM Viewer - Main Application Entry Point

A GUI application for viewing images with YOLO annotations and finding similar objects using SAM2.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import QApplication, QMessageBox

from .ui.main_window import MainWindow


logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def validate_dataset_path(dataset_path: str) -> tuple[bool, str]:
    """
    Validate that the dataset path contains required structure.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(dataset_path)
    
    if not path.exists():
        return False, f"Dataset path does not exist: {dataset_path}"
    
    if not path.is_dir():
        return False, f"Dataset path is not a directory: {dataset_path}"
    
    # Check for required subdirectories
    images_dir = path / "images"
    labels_dir = path / "labels"
    
    if not images_dir.exists():
        return False, f"Images directory not found: {images_dir}"
    
    if not labels_dir.exists():
        return False, f"Labels directory not found: {labels_dir}"
    
    # Check for classes.txt file
    classes_file = path / "classes.txt"
    if not classes_file.exists():
        return False, f"Classes file not found: {classes_file}"
    
    return True, ""


def main() -> None:
    """Main function for SAM Viewer application."""
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="SAM Viewer - View images with YOLO annotations and find similar objects using SAM2"
    )
    parser.add_argument(
        "-d", "--dataset", 
        type=str, 
        required=True,
        help="Path to dataset directory containing images/, labels/, and classes.txt"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else args.log_level
    setup_logging(log_level)
    
    logger.info("Starting SAM Viewer application")
    logger.info(f"Dataset path: {args.dataset}")
    
    # Validate dataset path
    is_valid, error_msg = validate_dataset_path(args.dataset)
    if not is_valid:
        logger.error(f"Dataset validation failed: {error_msg}")
        print(f"Error: {error_msg}")
        sys.exit(1)
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("SAM Viewer")
    app.setApplicationVersion("1.0.0")
    
    try:
        # Create and show main window
        main_window = MainWindow(args.dataset)
        main_window.show()
        
        logger.info("Application started successfully")
        
        # Run application event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        QMessageBox.critical(None, "Error", f"Failed to start application:\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()