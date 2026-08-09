"""
WALDO Detector Module
====================

This module provides the core detection functionality using WALDO models.
Supports both YOLOv8 and YOLO11 (ultralytics>=8.3.0).

YOLO11 Improvements over YOLOv8:
- Enhanced feature extraction with improved backbone/neck architecture
- Better accuracy-speed tradeoff
- Improved small object detection
- Better export/quantization stability

Usage:
    # Use YOLO11 (recommended)
    detector = WALDODetector({
        'model_path': 'yolo11m.pt',
        'architecture': 'yolo11',
    })
    
    # Use YOLOv8 (legacy)
    detector = WALDODetector({
        'model_path': 'yolov8m.pt',
        'architecture': 'yolov8',
    })
"""

import os
import numpy as np
import torch
from ultralytics import YOLO

# YOLO11 model variants
YOLO11_VARIANTS = {
    'nano': 'yolo11n.pt',
    'small': 'yolo11s.pt',
    'medium': 'yolo11m.pt',
    'large': 'yolo11l.pt',
    'xlarge': 'yolo11x.pt',
}

# YOLOv8 model variants (legacy)
YOLOV8_VARIANTS = {
    'nano': 'yolov8n.pt',
    'small': 'yolov8s.pt',
    'medium': 'yolov8m.pt',
    'large': 'yolov8l.pt',
    'xlarge': 'yolov8x.pt',
}


class WALDODetector:
    """
    WALDO object detector using YOLO11 or YOLOv8 models.
    
    This class handles loading the WALDO model and performing object detection
    on images or video frames. YOLO11 is recommended for better performance.
    
    Attributes:
        architecture: 'yolo11' (default) or 'yolov8'
        variant: Model size variant (nano, small, medium, large, xlarge)
    """
    
    def __init__(self, config):
        """
        Initialize the WALDO detector.
        
        Args:
            config (dict): Configuration dictionary with the following keys:
                - model_path (str): Path to the WALDO model weights
                - architecture (str): 'yolo11' (default) or 'yolov8'
                - variant (str): Model size variant (nano, small, medium, large, xlarge)
                - confidence_threshold (float): Detection confidence threshold
                - device (str): Device to run inference on ('cuda', 'cpu')
                - precision (str): Model precision ('fp32', 'fp16', 'int8')
        """
        self.config = config
        self.architecture = config.get('architecture', 'yolo11')  # Default to YOLO11
        self.variant = config.get('variant', 'medium')
        self.confidence_threshold = config.get('confidence_threshold', 0.25)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.precision = config.get('precision', 'fp16' if self.device == 'cuda' else 'fp32')
        
        # Determine model path
        model_path = config.get('model_path')
        if model_path is None:
            # Use default model based on architecture and variant
            if self.architecture == 'yolo11':
                model_path = YOLO11_VARIANTS.get(self.variant, 'yolo11m.pt')
            else:
                model_path = YOLOV8_VARIANTS.get(self.variant, 'yolov8m.pt')
        
        # Check if model exists (pretrained models are downloaded automatically)
        self.model_path = model_path
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """
        Load the WALDO model.
        
        Args:
            model_path (str): Path to the model weights
            
        Returns:
            YOLO: Loaded YOLO model
        """
        try:
            model = YOLO(model_path)
            # Set model parameters
            model.to(self.device)
            if self.precision == 'fp16' and self.device == 'cuda':
                model.half()  # Use half precision for faster inference
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load WALDO model: {str(e)}")
    
    def detect(self, frame, metadata=None):
        """
        Perform object detection on a single frame.
        
        Args:
            frame (numpy.ndarray): Input image frame
            metadata (dict, optional): Frame metadata
            
        Returns:
            list: List of detection results
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Invalid input frame")
        
        # Run inference
        results = self.model(frame, conf=self.confidence_threshold)
        
        # Convert results to standard format
        detections = self._process_results(results, metadata)
        
        return detections
    
    def _process_results(self, results, metadata=None):
        """
        Process YOLO results into standardized detection format.
        
        Args:
            results: YOLO detection results
            metadata (dict, optional): Frame metadata
            
        Returns:
            list: List of standardized detection dictionaries
        """
        detections = []
        
        # Process each detection
        for i, result in enumerate(results):
            boxes = result.boxes
            
            for j in range(len(boxes)):
                box = boxes[j]
                
                # Get coordinates (convert to int for pixel values)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Get confidence and class
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                # Create detection dictionary
                detection = {
                    'id': f"{i}_{j}",
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'frame_id': metadata.get('frame_id', 0) if metadata else 0,
                    'timestamp': metadata.get('timestamp', None) if metadata else None,
                    'source': metadata.get('source', 'unknown') if metadata else 'unknown'
                }
                
                detections.append(detection)
        
        return detections
