#!/usr/bin/env python3
"""
Inference Service for Defect Detection
=====================================
"""

import cv2
import os
import base64
import tempfile
import logging
from pathlib import Path
from ultralytics import YOLO

class InferenceService:
    def __init__(self):
        self.model_path = "/home/qualitas/Desktop/demo/silicone_trained_model/weights/best.pt"
        self.confidence_threshold = 0.4
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained YOLO model"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                self.logger.info(f"Model loaded successfully from: {self.model_path}")
            else:
                self.logger.error(f"Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise e
    
    def process_image(self, image_path):
        """Process a single image for defect detection"""
        try:
            # Validate image file
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'error': f'Image file not found: {image_path}',
                    'defect_count': 0,
                    'original_image': None,
                    'annotated_image': None
                }
            
            # Run inference
            results = self.model(image_path, conf=self.confidence_threshold)
            
            # Process results
            defect_count = 0
            annotated_image = None
            
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    defect_count = len(r.boxes)
                    # Get annotated image
                    annotated_image = r.plot()
                    break
            
            # If no defects, use original image
            if annotated_image is None:
                annotated_image = cv2.imread(image_path)
            
            # Convert images to base64
            original_b64 = self.image_to_base64(image_path)
            annotated_b64 = self.image_to_base64_cv2(annotated_image)
            
            return {
                'success': True,
                'defect_count': defect_count,
                'original_image': original_b64,
                'annotated_image': annotated_b64,
                'image_name': os.path.basename(image_path),
                'status': 'PASSED' if defect_count == 0 else 'FAILED'
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'defect_count': 0,
                'original_image': None,
                'annotated_image': None
            }
    
    def image_to_base64(self, image_path):
        """Convert image file to base64 string"""
        try:
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                img_b64 = base64.b64encode(img_data).decode('utf-8')
                return f"data:image/jpeg;base64,{img_b64}"
        except Exception as e:
            self.logger.error(f"Error converting image to base64: {str(e)}")
            return None
    
    def image_to_base64_cv2(self, cv_image):
        """Convert OpenCV image to base64 string"""
        try:
            # Encode image as JPEG
            _, buffer = cv2.imencode('.jpg', cv_image)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{img_b64}"
        except Exception as e:
            self.logger.error(f"Error converting CV image to base64: {str(e)}")
            return None
    
    def update_confidence(self, new_confidence):
        """Update confidence threshold"""
        if 0.0 <= new_confidence <= 1.0:
            self.confidence_threshold = new_confidence
            self.logger.info(f"Confidence threshold updated to: {new_confidence}")
            return True
        else:
            self.logger.error(f"Invalid confidence threshold: {new_confidence}")
            return False
