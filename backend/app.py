#!/usr/bin/env python3
"""
Flask Backend for Defect Detection Demo
=====================================
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
import random
from datetime import datetime
from pathlib import Path
import base64
from inference_service import InferenceService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/qualitas/Desktop/demo/backend/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize inference service
inference_service = InferenceService()

@app.route('/api/select_folder', methods=['POST'])
def select_folder():
    """Select image folder and get list of images"""
    try:
        data = request.get_json()
        folder_path = data.get('folder_path')
        
        if not folder_path or not os.path.exists(folder_path):
            return jsonify({'error': 'Invalid folder path'}), 400
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for file_path in Path(folder_path).iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path.name)
        
        image_files.sort()  # Sort alphabetically first
        random.shuffle(image_files)  # Then shuffle randomly
        
        logger.info(f"Selected folder: {folder_path}, Found {len(image_files)} images (shuffled)")
        
        return jsonify({
            'success': True,
            'folder_path': folder_path,
            'images': image_files,
            'count': len(image_files)
        })
        
    except Exception as e:
        logger.error(f"Error selecting folder: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_image', methods=['POST'])
def process_image():
    """Process a single image for defect detection"""
    try:
        data = request.get_json()
        folder_path = data.get('folder_path')
        image_name = data.get('image_name')
        
        if not folder_path or not image_name:
            return jsonify({'error': 'Missing folder_path or image_name'}), 400
        
        image_path = os.path.join(folder_path, image_name)
        
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404
        
        # Process image
        result = inference_service.process_image(image_path)
        
        if result['success']:
            logger.info(f"Successfully processed: {image_name}, Defects: {result['defect_count']}")
            return jsonify(result)
        else:
            logger.error(f"Failed to process: {image_name}, Error: {result['error']}")
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error processing image {image_name}: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/get_image/<path:image_path>')
def get_image(image_path):
    """Serve image files"""
    try:
        if os.path.exists(image_path):
            return send_file(image_path)
        else:
            return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    logger.info("Starting Defect Detection Demo Backend...")
    app.run(host='0.0.0.0', port=5000, debug=True)
