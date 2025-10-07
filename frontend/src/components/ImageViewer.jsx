import React from 'react';
import './ImageViewer.css';

const ImageViewer = ({ originalImage, annotatedImage, imageName, isProcessing }) => {
  return (
    <div className="image-viewer">
      <div className="image-section">
        <div className="image-box">
          <h3>Original Image</h3>
          <div className="image-container">
            {originalImage ? (
              <img 
                src={originalImage} 
                alt="Original" 
                className="image"
                style={{ objectFit: 'contain' }}
              />
            ) : (
              <div className="placeholder">
                {isProcessing ? 'Processing...' : 'No image selected'}
              </div>
            )}
          </div>
        </div>

        <div className="image-box">
          <h3>Detection Result</h3>
          <div className="image-container">
            {annotatedImage ? (
              <img 
                src={annotatedImage} 
                alt="Detection Result" 
                className="image"
                style={{ objectFit: 'contain' }}
              />
            ) : (
              <div className="placeholder">
                {isProcessing ? 'Processing...' : 'No result available'}
              </div>
            )}
          </div>
        </div>
      </div>

      {imageName && (
        <div className="image-info">
          <span className="image-name">{imageName}</span>
        </div>
      )}
    </div>
  );
};

export default ImageViewer;
