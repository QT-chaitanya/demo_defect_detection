import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';
import ImageViewer from './components/ImageViewer';
import StatusIndicator from './components/StatusIndicator';
import FolderSelector from './components/FolderSelector';

function App() {
  const [folderPath, setFolderPath] = useState('');
  const [images, setImages] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [currentImage, setCurrentImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isLooping, setIsLooping] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('IDLE');
  const [passedCount, setPassedCount] = useState(0);
  const [failedCount, setFailedCount] = useState(0);
  const intervalRef = useRef(null);

  // Auto-loop effect
  useEffect(() => {
    if (isLooping && images.length > 0) {
      intervalRef.current = setInterval(() => {
        setCurrentImageIndex(prev => (prev + 1) % images.length);
      }, 3000);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isLooping, images.length]);

  // Process current image when index changes
  useEffect(() => {
    if (images.length > 0 && folderPath) {
      processCurrentImage();
    }
  }, [currentImageIndex, images, folderPath]);

  const handleFolderSelect = async (path) => {
    try {
      setError('');
      setIsProcessing(true);
      
      const response = await axios.post('/api/select_folder', {
        folder_path: path
      });

      if (response.data.success) {
        setFolderPath(path);
        setImages(response.data.images);
        setCurrentImageIndex(0);
        setStatus('READY');
        // Reset counters when new folder is selected
        setPassedCount(0);
        setFailedCount(0);
      } else {
        setError(response.data.error || 'Failed to select folder');
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Error selecting folder');
    } finally {
      setIsProcessing(false);
    }
  };

  const processCurrentImage = async () => {
    if (!images.length || !folderPath) return;

    try {
      setIsProcessing(true);
      setError('');

      const response = await axios.post('/api/process_image', {
        folder_path: folderPath,
        image_name: images[currentImageIndex]
      });

      if (response.data.success) {
        setCurrentImage(response.data);
        setStatus(response.data.status);
        
        // Update counters based on status
        if (response.data.status === 'PASSED') {
          setPassedCount(prev => prev + 1);
        } else if (response.data.status === 'FAILED') {
          setFailedCount(prev => prev + 1);
        }
      } else {
        setError(`Failed to process ${images[currentImageIndex]}: ${response.data.error}`);
        // Skip to next image on error
        setTimeout(() => {
          setCurrentImageIndex(prev => (prev + 1) % images.length);
        }, 1000);
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Error processing image');
      // Skip to next image on error
      setTimeout(() => {
        setCurrentImageIndex(prev => (prev + 1) % images.length);
      }, 1000);
    } finally {
      setIsProcessing(false);
    }
  };

  const toggleLoop = () => {
    setIsLooping(!isLooping);
  };

  const nextImage = () => {
    if (images.length > 0) {
      setCurrentImageIndex(prev => (prev + 1) % images.length);
    }
  };

  const prevImage = () => {
    if (images.length > 0) {
      setCurrentImageIndex(prev => (prev - 1 + images.length) % images.length);
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <div className="header">
        <div className="header-content">
          <img 
            src="/qualitas_logo_white.png" 
            alt="Qualitas Technologies" 
            className="header-logo"
          />
          <h1>SILICON TUBE DEFECT DETECTION</h1>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        {/* Left Panel - Images */}
        <div className="image-panel">
          <div className="image-container">
            <ImageViewer 
              originalImage={currentImage?.original_image}
              annotatedImage={currentImage?.annotated_image}
              imageName={currentImage?.image_name}
              isProcessing={isProcessing}
            />
          </div>
        </div>

        {/* Right Panel - Controls and Status */}
        <div className="control-panel">
          {/* Folder Selection */}
          <FolderSelector 
            onFolderSelect={handleFolderSelect}
            isProcessing={isProcessing}
          />

          {/* Status Indicators */}
          <div className="status-section">
            <StatusIndicator 
              status={status}
              passedCount={passedCount}
              failedCount={failedCount}
            />
          </div>

          {/* Controls */}
          <div className="controls">
            <button 
              onClick={toggleLoop}
              className={`control-btn ${isLooping ? 'active' : ''}`}
            >
              {isLooping ? 'Stop Loop' : 'Start Loop'}
            </button>
            
            <button 
              onClick={prevImage}
              disabled={!images.length || isProcessing}
              className="control-btn"
            >
              Previous
            </button>
            
            <button 
              onClick={nextImage}
              disabled={!images.length || isProcessing}
              className="control-btn"
            >
              Next
            </button>
          </div>

          {/* Image Counter */}
          {images.length > 0 && (
            <div className="image-counter">
              Image {currentImageIndex + 1} of {images.length}
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="footer">
        <div className="footer-content">
          <img 
            src="/qualitas_logo_white.png" 
            alt="Qualitas Technologies" 
            className="footer-logo"
          />
          <span>Powered by <strong>Qualitas Technologies</strong></span>
        </div>
        <span>{new Date().toLocaleString()}</span>
      </div>
    </div>
  );
}

export default App;
