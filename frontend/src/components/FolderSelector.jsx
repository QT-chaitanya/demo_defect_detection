import React, { useState } from 'react';
import './FolderSelector.css';

const FolderSelector = ({ onFolderSelect, isProcessing }) => {
  const [folderPath, setFolderPath] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (folderPath.trim()) {
      onFolderSelect(folderPath.trim());
    }
  };

  return (
    <div className="folder-selector">
      <h3>Select Image Folder</h3>
      <form onSubmit={handleSubmit}>
        <div className="input-group">
          <input
            type="text"
            value={folderPath}
            onChange={(e) => setFolderPath(e.target.value)}
            placeholder="Enter folder path (e.g., /path/to/images)"
            className="folder-input"
            disabled={isProcessing}
          />
          <button 
            type="submit" 
            disabled={!folderPath.trim() || isProcessing}
            className="select-btn"
          >
            {isProcessing ? 'Processing...' : 'Select Folder'}
          </button>
        </div>
      </form>
      
      <div className="help-text">
        <p>Enter the full path to a folder containing images.</p>
        <p>Supported formats: JPG, JPEG, PNG, BMP, TIFF</p>
      </div>
    </div>
  );
};

export default FolderSelector;
