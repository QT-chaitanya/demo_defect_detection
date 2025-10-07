import React from 'react';
import './StatusIndicator.css';

const StatusIndicator = ({ status, passedCount, failedCount }) => {
  return (
    <div className="status-indicator">
      <div className={`status-box passed-box ${status === 'PASSED' ? 'blink-green' : ''}`}>
        <div className="status-label">PASSED</div>
        <div className="status-count">{passedCount}</div>
      </div>
      
      <div className={`status-box failed-box ${status === 'FAILED' ? 'blink-red' : ''}`}>
        <div className="status-label">FAILED</div>
        <div className="status-count">{failedCount}</div>
      </div>
    </div>
  );
};

export default StatusIndicator;
