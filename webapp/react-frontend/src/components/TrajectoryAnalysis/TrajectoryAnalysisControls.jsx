import React from 'react';
import './TrajectoryAnalysisControls.css';

const TrajectoryAnalysisControls = ({ availableNormalizations, activeNormalization, onNormalizationChange }) => {
  return (
    <div className="trajectory-analysis-controls">
      <div className="normalization-selector">
        <label>Normalization:</label>
        <div className="normalization-buttons">
          {availableNormalizations.map(norm => (
            <button
              key={norm.key}
              className={`normalization-button ${activeNormalization === norm.key ? 'selected' : ''}`}
              onClick={() => onNormalizationChange(norm.key)}
            >
              {norm.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default TrajectoryAnalysisControls;
