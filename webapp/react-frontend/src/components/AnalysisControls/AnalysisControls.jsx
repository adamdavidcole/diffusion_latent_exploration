import React from 'react';
import { useApp } from '../../context/AppContext';
import './AnalysisControls.css';

const AnalysisControls = () => {
  const { state, actions } = useApp();
  const { analysisViewBy } = state;

  const handleViewByChange = (viewBy) => {
    actions.setAnalysisViewBy(viewBy);
  };

  return (
    <div className="analysis-controls">
      <div className="view-by-selector">
        <label>View by:</label>
        <div className="view-by-buttons">
          <button
            className={`view-by-button ${analysisViewBy === 'metric' ? 'selected' : ''}`}
            onClick={() => handleViewByChange('metric')}
          >
            Metric
          </button>
          <button
            className={`view-by-button ${analysisViewBy === 'prompt' ? 'selected' : ''}`}
            onClick={() => handleViewByChange('prompt')}
          >
            Prompt
          </button>
        </div>
      </div>
    </div>
  );
};

export default AnalysisControls;
