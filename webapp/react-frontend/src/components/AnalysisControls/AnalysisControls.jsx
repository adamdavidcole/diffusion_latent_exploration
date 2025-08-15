import React from 'react';
import { useApp } from '../../context/AppContext';
import './AnalysisControls.css';

const AnalysisControls = () => {
  const { state, actions } = useApp();
  const { analysisViewBy, analysisChartSize } = state;

  const handleViewByChange = (viewBy) => {
    actions.setAnalysisViewBy(viewBy);
  };

  const handleChartSizeChange = (size) => {
    if (actions.setAnalysisChartSize) {
      actions.setAnalysisChartSize(size);
    }
  };

  const chartSize = analysisChartSize || 250;

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
      
      <div className="chart-size-control">
        <label>
          Chart Size: {chartSize}px
          <input
            type="range"
            min="150"
            max="400"
            value={chartSize}
            onChange={(e) => handleChartSizeChange(Number(e.target.value))}
            className="size-slider"
          />
        </label>
      </div>
    </div>
  );
};

export default AnalysisControls;
