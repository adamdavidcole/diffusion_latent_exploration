import React from 'react';
import { useApp } from '../../context/AppContext';
import './ExperimentHeader.css';

// Helper function to format duration
const formatDuration = (seconds) => {
  if (!seconds) return '';
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds.toFixed(1)}s`;
};

const ExperimentHeader = () => {
  const { state } = useApp();
  const { currentExperiment } = state;

  if (!currentExperiment) {
    return null;
  }

  return (
    <div className="experiment-header">
      <div className="experiment-title-row">
        <h2 id="experiment-title">{currentExperiment.name}</h2>
      </div>

      <div className="experiment-stats">
        {currentExperiment.model_id && (
          <span className="stat-item">
            {currentExperiment.model_id.split('/').pop().replace('-Diffusers', '').replace('Wan2.1-T2V-', '')}
          </span>
        )}
        <span className="stat-item">{currentExperiment.videos_count} videos</span>
        <span className="stat-item">{currentExperiment.variations_count} variations</span>
        <span className="stat-item">{currentExperiment.seeds_count} seeds</span>
        {currentExperiment.duration_seconds && (
          <span className="stat-item">{formatDuration(currentExperiment.duration_seconds)} duration</span>
        )}
        {currentExperiment.has_vlm_analysis && (
          <span className="stat-item analysis-indicator">📊 VLM Analysis</span>
        )}
        {currentExperiment.has_trajectory_analysis && (
          <span className="stat-item analysis-indicator">🎯 Trajectory Analysis</span>
        )}
      </div>

      <p
        id="base-prompt"
        className="base-prompt"
        title={currentExperiment.base_prompt}
      >
        {currentExperiment.base_prompt}
      </p>
    </div>
  );
};

export default ExperimentHeader;
