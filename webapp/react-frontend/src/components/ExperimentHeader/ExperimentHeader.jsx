import React from 'react';
import { useApp } from '../../context/AppContext';
import { formatAttentionBendingInfo } from '../../utils/attentionBendingUtils';
import './ExperimentHeader.css';

// Helper function to format duration
const formatDuration = (seconds) => {
  if (!seconds) return '';
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds.toFixed(1)}s`;
};

// Helper function to format CFG information
const formatCfgInfo = (experiment) => {
  const { cfg_scale, cfg_schedule_settings } = experiment;

  // Check if CFG schedule is enabled and has a schedule
  if (cfg_schedule_settings?.enabled && cfg_schedule_settings?.schedule) {
    const schedule = cfg_schedule_settings.schedule;
    const interpolation = cfg_schedule_settings.interpolation || 'linear';

    // Format schedule as compact string
    const scheduleStr = Object.entries(schedule)
      .map(([step, value]) => `${step}: ${value}`)
      .join(', ');

    return `CFG Schedule: {${scheduleStr}} (${interpolation})`;
  }

  // Fall back to basic CFG scale
  if (cfg_scale !== null && cfg_scale !== undefined) {
    return `CFG: ${cfg_scale}`;
  }

  return null;
};

// Helper function to render prompt schedule or static prompt
const renderPromptDisplay = (experiment) => {
  const { prompt_schedule_data, base_prompt } = experiment;

  // Check if prompt schedule is available
  if (prompt_schedule_data?.schedule && prompt_schedule_data?.interpolation) {
    const schedule = prompt_schedule_data.schedule;
    const interpolation = prompt_schedule_data.interpolation;
    const keyframes = prompt_schedule_data.keyframes || Object.keys(schedule).map(Number).sort((a, b) => a - b);

    return (
      <div className="prompt-schedule-display">
        <div className="prompt-schedule-header">
          <strong>Prompt Interpolation ({interpolation.toUpperCase()}):</strong>
        </div>
        {keyframes.map((step, index) => (
          <div key={step} className="prompt-keyframe">
            <span className="keyframe-step">Step {step}:</span>{' '}
            <span className="keyframe-prompt">{schedule[step]}</span>
            {index < keyframes.length - 1 && (
              <span className="keyframe-arrow"> → </span>
            )}
          </div>
        ))}
      </div>
    );
  }

  // Fallback to static prompt
  return (
    <p id="base-prompt" className="base-prompt" title={base_prompt}>
      {base_prompt}
    </p>
  );
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
          {formatCfgInfo(currentExperiment) && (
            <span className="stat-item cfg-info">{formatCfgInfo(currentExperiment)}</span>
          )}
          {currentExperiment.has_vlm_analysis && (
            <span className="stat-item analysis-indicator">📊 VLM Analysis</span>
          )}
          {currentExperiment.has_trajectory_analysis && (
            <span className="stat-item analysis-indicator">🎯 Trajectory Analysis</span>
          )}
          {formatAttentionBendingInfo(currentExperiment) && (
            <span
              className={`stat-item attention-bending-indicator ${formatAttentionBendingInfo(currentExperiment).phase}`}
              title={formatAttentionBendingInfo(currentExperiment).details.join(' • ')}
            >
              {formatAttentionBendingInfo(currentExperiment).summary}
            </span>
          )}
        </div>
      </div>

      {/* Attention Bending Details Section */}
      {formatAttentionBendingInfo(currentExperiment) && (
        <div className="attention-bending-details">
          {formatAttentionBendingInfo(currentExperiment).details.map((detail, idx) => (
            <span key={idx} className="bending-config-detail">
              {detail}
            </span>
          ))}
        </div>
      )}

      {renderPromptDisplay(currentExperiment)}
    </div>
  );
};

export default ExperimentHeader;
