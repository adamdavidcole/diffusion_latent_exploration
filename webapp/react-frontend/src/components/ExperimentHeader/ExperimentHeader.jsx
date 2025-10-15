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

// Helper function to format attention bending information
const formatAttentionBendingInfo = (experiment) => {
  const { attention_bending_settings } = experiment;
  
  if (!attention_bending_settings?.enabled) {
    return null;
  }
  
  const { configs, apply_to_output, num_configs } = attention_bending_settings;
  
  if (!configs || configs.length === 0) {
    return null;
  }
  
  // Helper to format parameter value intelligently
  const formatParamValue = (key, value) => {
    if (value === null || value === undefined) return null;
    
    // Skip default values
    if (key === 'strength' && value === 1.0) return null;
    if (key === 'renormalize' && value === true) return null;
    if (key === 'preserve_sparsity' && value === false) return null;
    if (key === 'angle' && value === 0) return null;
    if (key === 'translate_x' && value === 0) return null;
    if (key === 'translate_y' && value === 0) return null;
    
    // Format specific types
    if (typeof value === 'boolean') {
      return value ? key.replace(/_/g, ' ') : null;
    }
    
    if (key === 'strength') {
      return `${(value * 100).toFixed(0)}% strength`;
    }
    
    if (key === 'apply_to_timesteps' && Array.isArray(value) && value.length === 2) {
      return `steps ${value[0]}-${value[1]}`;
    }
    
    if (key === 'apply_to_layers' && Array.isArray(value)) {
      return `layers [${value.join(', ')}]`;
    }
    
    if (key === 'region' && Array.isArray(value) && value.length === 4) {
      return `region [${value.map(v => v.toFixed(2)).join(', ')}]`;
    }
    
    // Numeric values
    if (typeof value === 'number') {
      if (key.includes('factor') || key.includes('scale')) {
        return `${key.replace(/_/g, ' ')}: ${value}x`;
      }
      if (key === 'angle') {
        return `${value}°`;
      }
      return `${key.replace(/_/g, ' ')}: ${value}`;
    }
    
    return `${key.replace(/_/g, ' ')}: ${value}`;
  };
  
  // Mode-specific parameter relevance
  const getModeRelevantParams = (mode) => {
    const paramsByMode = {
      'amplify': ['amplify_factor', 'strength', 'apply_to_timesteps', 'renormalize'],
      'scale': ['scale_factor', 'strength', 'apply_to_timesteps', 'renormalize'],
      'rotate': ['angle', 'strength', 'apply_to_timesteps', 'renormalize'],
      'translate': ['translate_x', 'translate_y', 'strength', 'apply_to_timesteps', 'renormalize'],
      'flip': ['flip_horizontal', 'flip_vertical', 'strength', 'apply_to_timesteps', 'renormalize'],
      'blur': ['kernel_size', 'sigma', 'strength', 'apply_to_timesteps', 'renormalize'],
      'sharpen': ['kernel_size', 'sharpen_amount', 'strength', 'apply_to_timesteps', 'renormalize'],
      'regional_mask': ['region', 'region_feather', 'strength', 'apply_to_timesteps', 'renormalize'],
      
      // Legacy support - in case old configs exist
      'spatial_scale': ['scale_factor', 'strength', 'apply_to_timesteps', 'renormalize'],
    };
    return paramsByMode[mode] || ['strength', 'apply_to_timesteps', 'renormalize'];
  };
  
  // Format config details
  const configSummaries = configs.map(cfg => {
    const relevantParams = getModeRelevantParams(cfg.mode);
    
    // Build parameter list
    const params = [];
    relevantParams.forEach(paramKey => {
      const formatted = formatParamValue(paramKey, cfg[paramKey]);
      if (formatted) {
        params.push(formatted);
      }
    });
    
    // Build the detail string
    let details = `"${cfg.token}" → ${cfg.mode}`;
    if (params.length > 0) {
      details += ` (${params.join(', ')})`;
    }
    
    return details;
  });
  
  const phase = apply_to_output ? 'Active' : 'Viz Only';
  
  return {
    summary: `🎨 Attention Bending [${phase}]`,
    details: configSummaries,
    phase: apply_to_output ? 'phase-2' : 'phase-1'
  };
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
