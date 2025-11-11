/**
 * Utility functions for formatting attention bending information
 */

// Helper to format parameter value intelligently
const formatParamValue = (key, value) => {
  if (value === null || value === undefined) return null;

  // Skip default values
  if (key === 'strength' && value === 1.0) return null;
  if (key === 'renormalize' && value === true) return null;
  if (key === 'preserve_sparsity' && value === false) return null;
  if (key === 'crop_rotated' && value === true) return null;  // Default is true, skip if true
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
  mode = mode && mode.toLowerCase();
  const paramsByMode = {
    'amplify': ['amplify_factor', 'strength', 'apply_to_timesteps', 'renormalize'],
    'scale': ['scale_factor', 'strength', 'apply_to_timesteps', 'renormalize'],
    'rotate': ['angle', 'strength', 'crop_rotated', 'apply_to_timesteps', 'renormalize'],
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

/**
 * Format attention bending information for display
 * @param {Object} experiment - The experiment object
 * @param {Object} options - Formatting options
 * @param {boolean} options.abbreviated - Use abbreviated format (for sidebar collapsed view)
 * @returns {Object|null} - Formatted attention bending info or null if not applicable
 */
export const formatAttentionBendingInfo = (experiment, options = {}) => {
  const { abbreviated = false } = options;
  const { attention_bending_settings } = experiment;

  console.log("attention_bending_settings", attention_bending_settings)

  if (!attention_bending_settings?.enabled) {
    return null;
  }

  const { configs, apply_to_output } = attention_bending_settings;

  if (!configs || configs.length === 0) {
    return null;
  }

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
    if (abbreviated) {
      // Abbreviated format: just token and mode
      return `"${cfg.token}" → ${cfg.mode}`;
    } else {
      // Full format: include parameters
      let details = `"${cfg.token}" → ${cfg.mode}`;
      if (params.length > 0) {
        details += ` (${params.join(', ')})`;
      }
      return details;
    }
  });

  const phase = apply_to_output ? 'Active' : 'Viz Only';

  return {
    summary: `🎨 Attention Bending [${phase}]`,
    details: configSummaries,
    phase: apply_to_output ? 'phase-2' : 'phase-1',
    configCount: configs.length
  };
};

/**
 * Get a compact summary of attention bending for sidebar
 * @param {Object} experiment - The experiment object
 * @returns {string|null} - Compact summary or null
 */
export const getAttentionBendingSummary = (experiment) => {
  const info = formatAttentionBendingInfo(experiment, { abbreviated: true });
  if (!info) return null;

  const { configCount, phase } = info;
  const phaseIcon = phase === 'phase-2' ? '✓' : '○';
  return `${phaseIcon} ${configCount} bend${configCount > 1 ? 's' : ''}`;
};
