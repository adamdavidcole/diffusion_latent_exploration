import React, { useState, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import { formatDuration } from '../../utils/formatters';

// Helper function to format CFG information for tooltip
const formatCfgInfo = (experiment) => {
    const { cfg_scale, cfg_schedule_settings } = experiment;
    
    // Check if CFG schedule is enabled and has a schedule
    if (cfg_schedule_settings?.enabled && cfg_schedule_settings?.schedule) {
        const schedule = cfg_schedule_settings.schedule;
        const interpolation = cfg_schedule_settings.interpolation || 'linear';
        
        // Format schedule as compact string
        const scheduleStr = Object.entries(schedule)
            .map(([step, value]) => `${step}:${value}`)
            .join(', ');
        
        return `CFG Schedule: {${scheduleStr}} (${interpolation})`;
    }
    
    // Fall back to basic CFG scale
    if (cfg_scale !== null && cfg_scale !== undefined) {
        return `CFG: ${cfg_scale}`;
    }
    
    return null;
};

// Helper function to format attention bending info
const getAttentionBendingInfo = (experiment) => {
    const { attention_bending_settings } = experiment;
    
    if (!attention_bending_settings?.enabled || !attention_bending_settings.configs?.length) {
        return null;
    }
    
    const { configs, apply_to_output } = attention_bending_settings;
    const phase = apply_to_output ? 'Active' : 'Viz';
    const icon = apply_to_output ? '🎨' : '👁️';
    
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
            return `${(value * 100).toFixed(0)}%`;
        }
        
        if (key === 'apply_to_timesteps' && Array.isArray(value) && value.length === 2) {
            return `steps ${value[0]}-${value[1]}`;
        }
        
        if (key === 'apply_to_layers' && Array.isArray(value)) {
            return `layers [${value.join(', ')}]`;
        }
        
        if (key === 'region' && Array.isArray(value) && value.length === 4) {
            return `[${value.map(v => v.toFixed(2)).join(', ')}]`;
        }
        
        // Numeric values
        if (typeof value === 'number') {
            if (key.includes('factor') || key.includes('scale')) {
                return `${value}x`;
            }
            if (key === 'angle') {
                return `${value}°`;
            }
            return `${value}`;
        }
        
        return `${value}`;
    };
    
    // Mode-specific parameter relevance (same as ExperimentHeader)
    const getModeRelevantParams = (mode) => {
        const paramsByMode = {
            'amplify': ['amplify_factor', 'strength', 'apply_to_timesteps'],
            'scale': ['scale_factor', 'strength', 'apply_to_timesteps'],
            'rotate': ['angle', 'strength', 'apply_to_timesteps'],
            'translate': ['translate_x', 'translate_y', 'strength', 'apply_to_timesteps'],
            'flip': ['flip_horizontal', 'flip_vertical', 'strength', 'apply_to_timesteps'],
            'blur': ['kernel_size', 'sigma', 'strength', 'apply_to_timesteps'],
            'sharpen': ['kernel_size', 'sharpen_amount', 'strength', 'apply_to_timesteps'],
            'regional_mask': ['region', 'region_feather', 'strength', 'apply_to_timesteps'],
            
            // Legacy support - in case old configs exist
            'spatial_scale': ['scale_factor', 'strength', 'apply_to_timesteps'],
        };
        return paramsByMode[mode] || ['strength', 'apply_to_timesteps'];
    };
    
    return {
        badge: `${icon} ${configs.length}`,
        tooltip: configs.map(cfg => {
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
            let detail = `"${cfg.token}" → ${cfg.mode}`;
            if (params.length > 0) {
                detail += ` (${params.join(', ')})`;
            }
            
            return detail;
        }),
        phase: apply_to_output ? 'active' : 'viz'
    };
};

const ExperimentItem = ({ experiment, isActive, onSelect }) => {
    const [showTooltip, setShowTooltip] = useState(false);
    const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
    const { state } = useApp();

    const handleMouseEnter = useCallback((e) => {
        if (state.sidebarCollapsed) {
            const rect = e.currentTarget.getBoundingClientRect();
            const tooltipWidth = 350; // max-width from CSS
            const tooltipHeight = 200; // estimated height

            let x = rect.right + 10;
            let y = rect.top + rect.height / 2;

            // Ensure tooltip doesn't go off right edge of screen
            if (x + tooltipWidth > window.innerWidth) {
                x = rect.left - tooltipWidth - 10;
            }

            // Ensure tooltip doesn't go off bottom edge of screen
            if (y + tooltipHeight / 2 > window.innerHeight) {
                y = window.innerHeight - tooltipHeight / 2 - 20;
            }

            // Ensure tooltip doesn't go off top edge of screen
            if (y - tooltipHeight / 2 < 20) {
                y = tooltipHeight / 2 + 20;
            }

            setTooltipPosition({ x, y });
            setShowTooltip(true);
        }
    }, [state.sidebarCollapsed]);

    const handleMouseLeave = useCallback(() => {
        setShowTooltip(false);
    }, []);

    const truncatedName = experiment.name.length > 30
        ? experiment.name.substring(0, 30) + '...'
        : experiment.name;

    const truncatedPrompt = experiment.base_prompt.length > 100
        ? experiment.base_prompt.substring(0, 100) + '...'
        : experiment.base_prompt;

    // Extract a shorter model name for display
    const getModelDisplayName = (modelId) => {
        if (!modelId || modelId === 'Unknown model') return 'Unknown';

        // Extract just the model name from full paths like "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        if (modelId.includes('/')) {
            const parts = modelId.split('/');
            const modelName = parts[parts.length - 1];
            // Further simplify by removing common suffixes
            return modelName
                .replace('-Diffusers', '')
                .replace('Wan2.1-T2V-', '')
                // .replace('WAN-', '')
                || modelId;
        }
        return modelId;
    };

    const modelDisplayName = getModelDisplayName(experiment.model_id);
    const attentionBendingInfo = getAttentionBendingInfo(experiment);

    // Helper to render prompt or prompt schedule
    const renderPromptText = () => {
        if (experiment.prompt_schedule_data?.schedule && experiment.prompt_schedule_data?.interpolation) {
            const schedule = experiment.prompt_schedule_data.schedule;
            const interpolation = experiment.prompt_schedule_data.interpolation;
            const keyframes = experiment.prompt_schedule_data.keyframes || 
                Object.keys(schedule).map(Number).sort((a, b) => a - b);
            
            // Create a compact summary for the main view
            const firstPrompt = schedule[keyframes[0]];
            const lastPrompt = schedule[keyframes[keyframes.length - 1]];
            
            return `Prompt Interpolation (${interpolation}): ${keyframes.length} keyframes`;
        }
        return experiment.base_prompt;
    };

    // Helper to render full prompt schedule in tooltip
    const renderTooltipPrompt = () => {
        if (experiment.prompt_schedule_data?.schedule && experiment.prompt_schedule_data?.interpolation) {
            const schedule = experiment.prompt_schedule_data.schedule;
            const interpolation = experiment.prompt_schedule_data.interpolation;
            const keyframes = experiment.prompt_schedule_data.keyframes || 
                Object.keys(schedule).map(Number).sort((a, b) => a - b);
            
            return (
                <>
                    <strong>Prompt Schedule ({interpolation.toUpperCase()}):</strong>
                    {keyframes.map((step, index) => (
                        <React.Fragment key={step}>
                            <br />• Step {step}: {schedule[step].length > 80 ? 
                                schedule[step].substring(0, 80) + '...' : 
                                schedule[step]}
                        </React.Fragment>
                    ))}
                </>
            );
        }
        return (
            <>
                <strong>Base Prompt:</strong>
                <br />{truncatedPrompt}
            </>
        );
    };

    return (
        <>
            <div
                className={`experiment-item ${isActive ? 'active' : ''}`}
                onClick={() => onSelect(experiment.name)}
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
            >
                <div className="experiment-item-header">
                    <div className="experiment-name">
                        {experiment.name}
                        {attentionBendingInfo && (
                            <span 
                                className={`attention-bending-badge ${attentionBendingInfo.phase}`}
                                title="Attention Bending enabled"
                            >
                                {attentionBendingInfo.badge}
                            </span>
                        )}
                    </div>
                    <div className="experiment-model">{modelDisplayName}</div>
                    <div className="experiment-meta">
                        <span>{experiment.videos_count} videos</span>
                        <span className="dot">•</span>
                        <span>{experiment.variations_count} variations</span>
                        <span>{experiment.seeds_count} seeds</span>
                        {experiment.duration_seconds && (
                            <span>{parseFloat(experiment.duration_seconds.toFixed(1))}s</span>
                        )}
                    </div>
                    <div className="experiment-prompt">{renderPromptText()}</div>
                </div>
            </div>

            {/* Tooltip */}
            {showTooltip && state.sidebarCollapsed && (
                <div
                    className="tooltip show"
                    style={{
                        left: tooltipPosition.x,
                        top: tooltipPosition.y,
                        transform: 'translateY(-50%)'
                    }}
                >
                    <strong>{truncatedName}</strong>
                    <br /><br />
                    <strong>Statistics:</strong>
                    <br />• {modelDisplayName} model
                    <br />• {experiment.videos_count} videos
                    <br />• {experiment.variations_count} variations
                    <br />• {experiment.seeds_count} seeds
                    {experiment.duration_seconds && (
                        <>
                            <br />• {formatDuration(experiment.duration_seconds)} duration
                        </>
                    )}
                    {formatCfgInfo(experiment) && (
                        <>
                            <br />• {formatCfgInfo(experiment)}
                        </>
                    )}
                    {attentionBendingInfo && (
                        <>
                            <br /><br />
                            <strong>Attention Bending ({attentionBendingInfo.phase === 'active' ? 'Active' : 'Visualization Only'}):</strong>
                            {attentionBendingInfo.tooltip.map((detail, idx) => (
                                <React.Fragment key={idx}>
                                    <br />• {detail}
                                </React.Fragment>
                            ))}
                        </>
                    )}
                    <br /><br />
                    {renderTooltipPrompt()}
                </div>
            )}
        </>
    );
};

export default React.memo(ExperimentItem);
