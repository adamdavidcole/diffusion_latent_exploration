import React, { useState, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import { formatDuration } from '../../utils/formatters';
import { formatAttentionBendingInfo, getAttentionBendingSummary } from '../../utils/attentionBendingUtils';

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

    const truncatedName = experiment.name && experiment.name.length > 30
        ? experiment.name.substring(0, 30) + '...'
        : (experiment.name || 'Unknown');

    const basePrompt = experiment.base_prompt || 'No prompt available';
    const truncatedPrompt = basePrompt.length > 100
        ? basePrompt.substring(0, 100) + '...'
        : basePrompt;

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
    
    // Get attention bending info using utility
    const attentionBendingInfo = formatAttentionBendingInfo(experiment, { abbreviated: false });
    const attentionBendingSummary = getAttentionBendingSummary(experiment);

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
                        {attentionBendingSummary && (
                            <span
                                className={`attention-bending-badge ${attentionBendingInfo?.phase || ''}`}
                                title="Attention Bending enabled"
                            >
                                {attentionBendingSummary}
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
                    {/* Attention bending details in expanded view */}
                    {!state.sidebarCollapsed && attentionBendingInfo && (
                        <div className="experiment-attention-details">
                            {attentionBendingInfo.details.map((detail, idx) => (
                                <div key={idx} className="attention-detail-item">
                                    {detail}
                                </div>
                            ))}
                        </div>
                    )}
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
                            <strong>{attentionBendingInfo.summary}:</strong>
                            {attentionBendingInfo.details.map((detail, idx) => (
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
