import React, { useState, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import { formatDuration } from '../../utils/formatters';

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

    return (
        <>
            <div
                className={`experiment-item ${isActive ? 'active' : ''}`}
                onClick={() => onSelect(experiment.name)}
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
            >
                <div className="experiment-header">
                    <div className="experiment-name">{experiment.name}</div>
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
                    <div className="experiment-prompt">{experiment.base_prompt}</div>
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
                    <br /><br />
                    <strong>Base Prompt:</strong>
                    <br />{truncatedPrompt}
                </div>
            )}
        </>
    );
};

export default React.memo(ExperimentItem);
