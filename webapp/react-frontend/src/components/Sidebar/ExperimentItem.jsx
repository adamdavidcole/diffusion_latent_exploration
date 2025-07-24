import React, { useState, useCallback } from 'react';
import { useApp } from '../../context/AppContext';

const ExperimentItem = ({ experiment, isActive, onSelect }) => {
    const [showTooltip, setShowTooltip] = useState(false);
    const { state } = useApp();

    const handleMouseEnter = useCallback(() => {
        if (state.sidebarCollapsed) {
            setShowTooltip(true);
        }
    }, [state.sidebarCollapsed]);

    const handleMouseLeave = useCallback(() => {
        setShowTooltip(false);
    }, []);

    const truncatedName = experiment.name.length > 40
        ? experiment.name.substring(0, 40) + '...'
        : experiment.name;

    const truncatedPrompt = experiment.base_prompt.length > 100
        ? experiment.base_prompt.substring(0, 100) + '...'
        : experiment.base_prompt;

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
                    <div className="experiment-meta">
                        <span>{experiment.videos_count} videos</span>
                        <span>{experiment.variations_count} variations</span>
                        <span>{experiment.seeds_count} seeds</span>
                    </div>
                    <div className="experiment-prompt">{experiment.base_prompt}</div>
                </div>
            </div>

            {/* Tooltip */}
            {showTooltip && state.sidebarCollapsed && (
                <div className="tooltip show">
                    <strong>{truncatedName}</strong>
                    <br /><br />
                    <strong>Statistics:</strong>
                    <br />• {experiment.videos_count} videos
                    <br />• {experiment.variations_count} variations
                    <br />• {experiment.seeds_count} seeds
                    <br /><br />
                    <strong>Base Prompt:</strong>
                    <br />{truncatedPrompt}
                </div>
            )}
        </>
    );
};

export default React.memo(ExperimentItem);
