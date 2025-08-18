import React, { useState } from 'react';
import { TrajectoryAnalysisDescriptions } from './TrajectoryAnalysisDescriptions';

const TrajectoryInfoTooltip = ({ metricKey, title }) => {
    const [showTooltip, setShowTooltip] = useState(false);
    const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });

    const description = TrajectoryAnalysisDescriptions[metricKey];

    if (!description) return null;

    const handleMouseEnter = (e) => {
        const rect = e.target.getBoundingClientRect();
        const tooltipWidth = 400; // Estimated tooltip width
        const tooltipHeight = 200; // Estimated tooltip height

        let left = rect.right + 10; // Default to right of icon
        let top = rect.top;

        // Adjust if tooltip would go off right edge
        if (left + tooltipWidth > window.innerWidth) {
            left = rect.left - tooltipWidth - 10;
        }

        // Adjust if tooltip would go off bottom edge
        if (top + tooltipHeight > window.innerHeight) {
            top = window.innerHeight - tooltipHeight - 10;
        }

        // Adjust if tooltip would go off top edge
        if (top < 10) {
            top = 10;
        }

        setTooltipPosition({ top, left });
        setShowTooltip(true);
    };

    const handleMouseLeave = () => {
        setShowTooltip(false);
    };

    return (
        <>
            <span
                className="info-icon"
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
                style={{
                    display: 'inline-block',
                    marginLeft: '8px',
                    width: '16px',
                    height: '16px',
                    borderRadius: '50%',
                    backgroundColor: '#4A90E2',
                    color: 'white',
                    fontSize: '10px',
                    lineHeight: '16px',
                    textAlign: 'center',
                    cursor: 'help',
                    fontWeight: 'bold',
                    top: "-2px",
                    position: "relative"
                }}
            >
                i
            </span>

            {showTooltip && (
                <div
                    style={{
                        position: 'fixed',
                        top: `${tooltipPosition.top}px`,
                        left: `${tooltipPosition.left}px`,
                        backgroundColor: 'rgba(0, 0, 0, 0.95)',
                        color: 'white',
                        padding: '16px',
                        borderRadius: '8px',
                        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
                        zIndex: 1000,
                        maxWidth: '400px',
                        fontSize: '14px',
                        lineHeight: '1.3',
                        border: '1px solid #4A90E2',
                        textTransform: 'none',
                        fontWeight: '300'
                    }}
                    dangerouslySetInnerHTML={{ __html: description }}
                />
            )}
        </>
    );
};

export default TrajectoryInfoTooltip;
