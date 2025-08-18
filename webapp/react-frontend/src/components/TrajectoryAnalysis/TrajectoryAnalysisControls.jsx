import React from 'react';
import './TrajectoryAnalysisControls.css';

const TrajectoryAnalysisControls = ({
    availableNormalizations,
    activeNormalization,
    onNormalizationChange,
    chartSize,
    onChartSizeChange,
    beginAtZero,
    onBeginAtZeroChange,
    showFullVariationText,
    onShowFullVariationTextChange
}) => {

    const orderedNormalizationKeys = ["no_norm", "snr_norm_only", "full_norm"];


    const sortedAvailableNormalizations = availableNormalizations.sort((a, b) => {
        const indexA = orderedNormalizationKeys.indexOf(a.key);
        const indexB = orderedNormalizationKeys.indexOf(b.key);
        return indexA - indexB;
    });

    return (
        <div className="trajectory-analysis-controls">
            <div className="normalization-selector">
                <label>Normalization:</label>
                <div className="normalization-buttons">
                    {sortedAvailableNormalizations.map(norm => (
                        <button
                            key={norm.key}
                            className={`normalization-button ${activeNormalization === norm.key ? 'selected' : ''}`}
                            onClick={() => onNormalizationChange(norm.key)}
                        >
                            {norm.label}
                        </button>
                    ))}
                </div>
            </div>

            <div className="chart-size-control">
                <label>
                    Chart Size: {chartSize}px
                    <input
                        type="range"
                        min="200"
                        max="600"
                        value={chartSize}
                        onChange={(e) => onChartSizeChange(Number(e.target.value))}
                        className="size-slider"
                    />
                </label>
            </div>

            <div className="begin-at-zero-control">
                <label>
                    <input
                        type="checkbox"
                        checked={beginAtZero}
                        onChange={(e) => onBeginAtZeroChange(e.target.checked)}
                    />
                    Begin Y-axis at Zero
                </label>
            </div>

            <div className="full-variation-text-control">
                <label>
                    <input
                        type="checkbox"
                        checked={showFullVariationText}
                        onChange={(e) => onShowFullVariationTextChange(e.target.checked)}
                    />
                    Show Full Variation Text
                </label>
            </div>
        </div>
    );
};

export default TrajectoryAnalysisControls;
