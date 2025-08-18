import React, { useState, useEffect } from 'react';
import MetricComparisonChart from './MetricComparisonChart';
import ScatterChart from './ScatterChart';
import { TrajectoryAnalysisDescriptions } from '../TrajectoryAnalysis/TrajectoryAnalysisDescriptions';
import { getVariationTextFromPromptKey } from '../../utils/variationText';
import { getPromptGroupColors } from '../../utils/chartColors';
import './TrajectoryChartModal.css';

const TrajectoryChartModal = ({
    isOpen,
    onClose,
    chartData,
    metricKey,
    title,
    analysisData,
    promptGroups,
    currentExperiment,
    chartSize,
    beginAtZero,
    showFullVariationText,
    onShowFullVariationTextChange
}) => {
    if (!isOpen || !chartData || !metricKey) return null;

    const handleBackdropClick = (e) => {
        if (e.target === e.currentTarget) {
            onClose();
        }
    };

    // Get individual values for detailed scatter plots
    const getIndividualValuesData = () => {
        const individualData = {};

        promptGroups.forEach(promptGroup => {
            let values = [];

            // Extract individual values based on metric type
            if (metricKey === 'trajectory_length') {
                values = analysisData?.temporal_analysis?.[promptGroup]?.trajectory_length?.individual_lengths || [];
            } else if (metricKey === 'velocity_analysis') {
                values = analysisData?.temporal_analysis?.[promptGroup]?.velocity_analysis?.mean_velocity_by_video ||
                    analysisData?.temporal_analysis?.[promptGroup]?.velocity_analysis?.mean_velocity || [];
            } else if (metricKey === 'acceleration_analysis') {
                values = analysisData?.temporal_analysis?.[promptGroup]?.acceleration_analysis?.mean_acceleration_by_video ||
                    analysisData?.temporal_analysis?.[promptGroup]?.acceleration_analysis?.mean_acceleration || [];
            } else if (metricKey === 'endpoint_distance') {
                values = analysisData?.temporal_analysis?.[promptGroup]?.endpoint_distance?.individual_distances || [];
            } else if (metricKey === 'tortuosity') {
                values = analysisData?.temporal_analysis?.[promptGroup]?.tortuosity?.individual_tortuosity || [];
            } else if (metricKey === 'log_volume_stats') {
                values = analysisData?.individual_trajectory_geometry?.[promptGroup]?.log_volume_stats?.individual_values || [];
            } else if (metricKey === 'effective_side_stats') {
                values = analysisData?.individual_trajectory_geometry?.[promptGroup]?.effective_side_stats?.individual_values || [];
            } else if (metricKey === 'endpoint_alignment_stats') {
                values = analysisData?.individual_trajectory_geometry?.[promptGroup]?.endpoint_alignment_stats?.individual_values || [];
            } else if (metricKey === 'turning_angle_stats') {
                values = analysisData?.individual_trajectory_geometry?.[promptGroup]?.turning_angle_stats?.individual_values || [];
            } else if (metricKey === 'circuitousness_stats') {
                values = analysisData?.individual_trajectory_geometry?.[promptGroup]?.circuitousness_stats?.individual_values || [];
            }

            // Convert to scatter plot format with trajectory index as x-axis
            if (values && values.length > 0) {
                individualData[promptGroup] = values.map((value, index) => ({
                    x: index + 1, // Trajectory number
                    y: value
                }));
            }
        });

        return individualData;
    };

    // Get error bar data for intervals
    const getErrorBarData = () => {
        const errorData = {};

        promptGroups.forEach(promptGroup => {
            let statsData = null;

            // Extract stats based on metric type
            if (metricKey === 'trajectory_length') {
                statsData = analysisData?.temporal_analysis?.[promptGroup]?.trajectory_length;
            } else if (metricKey === 'velocity_analysis') {
                statsData = analysisData?.temporal_analysis?.[promptGroup]?.velocity_analysis;
            } else if (metricKey === 'acceleration_analysis') {
                statsData = analysisData?.temporal_analysis?.[promptGroup]?.acceleration_analysis;
            } else if (metricKey === 'endpoint_distance') {
                statsData = analysisData?.temporal_analysis?.[promptGroup]?.endpoint_distance;
            } else if (metricKey === 'tortuosity') {
                statsData = analysisData?.temporal_analysis?.[promptGroup]?.tortuosity;
            } else if (metricKey === 'log_volume_stats') {
                statsData = analysisData?.individual_trajectory_geometry?.[promptGroup]?.log_volume_stats;
            } else if (metricKey === 'effective_side_stats') {
                statsData = analysisData?.individual_trajectory_geometry?.[promptGroup]?.effective_side_stats;
            } else if (metricKey === 'endpoint_alignment_stats') {
                statsData = analysisData?.individual_trajectory_geometry?.[promptGroup]?.endpoint_alignment_stats;
            } else if (metricKey === 'turning_angle_stats') {
                statsData = analysisData?.individual_trajectory_geometry?.[promptGroup]?.turning_angle_stats;
            } else if (metricKey === 'circuitousness_stats') {
                statsData = analysisData?.individual_trajectory_geometry?.[promptGroup]?.circuitousness_stats;
            }

            if (statsData) {
                errorData[promptGroup] = {
                    mean: statsData.mean,
                    std: statsData.std,
                    min: statsData.min,
                    max: statsData.max,
                    median: statsData.median
                };
            }
        });

        return errorData;
    };

    const individualValues = getIndividualValuesData();
    const errorBarData = getErrorBarData();
    const description = TrajectoryAnalysisDescriptions[metricKey] || '';
    const hasIndividualValues = Object.keys(individualValues).some(key => individualValues[key].length > 0);

    // Get consistent colors for all charts
    const consistentColors = getPromptGroupColors(promptGroups);

    // Calculate global ranges for individual trajectory charts
    const calculateGlobalRanges = () => {
        let globalYMin = Infinity;
        let globalYMax = -Infinity;
        let globalXMax = 0; // X is trajectory number, so min is 1

        Object.values(individualValues).forEach(trajectoryData => {
            if (trajectoryData && trajectoryData.length > 0) {
                trajectoryData.forEach(point => {
                    if (point.y != null) {
                        globalYMin = Math.min(globalYMin, point.y);
                        globalYMax = Math.max(globalYMax, point.y);
                    }
                    if (point.x != null) {
                        globalXMax = Math.max(globalXMax, point.x);
                    }
                });
            }
        });

        // Add some padding to the ranges
        const yRange = globalYMax - globalYMin;
        const yPadding = yRange * 0.1; // 10% padding

        return {
            xMin: 0.5, // Start slightly before trajectory 1
            xMax: globalXMax + 0.5, // End slightly after last trajectory
            yMin: globalYMin - yPadding,
            yMax: globalYMax + yPadding
        };
    };

    const globalRanges = hasIndividualValues ? calculateGlobalRanges() : null;

    // Create enhanced chart data with error bars
    const createEnhancedChartData = () => {
        if (!hasIndividualValues) return chartData;

        const enhancedData = { ...chartData };
        const errorInfo = [];

        promptGroups.forEach(promptGroup => {
            const stats = errorBarData[promptGroup];
            if (stats && stats.std) {
                errorInfo.push({
                    group: promptGroup,
                    mean: stats.mean,
                    std: stats.std,
                    min: stats.min,
                    max: stats.max
                });
            }
        });

        return enhancedData;
    };

    // Create combined trajectory data for all-in-one visualization
    const createCombinedTrajectoryData = () => {
        const combinedData = {};

        promptGroups.forEach((promptGroup, index) => {
            const values = individualValues[promptGroup];
            if (values && values.length > 0) {
                const label = showFullVariationText && currentExperiment ?
                    getVariationTextFromPromptKey(promptGroup, currentExperiment) :
                    promptGroup.replace('prompt_', 'P');

                combinedData[label] = values;
            }
        });

        return combinedData;
    };

    const combinedTrajectoryData = hasIndividualValues ? createCombinedTrajectoryData() : null;

    return (
        <div className="trajectory-chart-modal-backdrop" onClick={handleBackdropClick}>
            <div className="trajectory-chart-modal">
                <div className="trajectory-chart-modal-header">
                    <h3>{title} - Detailed View</h3>
                    <div className="trajectory-chart-modal-controls">
                        <label>
                            <input
                                type="checkbox"
                                checked={showFullVariationText}
                                onChange={(e) => onShowFullVariationTextChange(e.target.checked)}
                            />
                            Show Full Variation Text
                        </label>
                        <button
                            className="trajectory-chart-modal-close"
                            onClick={onClose}
                            aria-label="Close modal"
                        >
                            ×
                        </button>
                    </div>
                </div>

                <div className="trajectory-chart-modal-content">
                    {/* Main Chart */}
                    <div className="trajectory-chart-modal-main-chart">
                        {metricKey.includes('velocity_vs') ? (
                            <ScatterChart
                                data={chartData}
                                title={title}
                                size={800}
                                xLabel={metricKey === 'velocity_vs_log_volume' ? 'Velocity (mean per trajectory)' : 'Velocity (mean per trajectory)'}
                                yLabel={metricKey === 'velocity_vs_log_volume' ? 'Log Volume' : 'Circuitousness − 1.0'}
                                colors={consistentColors}
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        ) : (
                            <MetricComparisonChart
                                data={createEnhancedChartData()}
                                title={title}
                                size={800}
                                yLabel={getYLabel(metricKey)}
                                colors={consistentColors}
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        )}
                    </div>

                    {/* Description */}
                    {description && (
                        <div className="trajectory-chart-modal-description">
                            <h4>About This Metric</h4>
                            <div dangerouslySetInnerHTML={{ __html: description }} />
                        </div>
                    )}

                    {/* Statistics Summary */}
                    {Object.keys(errorBarData).length > 0 && (
                        <div className="trajectory-chart-modal-stats">
                            <h4>Statistical Summary</h4>
                            <div className="stats-grid">
                                {promptGroups.map(promptGroup => {
                                    const stats = errorBarData[promptGroup];
                                    if (!stats) return null;

                                    const label = showFullVariationText && currentExperiment ?
                                        getVariationTextFromPromptKey(promptGroup, currentExperiment) :
                                        promptGroup.replace('prompt_', 'P');

                                    return (
                                        <div key={promptGroup} className="stats-item">
                                            <strong>{label}</strong>
                                            <div className="stats-values">
                                                <span>Mean: {stats.mean?.toFixed(3)}</span>
                                                <span>Std: {stats.std?.toFixed(3)}</span>
                                                <span>Min: {stats.min?.toFixed(3)}</span>
                                                <span>Max: {stats.max?.toFixed(3)}</span>
                                                {stats.median && <span>Median: {stats.median?.toFixed(3)}</span>}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}



                    {/* Individual Values Scatter Plots */}
                    {hasIndividualValues && (
                        <div className="trajectory-chart-modal-individual-charts">
                            <h4>Individual Trajectory Values</h4>
                            <div className="individual-charts-grid">
                                {promptGroups.map((promptGroup, index) => {
                                    const values = individualValues[promptGroup];
                                    if (!values || values.length === 0) return null;

                                    const label = showFullVariationText && currentExperiment ?
                                        getVariationTextFromPromptKey(promptGroup, currentExperiment) :
                                        promptGroup.replace('prompt_', 'P');

                                    // Use consistent color for this prompt group
                                    const promptColor = consistentColors[index];

                                    return (
                                        <div key={promptGroup} className="individual-chart-container">
                                            <ScatterChart
                                                data={{ [promptGroup]: values }}
                                                title={`${label} - Individual Values`}
                                                size={350}
                                                xLabel="Trajectory #"
                                                yLabel={getYLabel(metricKey)}
                                                colors={[promptColor]}
                                                currentExperiment={currentExperiment}
                                                beginAtZero={beginAtZero}
                                                showFullVariationText={showFullVariationText}
                                                xMin={globalRanges?.xMin}
                                                xMax={globalRanges?.xMax}
                                                yMin={globalRanges?.yMin}
                                                yMax={globalRanges?.yMax}
                                            />
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {/* Combined Trajectory Plot */}
                    {hasIndividualValues && combinedTrajectoryData && Object.keys(combinedTrajectoryData).length > 1 && (
                        <div className="trajectory-chart-modal-combined-chart">
                            <h4>Combined Trajectory Comparison</h4>
                            <p className="combined-chart-description">
                                All individual trajectory values displayed together with consistent colors for direct comparison across prompt groups.
                            </p>
                            <div className="combined-chart-container">
                                <ScatterChart
                                    data={combinedTrajectoryData}
                                    title={`${title} - All Trajectories Combined`}
                                    size={800}
                                    xLabel="Trajectory #"
                                    yLabel={getYLabel(metricKey)}
                                    colors={consistentColors}
                                    currentExperiment={currentExperiment}
                                    beginAtZero={beginAtZero}
                                    showFullVariationText={showFullVariationText}
                                    xMin={globalRanges?.xMin}
                                    xMax={globalRanges?.xMax}
                                    yMin={globalRanges?.yMin}
                                    yMax={globalRanges?.yMax}
                                />
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

// Helper function to get appropriate Y-axis labels
const getYLabel = (metricKey) => {
    const labels = {
        'trajectory_length': 'Length Units',
        'velocity_analysis': 'Velocity',
        'acceleration_analysis': 'Acceleration',
        'tortuosity': 'Tortuosity',
        'endpoint_distance': 'Distance',
        'semantic_convergence': 'Steps',
        'speed_stats': 'Speed Units',
        'log_volume_stats': 'Log Volume',
        'effective_side_stats': 'Effective Side',
        'endpoint_alignment_stats': 'Alignment',
        'turning_angle_stats': 'Angle (radians)',
        'circuitousness_stats': 'Circuitousness',
        'efficiency_metrics': 'Efficiency',
        'step_variability_stats': 'Variability'
    };

    return labels[metricKey] || 'Value';
};

export default TrajectoryChartModal;
