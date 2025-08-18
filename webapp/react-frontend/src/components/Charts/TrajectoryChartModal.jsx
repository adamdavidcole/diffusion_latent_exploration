import React, { useState, useEffect } from 'react';
import MetricComparisonChart from './MetricComparisonChart';
import ScatterChart from './ScatterChart';
import { TrajectoryAnalysisDescriptions } from '../TrajectoryAnalysis/TrajectoryAnalysisDescriptions';
import { getVariationTextFromPromptKey } from '../../utils/variationText';
import { getPromptGroupColors } from '../../utils/chartColors';
import { extractChartData } from '../../utils/trajectoryDataHelpers';
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
    onShowFullVariationTextChange,
    activeNormalization
}) => {
    if (!isOpen || !chartData || !metricKey) return null;

    const handleBackdropClick = (e) => {
        if (e.target === e.currentTarget) {
            onClose();
        }
    };

    // Use the same data helper as TrajectoryAnalysis for consistent data access
    const helperData = extractChartData({ [activeNormalization]: analysisData }, activeNormalization);

    // Get individual values for detailed scatter plots using the data helper
    const getIndividualValuesData = () => {
        // For scatter plots, return the scatter data directly from helper
        if (metricKey === 'velocity_vs_log_volume') {
            return helperData.velocityVsLogVolume || {};
        } else if (metricKey === 'velocity_vs_circuitousness') {
            return helperData.velocityVsCircuitousness || {};
        }

        // For regular metrics, use the helper function
        return helperData.getIndividualValues(metricKey);
    };

    // Get error bar data for intervals using consistent approach
    const getErrorBarData = () => {
        // For scatter plots, no error bars
        if (metricKey.includes('velocity_vs')) {
            return {};
        }

        // Use the helper function for consistent statistical data extraction
        return helperData.getStatisticalData(metricKey);
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

    // Create enhanced chart data using helper data consistently
    const createEnhancedChartData = () => {
        // For scatter plots, use the helper data directly
        if (metricKey === 'velocity_vs_log_volume') {
            return helperData.velocityVsLogVolume || {};
        } else if (metricKey === 'velocity_vs_circuitousness') {
            return helperData.velocityVsCircuitousness || {};
        }

        // For regular metrics, prefer helper data if available, otherwise fall back to chartData
        const helperMetricMap = {
            'trajectory_length': 'trajectoryLength',
            'velocity_analysis': 'velocity',
            'acceleration_analysis': 'acceleration',
            'endpoint_distance': 'endpointDistance',
            'tortuosity': 'tortuosity',
            'semantic_convergence': 'semanticConvergence',
            'log_volume_stats': 'logVolume',
            'effective_side_stats': 'effectiveSide',
            'endpoint_alignment_stats': 'endpointAlignment',
            'turning_angle_stats': 'turningAngle',
            'circuitousness_stats': 'circuitousness',
            'efficiency_metrics': 'efficiency',
            'step_variability_stats': 'stepVariability'
        };

        const helperKey = helperMetricMap[metricKey];
        if (helperKey && helperData[helperKey]) {
            return helperData[helperKey];
        }

        // Fallback to original chartData if helper doesn't have the metric
        return chartData || {};
    };

    // Create combined trajectory data for all-in-one visualization
    const createCombinedTrajectoryData = () => {
        const combinedData = {};

        // For scatter plots, we can't combine them into trajectory-index format
        if (metricKey.includes('velocity_vs')) {
            return null; // No combined view for scatter plots
        }

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
                            <h4>{metricKey.includes('velocity_vs') ? 'Individual Scatter Plot Data' : 'Individual Trajectory Values'}</h4>
                            <div className="individual-charts-grid">
                                {promptGroups.map((promptGroup, index) => {
                                    const values = individualValues[promptGroup];
                                    if (!values || values.length === 0) return null;

                                    const label = showFullVariationText && currentExperiment ?
                                        getVariationTextFromPromptKey(promptGroup, currentExperiment) :
                                        promptGroup.replace('prompt_', 'P');

                                    // Use consistent color for this prompt group
                                    const promptColor = consistentColors[index];

                                    // For scatter plots, use appropriate labels
                                    const xLabel = metricKey.includes('velocity_vs') ? 
                                        'Velocity (mean per trajectory)' : 'Trajectory #';
                                    const yLabel = metricKey === 'velocity_vs_log_volume' ? 'Log Volume' :
                                        metricKey === 'velocity_vs_circuitousness' ? 'Circuitousness − 1.0' :
                                        getYLabel(metricKey);

                                    return (
                                        <div key={promptGroup} className="individual-chart-container">
                                            <ScatterChart
                                                data={{ [promptGroup]: values }}
                                                title={`${label} - Individual Values`}
                                                size={350}
                                                xLabel={xLabel}
                                                yLabel={yLabel}
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
