import React, { useState, useEffect } from 'react';
import MetricComparisonChart from './MetricComparisonChart';
import ScatterChart from './ScatterChart';
import LineChart from './LineChart';
import VarianceComparisonChart from './VarianceComparisonChart';
import BarChartWithLabels from './BarChartWithLabels';
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

    // Use the same data helper as TrajectoryAnalysis
    const helperData = extractChartData({ [activeNormalization]: analysisData }, activeNormalization);

    // Get individual values for detailed scatter plots using the data helper
    const getIndividualValuesData = () => {
        const individualData = {};

        // For scatter plots, return the scatter data directly from helper
        if (metricKey === 'velocity_vs_log_volume') {
            return helperData.velocityVsLogVolume || {};
        } else if (metricKey === 'velocity_vs_circuitousness') {
            return helperData.velocityVsCircuitousness || {};
        }

        // For regular metrics, we still need to extract individual values manually
        // since the helper only provides aggregated means, not individual trajectory values
        promptGroups.forEach(promptGroup => {
            let values = [];

            // Extract individual values based on metric type
            // Note: These are individual trajectory values, not available in the helper
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
            } else if (metricKey === 'semantic_convergence') {
                values = analysisData?.temporal_analysis?.[promptGroup]?.semantic_convergence?.individual_half_life || [];
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
            } else if (metricKey === 'efficiency_metrics') {
                values = analysisData?.individual_trajectory_geometry?.[promptGroup]?.efficiency_metrics?.individual_efficiency || [];
            } else if (metricKey === 'step_variability_stats') {
                values = analysisData?.individual_trajectory_geometry?.[promptGroup]?.step_variability_stats?.individual_values || [];
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

    // Get error bar data for intervals using consistent approach
    const getErrorBarData = () => {
        const errorData = {};

        // For scatter plots, no error bars
        if (metricKey.includes('velocity_vs')) {
            return {};
        }

        // For aggregated metrics that are in helper data, we could potentially
        // extract from helper, but since we need the full stats (std, min, max)
        // we still need to access analysisData directly for now
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
            } else if (metricKey === 'semantic_convergence') {
                statsData = analysisData?.temporal_analysis?.[promptGroup]?.semantic_convergence;
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
            } else if (metricKey === 'efficiency_metrics') {
                statsData = analysisData?.individual_trajectory_geometry?.[promptGroup]?.efficiency_metrics;
            } else if (metricKey === 'step_variability_stats') {
                statsData = analysisData?.individual_trajectory_geometry?.[promptGroup]?.step_variability_stats;
            }

            if (statsData) {
                // Use appropriate mean field based on metric type
                let meanValue = statsData.mean;
                if (metricKey === 'trajectory_length') {
                    meanValue = statsData.mean_length;
                } else if (metricKey === 'velocity_analysis') {
                    meanValue = statsData.overall_mean_velocity;
                } else if (metricKey === 'acceleration_analysis') {
                    meanValue = statsData.overall_mean_acceleration;
                } else if (metricKey === 'endpoint_distance') {
                    meanValue = statsData.mean_endpoint_distance;
                } else if (metricKey === 'tortuosity') {
                    meanValue = statsData.mean_tortuosity;
                } else if (metricKey === 'semantic_convergence') {
                    meanValue = statsData.mean_half_life;
                } else if (metricKey === 'efficiency_metrics') {
                    meanValue = statsData.mean_efficiency;
                }

                errorData[promptGroup] = {
                    mean: meanValue,
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
    const metricData = TrajectoryAnalysisDescriptions[metricKey] || {};
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
        // For scatter plots, use the helper data directly
        if (metricKey === 'velocity_vs_log_volume') {
            return helperData.velocityVsLogVolume || {};
        } else if (metricKey === 'velocity_vs_circuitousness') {
            return helperData.velocityVsCircuitousness || {};
        }

        // For variance comparison, return the special format
        if (metricKey === 'variance_comparison') {
            return {
                overall: helperData.overallVariance || {},
                acrossVideos: helperData.varianceAcrossVideos || {},
                acrossSteps: helperData.varianceAcrossSteps || {}
            };
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
            'step_variability_stats': 'stepVariability',
            // Spatial metrics
            'temporal_variance': 'temporalVariance',
            'spatial_variance': 'spatialVariance',
            'trajectory_pattern': 'trajectoryPattern',
            'evolution_ratio': 'evolutionRatio',
            'early_vs_late_significance': 'earlyVsLateSignificance',
            'trajectory_smoothness': 'trajectorySmooth',
            'phase_transition_strength': 'phaseTransitionStrength',
            'step_deltas_mean': 'stepDeltasMean',
            'step_deltas_std': 'stepDeltasStd',
            'progression_consistency': 'progressionConsistency',
            'progression_variability': 'progressionVariability',
            // Geometry derivatives
            'curvature_peak_mean': 'curvaturePeakMean',
            'jerk_peak_mean': 'jerkPeakMean'
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
                        ) : metricKey === 'variance_comparison' ? (
                            <VarianceComparisonChart
                                overallVarianceData={createEnhancedChartData().overall}
                                varianceAcrossVideosData={createEnhancedChartData().acrossVideos}
                                varianceAcrossStepsData={createEnhancedChartData().acrossSteps}
                                title={title}
                                size={800}
                                yLabel="Variance"
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        ) : metricKey === 'curvature_peak_mean' ? (
                            <BarChartWithLabels
                                data={createEnhancedChartData()}
                                labelData={helperData.curvaturePeakStepMean}
                                title={title}
                                size={800}
                                yLabel="Curvature"
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        ) : metricKey === 'jerk_peak_mean' ? (
                            <BarChartWithLabels
                                data={createEnhancedChartData()}
                                labelData={helperData.jerkPeakStepMean}
                                title={title}
                                size={800}
                                yLabel="Jerk"
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        ) : isArrayDataType(metricKey) ? (
                            <LineChart
                                data={createEnhancedChartData()}
                                title={title}
                                size={800}
                                xLabel="Diffusion Step"
                                yLabel={getYLabel(metricKey)}
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
                    {metricData && Object.keys(metricData).length > 0 && (
                        <div className="trajectory-chart-modal-description">
                            <h4>About This Metric</h4>
                            
                            {/* Short Description */}
                            {metricData.short_description && (
                                <div className="metric-short-description">
                                    <p><strong>Overview:</strong> {metricData.short_description}</p>
                                </div>
                            )}
                            
                            {/* Mathematical Formula */}
                            {metricData.formula && (
                                <div className="metric-formula">
                                    <h5>Mathematical Formula</h5>
                                    <div className="formula-display">
                                        <code>{metricData.formula}</code>
                                    </div>
                                </div>
                            )}
                            
                            {/* Implementation Code */}
                            {metricData.formula_code && (
                                <div className="metric-code">
                                    <h5>Implementation</h5>
                                    <pre className="code-block">
                                        <code>{metricData.formula_code}</code>
                                    </pre>
                                </div>
                            )}
                            
                            {/* Detailed Description */}
                            {metricData.description && (
                                <div className="metric-detailed-description">
                                    <h5>Interpretation</h5>
                                    <p>{metricData.description}</p>
                                </div>
                            )}
                            
                            {/* Source Code Link */}
                            {metricData.source_url && (
                                <div className="metric-source">
                                    <h5>Source Code</h5>
                                    <a 
                                        href={metricData.source_url} 
                                        target="_blank" 
                                        rel="noopener noreferrer"
                                        className="source-link"
                                    >
                                        View implementation in repository →
                                    </a>
                                </div>
                            )}
                            
                            {/* Fallback for old format */}
                            {!metricData.short_description && typeof metricData === 'string' && (
                                <div dangerouslySetInnerHTML={{ __html: metricData }} />
                            )}
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
        'step_variability_stats': 'Variability',
        // Spatial metrics
        'temporal_variance': 'Temporal Variance',
        'spatial_variance': 'Spatial Variance',
        'trajectory_pattern': 'Spatial Pattern Value',
        'evolution_ratio': 'Ratio',
        'early_vs_late_significance': 'Significance',
        'trajectory_smoothness': 'Smoothness',
        'phase_transition_strength': 'Transition Strength',
        'step_deltas_mean': 'Step Delta Mean',
        'step_deltas_std': 'Step Delta Std',
        'progression_consistency': 'Consistency',
        'progression_variability': 'Variability',
        // Geometry derivatives
        'curvature_peak_mean': 'Curvature',
        'jerk_peak_mean': 'Jerk'
    };

    return labels[metricKey] || 'Value';
};

// Helper function to determine if a metric uses array data (time series)
const isArrayDataType = (metricKey) => {
    const arrayMetrics = [
        'temporal_variance',
        'spatial_variance', 
        'trajectory_pattern',
        'step_deltas_mean',
        'step_deltas_std',
        'width_by_step',
        'branch_divergence'
    ];
    
    return arrayMetrics.includes(metricKey);
};

export default TrajectoryChartModal;
