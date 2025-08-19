/**
 * Minimal data extraction utilities for TrajectoryAnalysis
 * This is a much smaller, safer approach that just reduces code duplication
 * without changing the overall architecture
 */

export const extractChartData = (trajectoryData, currentNormalization) => {
    if (!trajectoryData?.[currentNormalization]) return {};
    
    const data = trajectoryData[currentNormalization];
    
    const extractTemporalMetric = (metricKey, valueKey = 'mean') => {
        const result = {};
        if (data.temporal_analysis) {
            Object.keys(data.temporal_analysis).forEach(promptGroup => {
                const metric = data.temporal_analysis[promptGroup]?.[metricKey];
                if (metric) {
                    result[promptGroup] = metric[valueKey];
                }
            });
        }
        return result;
    };

    const extractGeometricMetric = (metricKey, valueKey = 'mean') => {
        const result = {};
        if (data.individual_trajectory_geometry) {
            Object.keys(data.individual_trajectory_geometry).forEach(promptGroup => {
                const metric = data.individual_trajectory_geometry[promptGroup]?.[metricKey];
                if (metric) {
                    result[promptGroup] = metric[valueKey];
                }
            });
        }
        return result;
    };

    const extractScatterData = (temporalMetric, geometricMetric) => {
        const result = {};
        if (data.temporal_analysis && data.individual_trajectory_geometry) {
            Object.keys(data.temporal_analysis).forEach(promptGroup => {
                const temporal = data.temporal_analysis[promptGroup]?.[temporalMetric];
                const geometric = data.individual_trajectory_geometry[promptGroup]?.[geometricMetric];
                
                if (temporal && geometric) {
                    const temporalValues = temporal.mean_velocity_by_video || 
                                          temporal.mean_velocity || [];
                    const geometricValues = geometric.individual_values || [];
                    
                    if (temporalValues.length > 0 && geometricValues.length > 0) {
                        const adjustedGeometricValues = geometricMetric === 'circuitousness_stats' ? 
                            geometricValues.map(val => val - 1.0) : geometricValues;
                        
                        result[promptGroup] = temporalValues.map((x, i) => ({
                            x: x,
                            y: adjustedGeometricValues[i]
                        })).filter(point => point.y !== undefined);
                    }
                }
            });
        }
        return result;
    };

    const extractIndividualValues = (metricKey) => {
        const result = {};
        Object.keys(data.temporal_analysis || {}).forEach(promptGroup => {
            let values = [];
            
            if (metricKey === 'trajectory_length') {
                values = data.temporal_analysis?.[promptGroup]?.trajectory_length?.individual_lengths || [];
            } else if (metricKey === 'velocity_analysis') {
                values = data.temporal_analysis?.[promptGroup]?.velocity_analysis?.mean_velocity_by_video ||
                        data.temporal_analysis?.[promptGroup]?.velocity_analysis?.mean_velocity || [];
            } else if (metricKey === 'acceleration_analysis') {
                values = data.temporal_analysis?.[promptGroup]?.acceleration_analysis?.mean_acceleration_by_video ||
                        data.temporal_analysis?.[promptGroup]?.acceleration_analysis?.mean_acceleration || [];
            } else if (metricKey === 'endpoint_distance') {
                values = data.temporal_analysis?.[promptGroup]?.endpoint_distance?.individual_distances || [];
            } else if (metricKey === 'tortuosity') {
                values = data.temporal_analysis?.[promptGroup]?.tortuosity?.individual_tortuosity || [];
            } else if (metricKey === 'semantic_convergence') {
                values = data.temporal_analysis?.[promptGroup]?.semantic_convergence?.individual_half_life || [];
            }

            if (values && values.length > 0) {
                result[promptGroup] = values.map((value, index) => ({
                    x: index + 1, // Trajectory number
                    y: value
                }));
            }
        });

        // Add geometric metrics
        Object.keys(data.individual_trajectory_geometry || {}).forEach(promptGroup => {
            let values = [];
            
            if (metricKey === 'log_volume_stats') {
                values = data.individual_trajectory_geometry?.[promptGroup]?.log_volume_stats?.individual_values || [];
            } else if (metricKey === 'effective_side_stats') {
                values = data.individual_trajectory_geometry?.[promptGroup]?.effective_side_stats?.individual_values || [];
            } else if (metricKey === 'endpoint_alignment_stats') {
                values = data.individual_trajectory_geometry?.[promptGroup]?.endpoint_alignment_stats?.individual_values || [];
            } else if (metricKey === 'turning_angle_stats') {
                values = data.individual_trajectory_geometry?.[promptGroup]?.turning_angle_stats?.individual_values || [];
            } else if (metricKey === 'circuitousness_stats') {
                values = data.individual_trajectory_geometry?.[promptGroup]?.circuitousness_stats?.individual_values || [];
            } else if (metricKey === 'efficiency_metrics') {
                values = data.individual_trajectory_geometry?.[promptGroup]?.efficiency_metrics?.individual_efficiency || [];
            } else if (metricKey === 'step_variability_stats') {
                values = data.individual_trajectory_geometry?.[promptGroup]?.step_variability_stats?.individual_values || [];
            }

            if (values && values.length > 0) {
                result[promptGroup] = values.map((value, index) => ({
                    x: index + 1, // Trajectory number
                    y: value
                }));
            }
        });

        return result;
    };

    const extractStatisticalData = (metricKey) => {
        const result = {};
        
        Object.keys(data.temporal_analysis || {}).forEach(promptGroup => {
            let statsData = null;
            
            if (metricKey === 'trajectory_length') {
                statsData = data.temporal_analysis?.[promptGroup]?.trajectory_length;
            } else if (metricKey === 'velocity_analysis') {
                statsData = data.temporal_analysis?.[promptGroup]?.velocity_analysis;
            } else if (metricKey === 'acceleration_analysis') {
                statsData = data.temporal_analysis?.[promptGroup]?.acceleration_analysis;
            } else if (metricKey === 'endpoint_distance') {
                statsData = data.temporal_analysis?.[promptGroup]?.endpoint_distance;
            } else if (metricKey === 'tortuosity') {
                statsData = data.temporal_analysis?.[promptGroup]?.tortuosity;
            } else if (metricKey === 'semantic_convergence') {
                statsData = data.temporal_analysis?.[promptGroup]?.semantic_convergence;
            }

            if (statsData) {
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
                }

                result[promptGroup] = {
                    mean: meanValue,
                    std: statsData.std,
                    min: statsData.min,
                    max: statsData.max,
                    median: statsData.median
                };
            }
        });

        // Add geometric metrics
        Object.keys(data.individual_trajectory_geometry || {}).forEach(promptGroup => {
            let statsData = null;
            
            if (metricKey === 'log_volume_stats') {
                statsData = data.individual_trajectory_geometry?.[promptGroup]?.log_volume_stats;
            } else if (metricKey === 'effective_side_stats') {
                statsData = data.individual_trajectory_geometry?.[promptGroup]?.effective_side_stats;
            } else if (metricKey === 'endpoint_alignment_stats') {
                statsData = data.individual_trajectory_geometry?.[promptGroup]?.endpoint_alignment_stats;
            } else if (metricKey === 'turning_angle_stats') {
                statsData = data.individual_trajectory_geometry?.[promptGroup]?.turning_angle_stats;
            } else if (metricKey === 'circuitousness_stats') {
                statsData = data.individual_trajectory_geometry?.[promptGroup]?.circuitousness_stats;
            } else if (metricKey === 'efficiency_metrics') {
                statsData = data.individual_trajectory_geometry?.[promptGroup]?.efficiency_metrics;
            } else if (metricKey === 'step_variability_stats') {
                statsData = data.individual_trajectory_geometry?.[promptGroup]?.step_variability_stats;
            }

            if (statsData) {
                let meanValue = statsData.mean;
                if (metricKey === 'efficiency_metrics') {
                    meanValue = statsData.mean_efficiency;
                }

                result[promptGroup] = {
                    mean: meanValue,
                    std: statsData.std,
                    min: statsData.min,
                    max: statsData.max,
                    median: statsData.median
                };
            }
        });

        return result;
    };

    return {
        // Temporal metrics
        trajectoryLength: extractTemporalMetric('trajectory_length', 'mean_length'),
        velocity: extractTemporalMetric('velocity_analysis', 'overall_mean_velocity'),
        acceleration: extractTemporalMetric('acceleration_analysis', 'overall_mean_acceleration'),
        endpointDistance: extractTemporalMetric('endpoint_distance', 'mean_endpoint_distance'),
        tortuosity: extractTemporalMetric('tortuosity', 'mean_tortuosity'),
        semanticConvergence: extractTemporalMetric('semantic_convergence', 'mean_half_life'),
        
        // Geometric metrics
        logVolume: extractGeometricMetric('log_volume_stats', 'mean'),
        effectiveSide: extractGeometricMetric('effective_side_stats', 'mean'),
        endpointAlignment: extractGeometricMetric('endpoint_alignment_stats', 'mean'),
        turningAngle: extractGeometricMetric('turning_angle_stats', 'mean'),
        circuitousness: extractGeometricMetric('circuitousness_stats', 'mean'),
        efficiency: extractGeometricMetric('efficiency_metrics', 'mean_efficiency'),
        stepVariability: extractGeometricMetric('step_variability_stats', 'mean'),
        
        // Scatter plots
        velocityVsLogVolume: extractScatterData('velocity_analysis', 'log_volume_stats'),
        velocityVsCircuitousness: extractScatterData('velocity_analysis', 'circuitousness_stats'),
        
        // Individual values (for modal detailed views)
        getIndividualValues: extractIndividualValues,
        
        // Statistical data (for modal summary views)
        getStatisticalData: extractStatisticalData
    };
};

export default { extractChartData };
