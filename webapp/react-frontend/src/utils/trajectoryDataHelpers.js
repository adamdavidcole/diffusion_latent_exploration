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

    const extractStructuralMetric = (metricKey, valueKey = 'mean') => {
        const result = {};
        if (data.structural_analysis) {
            Object.keys(data.structural_analysis).forEach(promptGroup => {
                const metric = data.structural_analysis[promptGroup]?.[metricKey];
                if (metric) {
                    result[promptGroup] = metric[valueKey];
                }
            });
        }
        return result;
    };

    const extractSpatialEvolutionMetric = (metricKey, valueKey = 'mean') => {
        const result = {};
        if (data.spatial_patterns?.trajectory_spatial_evolution) {
            Object.keys(data.spatial_patterns.trajectory_spatial_evolution).forEach(promptGroup => {
                const metric = data.spatial_patterns.trajectory_spatial_evolution[promptGroup]?.[metricKey];
                if (metric) {
                    result[promptGroup] = metric[valueKey] !== undefined ? metric[valueKey] : metric;
                }
            });
        }
        return result;
    };

    const extractSpatialProgressionMetric = (metricKey, valueKey = 'mean') => {
        const result = {};
        if (data.spatial_patterns?.spatial_progression_patterns) {
            Object.keys(data.spatial_patterns.spatial_progression_patterns).forEach(promptGroup => {
                const metric = data.spatial_patterns.spatial_progression_patterns[promptGroup]?.[metricKey];
                if (metric) {
                    result[promptGroup] = metric[valueKey] !== undefined ? metric[valueKey] : metric;
                }
            });
        }
        return result;
    };

    const extractGeometryDerivativeMetric = (metricKey) => {
        const result = {};
        if (data.geometry_derivatives) {
            Object.keys(data.geometry_derivatives).forEach(promptGroup => {
                const metric = data.geometry_derivatives[promptGroup]?.[metricKey];
                if (metric !== undefined) {
                    result[promptGroup] = metric;
                }
            });
        }
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

        // Structural Analysis - variance arrays
        temporalVariance: extractStructuralMetric('latent_space_variance', 'temporal_variance'),
        spatialVariance: extractStructuralMetric('latent_space_variance', 'spatial_variance'),
        
        // Structural Analysis - individual variance values
        overallVariance: extractStructuralMetric('latent_space_variance', 'overall_variance'),
        varianceAcrossVideos: extractStructuralMetric('latent_space_variance', 'variance_across_videos'),
        varianceAcrossSteps: extractStructuralMetric('latent_space_variance', 'variance_across_steps'),

        // Spatial Evolution Patterns - arrays
        trajectoryPattern: extractSpatialEvolutionMetric('trajectory_pattern'),
        
        // Spatial Evolution Patterns - individual values
        evolutionRatio: extractSpatialEvolutionMetric('evolution_ratio'),
        earlyVsLateSignificance: extractSpatialEvolutionMetric('early_vs_late_significance'),
        trajectorySmooth: extractSpatialEvolutionMetric('trajectory_smoothness'),
        phaseTransitionStrength: extractSpatialEvolutionMetric('phase_transition_strength'),

        // Spatial Progression Patterns - arrays
        stepDeltasMean: extractSpatialProgressionMetric('step_deltas_mean'),
        stepDeltasStd: extractSpatialProgressionMetric('step_deltas_std'),
        
        // Spatial Progression Patterns - individual values
        progressionConsistency: extractSpatialProgressionMetric('progression_consistency'),
        progressionVariability: extractSpatialProgressionMetric('progression_variability'),

        // Geometry Derivatives
        curvaturePeakMean: extractGeometryDerivativeMetric('curvature_peak_mean'),
        curvaturePeakStepMean: extractGeometryDerivativeMetric('curvature_peak_step_mean'),
        jerkPeakMean: extractGeometryDerivativeMetric('jerk_peak_mean'),
        jerkPeakStepMean: extractGeometryDerivativeMetric('jerk_peak_step_mean')
    };
};

export default { extractChartData };
