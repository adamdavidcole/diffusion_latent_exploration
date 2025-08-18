import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import TrajectoryAnalysisControls from './TrajectoryAnalysisControls';
import MetricComparisonChart from '../Charts/MetricComparisonChart';
import LineChart from '../Charts/LineChart';
import { getVariationTextFromPromptKey } from '../../utils/variationText';
import './TrajectoryAnalysis.css';

const CHART_DEFAULT_SIZE = 550;

const TrajectoryAnalysis = ({ experimentPath }) => {
    const { state } = useApp();
    const { currentExperiment } = state;

    // Local state for trajectory analysis
    const [trajectoryData, setTrajectoryData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [activeSection, setActiveSection] = useState('temporal');
    const [activeNormalization, setActiveNormalization] = useState('no_norm');
    const [chartSize, setChartSize] = useState(CHART_DEFAULT_SIZE);
    const [beginAtZero, setBeginAtZero] = useState(false);
    const [showFullVariationText, setShowFullVariationText] = useState(false);
    const [showGlobalLegend, setShowGlobalLegend] = useState(true);

    // Available sections
    const sections = [
        { key: 'temporal', label: 'Temporal' },
        { key: 'geometric', label: 'Geometric' },
        { key: 'spatial', label: 'Spatial' },
        { key: 'channel', label: 'Channel' },
        { key: 'other', label: 'Other' }
    ];

    // Load trajectory analysis data
    useEffect(() => {
        const loadTrajectoryAnalysis = async () => {
            if (!experimentPath || !currentExperiment?.has_trajectory_analysis) return;

            try {
                setLoading(true);
                setError(null);
                const data = await api.getExperimentTrajectoryAnalysis(experimentPath);
                setTrajectoryData(data);

                // Set default normalization to the first available one
                if (data?.trajectory_analysis && Object.keys(data.trajectory_analysis).length > 0) {
                    const availableNorms = Object.keys(data.trajectory_analysis);
                    if (availableNorms.includes('no_norm')) {
                        setActiveNormalization('no_norm');
                    } else if (availableNorms.includes('snr_norm_only')) {
                        setActiveNormalization('snr_norm_only');
                    } else if (availableNorms.includes('full_norm')) {
                        setActiveNormalization('full_norm');
                    } else {
                        setActiveNormalization(availableNorms[0]);
                    }
                }
            } catch (err) {
                console.error('Error loading trajectory analysis:', err);
                setError(`Failed to fetch latent trajectory analysis data: ${err.message}`);
            } finally {
                setLoading(false);
            }
        };

        loadTrajectoryAnalysis();
    }, [experimentPath, currentExperiment]);

    // Create global label mappings for legend
    const createGlobalLabelMappings = (promptGroups) => {
        const mappings = {};
        promptGroups.forEach(key => {
            const abbreviatedLabel = key.replace('prompt_', 'P');
            const variationText = currentExperiment ?
                getVariationTextFromPromptKey(key, currentExperiment) :
                abbreviatedLabel;
            mappings[abbreviatedLabel] = variationText;
        });
        return mappings;
    };

    // Get current analysis data based on selected normalization
    const getCurrentAnalysisData = () => {
        if (!trajectoryData?.trajectory_analysis?.[activeNormalization]?.data) {
            return null;
        }
        return trajectoryData.trajectory_analysis[activeNormalization].data;
    };

    // Get available normalization options
    const getAvailableNormalizations = () => {
        if (!trajectoryData?.trajectory_analysis) return [];
        return Object.keys(trajectoryData.trajectory_analysis).map(key => ({
            key,
            label: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
        }));
    };

    // Extract prompt groups from the data
    const getPromptGroups = (analysisData) => {
        if (!analysisData) return [];

        // Look for prompt groups in various sections
        const sections = ['temporal_analysis', 'individual_trajectory_geometry', 'structural_analysis', 'normative_strength'];
        for (const section of sections) {
            if (analysisData[section]) {
                return Object.keys(analysisData[section]).filter(key => key.startsWith('prompt_'));
            }
        }
        return [];
    };

    // Render data for each section
    const renderSectionData = (analysisData) => {
        const promptGroups = getPromptGroups(analysisData);

        switch (activeSection) {
            case 'temporal':
                return renderTemporalAnalysis(analysisData, promptGroups);

            case 'geometric':
                return (
                    <div className="section-data">
                        <h4>Geometric Analysis - Log Volume Stats</h4>
                        <div className="data-grid">
                            {promptGroups.map(promptGroup => {
                                const logVolumeMean = analysisData?.individual_trajectory_geometry?.[promptGroup]?.log_volume_stats?.mean;
                                return (
                                    <div key={promptGroup} className="data-item">
                                        <span className="prompt-label">{promptGroup}:</span>
                                        <span className="data-value">{logVolumeMean?.toFixed(4) || 'N/A'}</span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                );

            case 'spatial':
                return (
                    <div className="section-data">
                        <h4>Spatial Analysis - Overall Variance</h4>
                        <div className="data-grid">
                            {promptGroups.map(promptGroup => {
                                const overallVariance = analysisData?.structural_analysis?.[promptGroup]?.latent_space_variance?.overall_variance;
                                return (
                                    <div key={promptGroup} className="data-item">
                                        <span className="prompt-label">{promptGroup}:</span>
                                        <span className="data-value">{overallVariance?.toFixed(4) || 'N/A'}</span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                );

            case 'channel':
                return (
                    <div className="section-data">
                        <h4>Channel Analysis</h4>
                        <p>TO DO</p>
                    </div>
                );

            case 'other':
                return (
                    <div className="section-data">
                        <h4>Other Analysis - Dominance Index</h4>
                        <div className="data-grid">
                            {promptGroups.map(promptGroup => {
                                const dominanceIndex = analysisData?.normative_strength?.[promptGroup]?.dominance_index;
                                return (
                                    <div key={promptGroup} className="data-item">
                                        <span className="prompt-label">{promptGroup}:</span>
                                        <span className="data-value">{dominanceIndex?.toFixed(4) || 'N/A'}</span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                );

            default:
                return <div>Unknown section</div>;
        }
    };

    // Render temporal analysis with charts
    const renderTemporalAnalysis = (analysisData, promptGroups) => {
        // Extract temporal metrics data
        const trajectoryLengthData = {};
        const velocityData = {};
        const accelerationData = {};
        const tortuosityData = {};
        const endpointDistanceData = {};
        const semanticConvergenceData = {};

        promptGroups.forEach(promptGroup => {
            const temporal = analysisData?.temporal_analysis?.[promptGroup];
            if (temporal) {
                trajectoryLengthData[promptGroup] = temporal.trajectory_length?.mean_length;
                velocityData[promptGroup] = temporal.velocity_analysis?.overall_mean_velocity;
                accelerationData[promptGroup] = temporal.acceleration_analysis?.overall_mean_acceleration;
                tortuosityData[promptGroup] = temporal.tortuosity?.mean_tortuosity;
                endpointDistanceData[promptGroup] = temporal.endpoint_distance?.mean_endpoint_distance;
                semanticConvergenceData[promptGroup] = temporal.semantic_convergence?.mean_half_life;
            }
        });

        return (
            <div className="section-data">
                <h4>Temporal Analysis</h4>

                <div className="metrics-grid" style={{ '--chart-size': `${chartSize}px` }}>
                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>Trajectory Length</h5>
                        <MetricComparisonChart
                            data={trajectoryLengthData}
                            title="Mean Length"
                            size={chartSize}
                            yLabel="Length Units"
                        />
                    </div>

                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>Velocity Analysis</h5>
                        <MetricComparisonChart
                            data={velocityData}
                            title="Overall Mean Velocity"
                            size={chartSize}
                            yLabel="Velocity"
                        />
                    </div>

                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>Acceleration Analysis</h5>
                        <MetricComparisonChart
                            data={accelerationData}
                            title="Overall Mean Acceleration"
                            size={chartSize}
                            yLabel="Acceleration"
                            currentExperiment={currentExperiment}
                            beginAtZero={beginAtZero}
                            showFullVariationText={showFullVariationText}
                        />
                    </div>

                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>Tortuosity</h5>
                        <MetricComparisonChart
                            data={tortuosityData}
                            title="Mean Tortuosity"
                            size={chartSize}
                            yLabel="Tortuosity"
                            currentExperiment={currentExperiment}
                            beginAtZero={beginAtZero}
                            showFullVariationText={showFullVariationText}
                        />
                    </div>

                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>Endpoint Distance</h5>
                        <MetricComparisonChart
                            data={endpointDistanceData}
                            title="Mean Endpoint Distance"
                            size={chartSize}
                            yLabel="Distance"
                            currentExperiment={currentExperiment}
                            beginAtZero={beginAtZero}
                            showFullVariationText={showFullVariationText}
                        />
                    </div>

                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>Semantic Convergence</h5>
                        <MetricComparisonChart
                            data={semanticConvergenceData}
                            title="Mean Half Life"
                            size={chartSize}
                            yLabel="Steps"
                            currentExperiment={currentExperiment}
                            beginAtZero={beginAtZero}
                            showFullVariationText={showFullVariationText}
                        />
                    </div>
                </div>
            </div>
        );
    };

    if (!currentExperiment?.has_trajectory_analysis) {
        return (
            <div className="trajectory-analysis">
                <div className="trajectory-unavailable">
                    <h3>Trajectory Analysis Unavailable</h3>
                    <p>This experiment does not have trajectory analysis data available.</p>
                </div>
            </div>
        );
    }

    if (loading) {
        return (
            <div className="trajectory-analysis">
                <div className="trajectory-loading">
                    <div className="loading-spinner"></div>
                    <p>Loading trajectory analysis data...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="trajectory-analysis">
                <div className="trajectory-error">
                    <h3>Error</h3>
                    <p>{error}</p>
                </div>
            </div>
        );
    }

    const analysisData = getCurrentAnalysisData();
    const promptGroups = getPromptGroups(analysisData);
    const globalLabelMappings = createGlobalLabelMappings(promptGroups);

    return (
        <div className="trajectory-analysis">
            <div className="trajectory-content">
                <h3>Trajectory Analysis</h3>

                {/* Section Tabs */}
                <div className="section-tabs">
                    {sections.map(section => (
                        <button
                            key={section.key}
                            className={`section-tab ${activeSection === section.key ? 'active' : ''}`}
                            onClick={() => setActiveSection(section.key)}
                        >
                            {section.label}
                        </button>
                    ))}
                </div>

                {/* Data Display */}
                {analysisData ? (
                    renderSectionData(analysisData)
                ) : (
                    <div className="no-data">
                        <p>No trajectory analysis data available for the selected normalization.</p>
                    </div>
                )}
            </div>

            {/* Global Legend */}
            {showGlobalLegend && currentExperiment && !showFullVariationText && promptGroups.length > 0 && (
                <div style={{
                    position: 'fixed',
                    bottom: '20px',
                    left: '20px',
                    backgroundColor: 'rgba(0, 0, 0, 0.85)',
                    padding: '12px',
                    borderRadius: '8px',
                    fontSize: '12px',
                    color: '#ffffff',
                    maxWidth: '250px',
                    maxHeight: '200px',
                    overflowY: 'auto',
                    zIndex: 50,
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)'
                }}>
                    <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: '8px',
                        borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
                        paddingBottom: '6px'
                    }}>
                        <strong>Prompt Legend</strong>
                        <button
                            onClick={() => setShowGlobalLegend(false)}
                            style={{
                                background: 'none',
                                border: 'none',
                                color: '#ffffff',
                                cursor: 'pointer',
                                fontSize: '14px',
                                padding: '0',
                                marginLeft: '8px'
                            }}
                        >
                            ×
                        </button>
                    </div>
                    {Object.entries(globalLabelMappings).map(([abbrev, full]) => (
                        <div key={abbrev} style={{ marginBottom: '4px', lineHeight: '1.3' }}>
                            <strong style={{ color: '#4A90E2' }}>{abbrev}:</strong> {full}
                        </div>
                    ))}
                </div>
            )}

            {/* Legend toggle when hidden */}
            {!showGlobalLegend && currentExperiment && !showFullVariationText && promptGroups.length > 0 && (
                <button
                    onClick={() => setShowGlobalLegend(true)}
                    style={{
                        position: 'fixed',
                        bottom: '20px',
                        left: '20px',
                        backgroundColor: 'rgba(0, 0, 0, 0.7)',
                        border: '1px solid #4A90E2',
                        color: '#ffffff',
                        borderRadius: '6px',
                        padding: '8px 12px',
                        fontSize: '11px',
                        cursor: 'pointer',
                        zIndex: 50,
                        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)'
                    }}
                    title="Show prompt legend"
                >
                    Show Legend
                </button>
            )}

            {/* Controls */}
            <TrajectoryAnalysisControls
                availableNormalizations={getAvailableNormalizations()}
                activeNormalization={activeNormalization}
                onNormalizationChange={setActiveNormalization}
                chartSize={chartSize}
                onChartSizeChange={setChartSize}
                beginAtZero={beginAtZero}
                onBeginAtZeroChange={setBeginAtZero}
                showFullVariationText={showFullVariationText}
                onShowFullVariationTextChange={setShowFullVariationText}
            />
        </div>
    );
};

export default TrajectoryAnalysis;
