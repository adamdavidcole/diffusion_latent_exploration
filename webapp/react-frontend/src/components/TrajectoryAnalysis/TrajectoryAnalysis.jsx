import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import TrajectoryAnalysisControls from './TrajectoryAnalysisControls';
import MetricComparisonChart from '../Charts/MetricComparisonChart';
import ScatterChart from '../Charts/ScatterChart';
import TrajectoryChartModal from '../Charts/TrajectoryChartModal';
import { getVariationTextFromPromptKey } from '../../utils/variationText';
import TrajectoryInfoTooltip from './TrajectoryInfoTooltip';
import { extractChartData } from '../../utils/trajectoryDataHelpers';
import './TrajectoryAnalysis.css';

const CHART_DEFAULT_SIZE = 600;

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

    // Modal state
    const [modalState, setModalState] = useState({
        isOpen: false,
        chartData: null,
        metricKey: null,
        title: null
    });

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

    // Modal functions
    const openChartModal = (chartData, metricKey, title) => {
        setModalState({
            isOpen: true,
            chartData,
            metricKey,
            title
        });
    };

    const closeChartModal = () => {
        setModalState({
            isOpen: false,
            chartData: null,
            metricKey: null,
            title: null
        });
    };

    // Render data for each section
    const renderSectionData = (analysisData) => {
        const promptGroups = getPromptGroups(analysisData);

        switch (activeSection) {
            case 'temporal':
                return renderTemporalAnalysis(analysisData, promptGroups);

            case 'geometric':
                return renderGeometricAnalysis(analysisData, promptGroups);

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
        // Use the data helper to extract chart data
        const chartData = extractChartData({ [activeNormalization]: analysisData }, activeNormalization);

        return (
            <div className="section-data">
                <h4>Temporal Analysis</h4>

                <div className="metrics-grid" style={{ '--chart-size': `${chartSize}px` }}>
                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>
                            Trajectory Length
                            <TrajectoryInfoTooltip metricKey="trajectory_length" title="Trajectory Length" />
                        </h5>
                        <div
                            className="clickable-chart"
                            onClick={() => openChartModal(chartData.trajectoryLength, 'trajectory_length', 'Trajectory Length')}
                            style={{ cursor: 'pointer' }}
                            title="Click for detailed view"
                        >
                            <MetricComparisonChart
                                data={chartData.trajectoryLength}
                                title="Mean Length"
                                size={chartSize}
                                yLabel="Length Units"
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        </div>
                    </div>

                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>
                            Velocity Analysis
                            <TrajectoryInfoTooltip metricKey="velocity_analysis" title="Velocity Analysis" />
                        </h5>
                        <div
                            className="clickable-chart"
                            onClick={() => openChartModal(chartData.velocity, 'velocity_analysis', 'Velocity Analysis')}
                            style={{ cursor: 'pointer' }}
                            title="Click for detailed view"
                        >
                            <MetricComparisonChart
                                data={chartData.velocity}
                                title="Overall Mean Velocity"
                                size={chartSize}
                                yLabel="Velocity"
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        </div>
                    </div>

                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>
                            Acceleration Analysis
                            <TrajectoryInfoTooltip metricKey="acceleration_analysis" title="Acceleration Analysis" />
                        </h5>
                        <div
                            className="clickable-chart"
                            onClick={() => openChartModal(chartData.acceleration, 'acceleration_analysis', 'Acceleration Analysis')}
                            style={{ cursor: 'pointer' }}
                            title="Click for detailed view"
                        >
                            <MetricComparisonChart
                                data={chartData.acceleration}
                                title="Overall Mean Acceleration"
                                size={chartSize}
                                yLabel="Acceleration"
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        </div>
                    </div>

                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>
                            Tortuosity
                            <TrajectoryInfoTooltip metricKey="tortuosity" title="Tortuosity" />
                        </h5>
                        <div
                            className="clickable-chart"
                            onClick={() => openChartModal(chartData.tortuosity, 'tortuosity', 'Tortuosity')}
                            style={{ cursor: 'pointer' }}
                            title="Click for detailed view"
                        >
                            <MetricComparisonChart
                                data={chartData.tortuosity}
                                title="Mean Tortuosity"
                                size={chartSize}
                                yLabel="Tortuosity"
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        </div>
                    </div>

                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>
                            Endpoint Distance
                            <TrajectoryInfoTooltip metricKey="endpoint_distance" title="Endpoint Distance" />
                        </h5>
                        <div
                            className="clickable-chart"
                            onClick={() => openChartModal(chartData.endpointDistance, 'endpoint_distance', 'Endpoint Distance')}
                            style={{ cursor: 'pointer' }}
                            title="Click for detailed view"
                        >
                            <MetricComparisonChart
                                data={chartData.endpointDistance}
                                title="Mean Endpoint Distance"
                                size={chartSize}
                                yLabel="Distance"
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        </div>
                    </div>

                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>
                            Semantic Convergence
                            <TrajectoryInfoTooltip metricKey="semantic_convergence" title="Semantic Convergence" />
                        </h5>
                        <div
                            className="clickable-chart"
                            onClick={() => openChartModal(chartData.semanticConvergence, 'semantic_convergence', 'Semantic Convergence')}
                            style={{ cursor: 'pointer' }}
                            title="Click for detailed view"
                        >
                            <MetricComparisonChart
                                data={chartData.semanticConvergence}
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
            </div>
        );
    };

    // Render geometric analysis with charts
    const renderGeometricAnalysis = (analysisData, promptGroups) => {
        // Use the data helper to extract chart data
        const chartData = extractChartData({ [activeNormalization]: analysisData }, activeNormalization);

        return (
            <div className="section-data">
                <h4>Geometric Analysis</h4>

                <div className="metrics-grid" style={{ '--chart-size': `${chartSize}px` }}>
                    {/* <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>
                            Speed Statistics
                            <TrajectoryInfoTooltip metricKey="speed_stats" title="Speed Statistics" />
                        </h5>
                        <MetricComparisonChart
                            data={speedData}
                            title="Mean Speed"
                            size={chartSize}
                            yLabel="Speed Units"
                            currentExperiment={currentExperiment}
                            beginAtZero={beginAtZero}
                            showFullVariationText={showFullVariationText}
                        />
                    </div> */}

                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>
                            Log Volume Statistics
                            <TrajectoryInfoTooltip metricKey="log_volume_stats" title="Log Volume Statistics" />
                        </h5>
                        <div
                            className="clickable-chart"
                            onClick={() => openChartModal(chartData.logVolume, 'log_volume_stats', 'Log Volume Statistics')}
                            style={{ cursor: 'pointer' }}
                            title="Click for detailed view"
                        >
                            <MetricComparisonChart
                                data={chartData.logVolume}
                                title="Mean Log Volume"
                                size={chartSize}
                                yLabel="Log Volume"
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        </div>
                    </div>

                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>
                            Effective Side Statistics
                            <TrajectoryInfoTooltip metricKey="effective_side_stats" title="Effective Side Statistics" />
                        </h5>
                        <div
                            className="clickable-chart"
                            onClick={() => openChartModal(chartData.effectiveSide, 'effective_side_stats', 'Effective Side Statistics')}
                            style={{ cursor: 'pointer' }}
                            title="Click for detailed view"
                        >
                            <MetricComparisonChart
                                data={chartData.effectiveSide}
                                title="Mean Effective Side"
                                size={chartSize}
                                yLabel="Effective Side"
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        </div>
                    </div>

                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>
                            Endpoint Alignment
                            <TrajectoryInfoTooltip metricKey="endpoint_alignment_stats" title="Endpoint Alignment" />
                        </h5>
                        <div
                            className="clickable-chart"
                            onClick={() => openChartModal(chartData.endpointAlignment, 'endpoint_alignment_stats', 'Endpoint Alignment')}
                            style={{ cursor: 'pointer' }}
                            title="Click for detailed view"
                        >
                            <MetricComparisonChart
                                data={chartData.endpointAlignment}
                                title="Mean Endpoint Alignment"
                                size={chartSize}
                                yLabel="Alignment"
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        </div>
                    </div>

                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>
                            Turning Angle Statistics
                            <TrajectoryInfoTooltip metricKey="turning_angle_stats" title="Turning Angle Statistics" />
                        </h5>
                        <div
                            className="clickable-chart"
                            onClick={() => openChartModal(chartData.turningAngle, 'turning_angle_stats', 'Turning Angle Statistics')}
                            style={{ cursor: 'pointer' }}
                            title="Click for detailed view"
                        >
                            <MetricComparisonChart
                                data={chartData.turningAngle}
                                title="Mean Turning Angle"
                                size={chartSize}
                                yLabel="Angle (radians)"
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        </div>
                    </div>

                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>
                            Circuitousness Statistics
                            <TrajectoryInfoTooltip metricKey="circuitousness_stats" title="Circuitousness Statistics" />
                        </h5>
                        <div
                            className="clickable-chart"
                            onClick={() => openChartModal(chartData.circuitousness, 'circuitousness_stats', 'Circuitousness Statistics')}
                            style={{ cursor: 'pointer' }}
                            title="Click for detailed view"
                        >
                            <MetricComparisonChart
                                data={chartData.circuitousness}
                                title="Mean Circuitousness"
                                size={chartSize}
                                yLabel="Circuitousness"
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        </div>
                    </div>

                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>
                            Efficiency Metrics
                            <TrajectoryInfoTooltip metricKey="efficiency_metrics" title="Efficiency Metrics" />
                        </h5>
                        <div
                            className="clickable-chart"
                            onClick={() => openChartModal(chartData.efficiency, 'efficiency_metrics', 'Efficiency Metrics')}
                            style={{ cursor: 'pointer' }}
                            title="Click for detailed view"
                        >
                            <MetricComparisonChart
                                data={chartData.efficiency}
                                title="Mean Efficiency"
                                size={chartSize}
                                yLabel="Efficiency"
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        </div>
                    </div>

                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>
                            Step Variability Statistics
                            <TrajectoryInfoTooltip metricKey="step_variability_stats" title="Step Variability Statistics" />
                        </h5>
                        <div
                            className="clickable-chart"
                            onClick={() => openChartModal(chartData.stepVariability, 'step_variability_stats', 'Step Variability Statistics')}
                            style={{ cursor: 'pointer' }}
                            title="Click for detailed view"
                        >
                            <MetricComparisonChart
                                data={chartData.stepVariability}
                                title="Mean Step Variability"
                                size={chartSize}
                                yLabel="Variability"
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        </div>
                    </div>

                    {/* Scatter Plot: Velocity vs Log Volume */}
                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>
                            Velocity vs Log Volume
                            <TrajectoryInfoTooltip metricKey="velocity_vs_log_volume" title="Velocity vs Log Volume Scatter Plot" />
                        </h5>
                        <div
                            className="clickable-chart"
                            onClick={() => openChartModal(chartData.velocityVsLogVolume, 'velocity_vs_log_volume', 'Velocity vs Log Volume')}
                            style={{ cursor: 'pointer' }}
                            title="Click for detailed view"
                        >
                            <ScatterChart
                                data={chartData.velocityVsLogVolume}
                                title="Velocity vs Log Volume (points = trajectories)"
                                size={chartSize}
                                xLabel="Velocity (mean per trajectory)"
                                yLabel="Log Volume"
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        </div>
                    </div>

                    {/* Scatter Plot: Velocity vs Circuitousness */}
                    <div className="metric-chart-container" style={{ width: `${chartSize}px` }}>
                        <h5>
                            Velocity vs Circuitousness
                            <TrajectoryInfoTooltip metricKey="velocity_vs_circuitousness" title="Velocity vs Circuitousness Scatter Plot" />
                        </h5>
                        <div
                            className="clickable-chart"
                            onClick={() => openChartModal(chartData.velocityVsCircuitousness, 'velocity_vs_circuitousness', 'Velocity vs Circuitousness')}
                            style={{ cursor: 'pointer' }}
                            title="Click for detailed view"
                        >
                            <ScatterChart
                                data={chartData.velocityVsCircuitousness}
                                title="Velocity vs Circuitousness (points = trajectories)"
                                size={chartSize}
                                xLabel="Velocity (mean per trajectory)"
                                yLabel="Circuitousness − 1.0"
                                currentExperiment={currentExperiment}
                                beginAtZero={beginAtZero}
                                showFullVariationText={showFullVariationText}
                            />
                        </div>
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

            {/* Chart Detail Modal */}
            <TrajectoryChartModal
                isOpen={modalState.isOpen}
                onClose={closeChartModal}
                chartData={modalState.chartData}
                metricKey={modalState.metricKey}
                title={modalState.title}
                analysisData={analysisData}
                promptGroups={promptGroups}
                currentExperiment={currentExperiment}
                chartSize={chartSize}
                beginAtZero={beginAtZero}
                showFullVariationText={showFullVariationText}
                onShowFullVariationTextChange={setShowFullVariationText}
                activeNormalization={activeNormalization}
            />
        </div>
    );
};

export default TrajectoryAnalysis;
