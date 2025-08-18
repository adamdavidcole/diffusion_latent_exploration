import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import TrajectoryAnalysisControls from './TrajectoryAnalysisControls';
import './TrajectoryAnalysis.css';

const TrajectoryAnalysis = ({ experimentPath }) => {
  const { state } = useApp();
  const { currentExperiment } = state;

  // Local state for trajectory analysis
  const [trajectoryData, setTrajectoryData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeSection, setActiveSection] = useState('temporal');
  const [activeNormalization, setActiveNormalization] = useState('full_norm');

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
          if (availableNorms.includes('full_norm')) {
            setActiveNormalization('full_norm');
          } else if (availableNorms.includes('snr_norm_only')) {
            setActiveNormalization('snr_norm_only');
          } else if (availableNorms.includes('no_norm')) {
            setActiveNormalization('no_norm');
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
        return (
          <div className="section-data">
            <h4>Temporal Analysis - Trajectory Length</h4>
            <div className="data-grid">
              {promptGroups.map(promptGroup => {
                const meanLength = analysisData?.temporal_analysis?.[promptGroup]?.trajectory_length?.mean_length;
                return (
                  <div key={promptGroup} className="data-item">
                    <span className="prompt-label">{promptGroup}:</span>
                    <span className="data-value">{meanLength?.toFixed(4) || 'N/A'}</span>
                  </div>
                );
              })}
            </div>
          </div>
        );

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

      {/* Controls */}
      <TrajectoryAnalysisControls
        availableNormalizations={getAvailableNormalizations()}
        activeNormalization={activeNormalization}
        onNormalizationChange={setActiveNormalization}
      />
    </div>
  );
};

export default TrajectoryAnalysis;
