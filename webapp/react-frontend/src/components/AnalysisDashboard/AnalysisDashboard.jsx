import React, { useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import AnalysisGrid from '../AnalysisGrid/AnalysisGrid';
import './AnalysisDashboard.css';

const AnalysisDashboard = ({ experimentPath }) => {
  const { state, actions } = useApp();
  const {
    currentAnalysis,
    analysisLoading,
    analysisError,
    currentExperiment,
    analysisSchema
  } = state;

  // Load analysis data when component mounts or experiment changes
  useEffect(() => {
    const loadAnalysisData = async () => {
      if (!experimentPath || !currentExperiment?.has_vlm_analysis) {
        return;
      }

      try {
        actions.setAnalysisLoading(true);
        actions.clearAnalysisError();

        const analysisData = await api.getExperimentAnalysis(experimentPath);
        actions.setCurrentAnalysis(analysisData);
      } catch (error) {
        console.error('Error loading analysis data:', error);
        actions.setAnalysisError(`Failed to load analysis data: ${error.message}`);
      } finally {
        actions.setAnalysisLoading(false);
      }
    };

    loadAnalysisData();
  }, [experimentPath, currentExperiment?.has_vlm_analysis]);

  if (!currentExperiment?.has_vlm_analysis) {
    return (
      <div className="analysis-dashboard">
        <div className="analysis-unavailable">
          <h3>VLM Analysis Not Available</h3>
          <p>This experiment does not have VLM analysis data.</p>
          <p>To generate analysis data, run the VLM analysis script on this experiment.</p>
        </div>
      </div>
    );
  }

  if (analysisLoading) {
    return (
      <div className="analysis-dashboard">
        <div className="analysis-loading">
          <div className="loading-spinner"></div>
          <p>Loading VLM analysis data...</p>
        </div>
      </div>
    );
  }

  if (analysisError) {
    return (
      <div className="analysis-dashboard">
        <div className="analysis-error">
          <h3>Error Loading Analysis</h3>
          <p>{analysisError}</p>
          <button
            onClick={() => window.location.reload()}
            className="retry-button"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!currentAnalysis?.vlm_analysis || !analysisSchema) {
    return (
      <div className="analysis-dashboard">
        <div className="analysis-unavailable">
          <h3>No Analysis Data</h3>
          <p>VLM analysis data or schema is not available for this experiment.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="analysis-dashboard">
      <AnalysisGrid />
    </div>
  );
};

export default AnalysisDashboard;
