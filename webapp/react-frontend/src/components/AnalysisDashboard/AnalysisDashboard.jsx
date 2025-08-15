import React, { useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import './AnalysisDashboard.css';

const AnalysisDashboard = ({ experimentPath }) => {
  const { state, actions } = useApp();
  const { currentAnalysis, analysisLoading, analysisError, currentExperiment } = state;

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

  // Helper function to safely get gender counts
  const getGenderCounts = (promptGroupData) => {
    try {
      return promptGroupData?.aggregated_data?.people?.sections?.demographics?.gender?.data?.counts || {};
    } catch (error) {
      console.error('Error accessing gender data:', error);
      return {};
    }
  };

  // Helper function to get variation name from prompt group
  const getVariationName = (promptGroupKey, promptGroupData) => {
    // Try to get variation name from the experiment's video grid
    if (currentExperiment?.video_grid) {
      const matchingVariation = currentExperiment.video_grid.find(v => 
        v.variation_id === promptGroupKey || 
        v.variation_id === `variation_${promptGroupKey.split('_')[1]}`
      );
      if (matchingVariation) {
        return matchingVariation.variation;
      }
    }
    
    // Fallback to prompt group key
    return promptGroupKey.replace('prompt_', 'Prompt ');
  };

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

  if (!currentAnalysis?.vlm_analysis) {
    return (
      <div className="analysis-dashboard">
        <div className="analysis-unavailable">
          <h3>No Analysis Data</h3>
          <p>VLM analysis data is not available for this experiment.</p>
        </div>
      </div>
    );
  }

  const { prompt_groups, overall } = currentAnalysis.vlm_analysis;

  return (
    <div className="analysis-dashboard">
      <div className="analysis-content">
        <h3>VLM Analysis Results</h3>
        
        {/* Overall Statistics */}
        {overall && (
          <div className="analysis-section">
            <h4>Overall Experiment Statistics</h4>
            <div className="overall-stats">
              <div className="stat-card">
                <span className="stat-label">Total Videos</span>
                <span className="stat-value">{overall.metadata?.successfully_loaded || 0}</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Total People</span>
                <span className="stat-value">{overall.aggregated_data?.people?.total_people || 0}</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Female Count</span>
                <span className="stat-value">
                  {getGenderCounts(overall)?.Female || 0}
                </span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Male Count</span>
                <span className="stat-value">
                  {getGenderCounts(overall)?.Male || 0}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Prompt Group Analysis */}
        <div className="analysis-section">
          <h4>Analysis by Prompt Group</h4>
          
          {Object.keys(prompt_groups).length === 0 ? (
            <p>No prompt group data available.</p>
          ) : (
            <div className="prompt-groups">
              {Object.entries(prompt_groups).map(([promptKey, promptData]) => {
                const genderCounts = getGenderCounts(promptData);
                const variationName = getVariationName(promptKey, promptData);
                
                return (
                  <div key={promptKey} className="prompt-group-card">
                    <h5 className="prompt-group-title">{variationName}</h5>
                    
                    <div className="prompt-stats">
                      <div className="stat-item">
                        <span className="stat-label">Videos:</span>
                        <span className="stat-value">{promptData.metadata?.successfully_loaded || 0}</span>
                      </div>
                      
                      <div className="stat-item">
                        <span className="stat-label">People:</span>
                        <span className="stat-value">{promptData.aggregated_data?.people?.total_people || 0}</span>
                      </div>
                      
                      <div className="stat-item">
                        <span className="stat-label">Female:</span>
                        <span className="stat-value">{genderCounts.Female || 0}</span>
                      </div>
                      
                      <div className="stat-item">
                        <span className="stat-label">Male:</span>
                        <span className="stat-value">{genderCounts.Male || 0}</span>
                      </div>
                      
                      {genderCounts.Ambiguous && (
                        <div className="stat-item">
                          <span className="stat-label">Ambiguous:</span>
                          <span className="stat-value">{genderCounts.Ambiguous}</span>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AnalysisDashboard;
