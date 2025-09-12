import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useApp } from '../../context/AppContext';
import './TabNavigation.css';

const TabNavigation = ({ experimentPath }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const { state } = useApp();
  const { currentExperiment } = state;

  // Determine active tab based on current path
  const isAnalysisTab = location.pathname.endsWith('/analysis');
  const isTrajectoryAnalysisTab = location.pathname.endsWith('/trajectory-analysis');
  const isLatentVideosTab = location.pathname.endsWith('/latent-videos');
  
  let activeTab = 'videos';
  if (isAnalysisTab) activeTab = 'analysis';
  if (isTrajectoryAnalysisTab) activeTab = 'trajectory-analysis';
  if (isLatentVideosTab) activeTab = 'latent-videos';

  const handleTabChange = (tab) => {
    if (tab === 'videos') {
      navigate(`/experiment/${experimentPath}`);
    } else if (tab === 'analysis') {
      navigate(`/experiment/${experimentPath}/analysis`);
    } else if (tab === 'trajectory-analysis') {
      navigate(`/experiment/${experimentPath}/trajectory-analysis`);
    } else if (tab === 'latent-videos') {
      navigate(`/experiment/${experimentPath}/latent-videos`);
    }
  };

  return (
    <div className="tab-navigation">
      <div className="tab-list" role="tablist">
        <button
          className={`tab-button ${activeTab === 'videos' ? 'active' : ''}`}
          role="tab"
          aria-selected={activeTab === 'videos'}
          onClick={() => handleTabChange('videos')}
        >
          📹 Videos
        </button>

        <button
          className={`tab-button ${activeTab === 'analysis' ? 'active' : ''} ${!currentExperiment?.has_vlm_analysis ? 'disabled' : ''
            }`}
          role="tab"
          aria-selected={activeTab === 'analysis'}
          onClick={() => currentExperiment?.has_vlm_analysis && handleTabChange('analysis')}
          disabled={!currentExperiment?.has_vlm_analysis}
          title={
            !currentExperiment?.has_vlm_analysis
              ? 'VLM analysis not available for this experiment'
              : 'View VLM analysis results'
          }
        >
          📊 VLM Analysis
          {!currentExperiment?.has_vlm_analysis && (
            <span className="disabled-indicator"> (unavailable)</span>
          )}
        </button>

        <button
          className={`tab-button ${activeTab === 'trajectory-analysis' ? 'active' : ''} ${!currentExperiment?.has_trajectory_analysis ? 'disabled' : ''
            }`}
          role="tab"
          aria-selected={activeTab === 'trajectory-analysis'}
          onClick={() => currentExperiment?.has_trajectory_analysis && handleTabChange('trajectory-analysis')}
          disabled={!currentExperiment?.has_trajectory_analysis}
          title={
            !currentExperiment?.has_trajectory_analysis
              ? 'Trajectory analysis not available for this experiment'
              : 'View trajectory analysis results'
          }
        >
          📈 Trajectory Analysis
          {!currentExperiment?.has_trajectory_analysis && (
            <span className="disabled-indicator"> (unavailable)</span>
          )}
        </button>

        <button
          className={`tab-button ${activeTab === 'latent-videos' ? 'active' : ''} ${!currentExperiment?.has_latent_videos ? 'disabled' : ''
            }`}
          role="tab"
          aria-selected={activeTab === 'latent-videos'}
          onClick={() => currentExperiment?.has_latent_videos && handleTabChange('latent-videos')}
          disabled={!currentExperiment?.has_latent_videos}
          title={
            !currentExperiment?.has_latent_videos
              ? 'Latent videos not available for this experiment'
              : 'View latent videos progression'
          }
        >
          🎬 Latent Videos
          {!currentExperiment?.has_latent_videos && (
            <span className="disabled-indicator"> (unavailable)</span>
          )}
        </button>
      </div>
    </div>
  );
};

export default TabNavigation;
