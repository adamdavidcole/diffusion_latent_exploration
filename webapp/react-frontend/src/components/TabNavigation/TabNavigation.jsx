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
  const isAttentionBendingTab = location.pathname.endsWith('/attention-bending');
  const isVideosTab = location.pathname.endsWith('/videos');

  let activeTab = 'videos';
  if (isAnalysisTab) activeTab = 'analysis';
  if (isTrajectoryAnalysisTab) activeTab = 'trajectory-analysis';
  if (isLatentVideosTab) activeTab = 'latent-videos';
  if (isAttentionBendingTab) activeTab = 'attention-bending';
  if (isVideosTab) activeTab = 'videos';

  const handleTabChange = (tab) => {
    if (tab === 'videos') {
      navigate(`/experiment/${experimentPath}/videos`);
    } else if (tab === 'analysis') {
      navigate(`/experiment/${experimentPath}/analysis`);
    } else if (tab === 'trajectory-analysis') {
      navigate(`/experiment/${experimentPath}/trajectory-analysis`);
    } else if (tab === 'latent-videos') {
      navigate(`/experiment/${experimentPath}/latent-videos`);
    } else if (tab === 'attention-bending') {
      navigate(`/experiment/${experimentPath}/attention-bending`);
    }
  };

  // Check if experiment has attention bending
  const hasAttentionBending = currentExperiment?.attention_bending_settings?.enabled;

  return (
    <div className="tab-navigation">
      <div className="tab-list" role="tablist">
        {/* Only show Attention Bending tab if enabled - FIRST position */}
        {hasAttentionBending && (
          <button
            className={`tab-button ${activeTab === 'attention-bending' ? 'active' : ''}`}
            role="tab"
            aria-selected={activeTab === 'attention-bending'}
            onClick={() => handleTabChange('attention-bending')}
            title="View attention bending visualization and analysis"
          >
            🎛️ Attention Bending
          </button>
        )}

        {/* Videos tab - always shown */}
        <button
          className={`tab-button ${activeTab === 'videos' ? 'active' : ''}`}
          role="tab"
          aria-selected={activeTab === 'videos'}
          onClick={() => handleTabChange('videos')}
          title="View all videos in standard grid"
        >
          📹 Videos
        </button>

        {/* Only show VLM Analysis tab if available */}
        {currentExperiment?.has_vlm_analysis && (
          <button
            className={`tab-button ${activeTab === 'analysis' ? 'active' : ''}`}
            role="tab"
            aria-selected={activeTab === 'analysis'}
            onClick={() => handleTabChange('analysis')}
            title="View VLM analysis results"
          >
            📊 VLM Analysis
          </button>
        )}

        {/* Only show Trajectory Analysis tab if available */}
        {currentExperiment?.has_trajectory_analysis && (
          <button
            className={`tab-button ${activeTab === 'trajectory-analysis' ? 'active' : ''}`}
            role="tab"
            aria-selected={activeTab === 'trajectory-analysis'}
            onClick={() => handleTabChange('trajectory-analysis')}
            title="View trajectory analysis results"
          >
            📈 Trajectory Analysis
          </button>
        )}

        {/* Only show Latent Videos tab if available - SECOND position when attention bending enabled */}
        {(currentExperiment?.has_latent_videos || currentExperiment?.has_attention_videos) && (
          <button
            className={`tab-button ${activeTab === 'latent-videos' ? 'active' : ''}`}
            role="tab"
            aria-selected={activeTab === 'latent-videos'}
            onClick={() => handleTabChange('latent-videos')}
            title={
              currentExperiment?.has_attention_videos && !currentExperiment?.has_latent_videos
                ? 'View attention videos (latent videos not decoded)'
                : 'View latent videos progression'
            }
          >
            🎬 Latent Videos
          </button>
        )}
      </div>
    </div>
  );
};

export default TabNavigation;
