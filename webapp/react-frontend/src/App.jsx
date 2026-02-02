import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useParams, useNavigate } from 'react-router-dom';
import { AppProvider, useApp } from './context/AppContext';
import { useVideoCache } from './hooks/useVideoCache';
import { api } from './services/api';
import Sidebar from './components/Sidebar/Sidebar';
import VideoGrid from './components/VideoGrid/VideoGrid';
import SyncControls from './components/Controls/SyncControls';
import AttentionControls from './components/Controls/AttentionControls';
import AnalysisControls from './components/AnalysisControls/AnalysisControls';
import ExperimentHeader from './components/ExperimentHeader/ExperimentHeader';
import TabNavigation from './components/TabNavigation/TabNavigation';
import AnalysisDashboard from './components/AnalysisDashboard/AnalysisDashboard';
import TrajectoryAnalysis from './components/TrajectoryAnalysis/TrajectoryAnalysis';
import LatentVideosView from './components/LatentVideosView/LatentVideosView';
import AttentionBendingView from './components/AttentionBendingView/AttentionBendingView';
import SimilarityMetricsModal from './components/SimilarityMetricsModal/SimilarityMetricsModal';
import FullSearchModal from './components/FullSearchModal/FullSearchModal';
import './styles.css';

// Experiment Route Handler
const ExperimentRoute = () => {
  const { "*": experimentPath } = useParams(); // Capture full path
  const navigate = useNavigate();
  const { state, actions } = useApp();

  // Determine route type
  const isAnalysisRoute = experimentPath && experimentPath.endsWith('/analysis');
  const isTrajectoryAnalysisRoute = experimentPath && experimentPath.endsWith('/trajectory-analysis');
  const isLatentVideosRoute = experimentPath && experimentPath.endsWith('/latent-videos');
  const isAttentionBendingRoute = experimentPath && experimentPath.endsWith('/attention-bending');

  const cleanExperimentPath = isAnalysisRoute
    ? experimentPath.replace('/analysis', '')
    : isTrajectoryAnalysisRoute
      ? experimentPath.replace('/trajectory-analysis', '')
      : isLatentVideosRoute
        ? experimentPath.replace('/latent-videos', '')
        : isAttentionBendingRoute
          ? experimentPath.replace('/attention-bending', '')
          : experimentPath;

  // Load specific experiment when experimentPath changes
  useEffect(() => {
    const loadExperiment = async () => {
      if (!cleanExperimentPath) return;

      try {
        actions.setLoading(true);
        actions.clearError();

        // Load experiments tree if not already loaded
        if (!state.experimentsTree) {
          const experimentsData = await api.getExperimentsSummary(); // Use fast summary endpoint
          actions.setExperimentsTree(experimentsData);

          // Also maintain flat list for backward compatibility
          const flatExperiments = api.flattenExperimentTree(experimentsData);
          actions.setExperiments(flatExperiments);

          // Try to find experiment in the tree structure
          const experiment = findExperimentInTree(experimentsData, cleanExperimentPath);
          if (!experiment) {
            console.warn(`Experiment "${cleanExperimentPath}" not found, redirecting to home`);
            navigate('/', { replace: true });
            return;
          }

          // Load experiment details
          const experimentData = await api.getExperiment(cleanExperimentPath);
          actions.setCurrentExperiment(experimentData);
        } else {
          // Try to find experiment in already loaded tree
          const experiment = findExperimentInTree(state.experimentsTree, cleanExperimentPath);
          if (!experiment) {
            console.warn(`Experiment "${cleanExperimentPath}" not found, redirecting to home`);
            navigate('/', { replace: true });
            return;
          }

          // Only load experiment details if it's not the current one
          if (!state.currentExperiment ||
            state.currentExperiment.name !== experiment.experiment_data.name) {
            const experimentData = await api.getExperiment(cleanExperimentPath);
            actions.setCurrentExperiment(experimentData);
          }
        }
      } catch (error) {
        console.error('Error loading experiment:', error);
        actions.setError(`Failed to load experiment: ${cleanExperimentPath}`);
      } finally {
        actions.setLoading(false);
      }
    };

    loadExperiment();
  }, [cleanExperimentPath, navigate]); // Removed state and actions dependencies to prevent loops

  return <AppContent experimentPath={cleanExperimentPath} isAnalysisRoute={isAnalysisRoute} isTrajectoryAnalysisRoute={isTrajectoryAnalysisRoute} isLatentVideosRoute={isLatentVideosRoute} isAttentionBendingRoute={isAttentionBendingRoute} />;
};

// Helper function to find experiment in tree by path
const findExperimentInTree = (tree, targetPath) => {
  if (!tree) return null;

  const traverse = (node) => {
    if (node.type === 'experiment') {
      const nodePath = node.path.replace(/^outputs\//, '');
      if (nodePath === targetPath) {
        return node;
      }
    } else if (node.children) {
      for (const child of node.children) {
        const result = traverse(child);
        if (result) return result;
      }
    }
    return null;
  };

  // Helper function to find the first experiment alphabetically (matching TreeExperimentList logic)
  const findFirstExperimentAlphabetically = (tree) => {
    if (!tree) return null;

    const sortChildren = (children) => {
      return children.sort((a, b) => {
        // Always folders first
        if (a.type !== b.type) {
          return a.type === 'folder' ? -1 : 1;
        }
        // Sort alphabetically
        return a.name.localeCompare(b.name);
      });
    };

    const findFirst = (node) => {
      if (node.type === 'experiment') {
        // Apply same filters as TreeExperimentList (default: minVideoCount = 20, no model filter, no search)
        if (node.experiment_data.videos_count >= 20) {
          return node;
        }
        return null;
      }

      if (node.type === 'folder' && node.children) {
        const sortedChildren = sortChildren(node.children);
        for (const child of sortedChildren) {
          const result = findFirst(child);
          if (result) return result;
        }
      }

      return null;
    };

    // Skip the top-level "outputs" folder and search its children
    if (tree.name === 'outputs' && tree.children) {
      const sortedChildren = sortChildren(tree.children);
      for (const child of sortedChildren) {
        const result = findFirst(child);
        if (result) return result;
      }
    } else {
      return findFirst(tree);
    }

    return null;
  }; return traverse(tree);
};

// Helper function to find the first experiment alphabetically (matching TreeExperimentList logic)
const findFirstExperimentAlphabetically = (tree) => {
  if (!tree) return null;

  const sortChildren = (children) => {
    return children.sort((a, b) => {
      // Always folders first
      if (a.type !== b.type) {
        return a.type === 'folder' ? -1 : 1;
      }
      // Sort alphabetically
      return a.name.localeCompare(b.name);
    });
  };

  const findFirst = (node) => {
    if (node.type === 'experiment') {
      // Apply same filters as TreeExperimentList (default: minVideoCount = 20, no model filter, no search)
      if (node.experiment_data.videos_count >= 20) {
        return node;
      }
      return null;
    }

    if (node.type === 'folder' && node.children) {
      const sortedChildren = sortChildren(node.children);
      for (const child of sortedChildren) {
        const result = findFirst(child);
        if (result) return result;
      }
    }

    return null;
  };

  // Skip the top-level "outputs" folder and search its children
  if (tree.name === 'outputs' && tree.children) {
    const sortedChildren = sortChildren(tree.children);
    for (const child of sortedChildren) {
      const result = findFirst(child);
      if (result) return result;
    }
  } else {
    return findFirst(tree);
  }

  return null;
};

// Home Route Handler
const HomeRoute = () => {
  const { state, actions } = useApp();
  const navigate = useNavigate();

  // Auto-load experiments and select first one
  useEffect(() => {
    const loadExperiments = async () => {
      if (state.experimentsTree) return; // Already loaded

      try {
        actions.setLoading(true);
        const experimentsData = await api.getExperimentsSummary(); // Use fast summary endpoint
        actions.setExperimentsTree(experimentsData);

        // Also maintain flat list for backward compatibility
        const flatExperiments = api.flattenExperimentTree(experimentsData);
        actions.setExperiments(flatExperiments);

        // Auto-select first experiment and redirect to it
        if (flatExperiments.length > 0 && !state.currentExperiment) {
          // Find the first experiment alphabetically (matching TreeExperimentList logic)
          const firstExperiment = findFirstExperimentAlphabetically(experimentsData);
          if (firstExperiment) {
            const experimentPath = firstExperiment.path.replace(/^outputs\//, '');
            navigate(`/experiment/${experimentPath}`, { replace: true });
          }
        }
      } catch (error) {
        console.error('Error loading experiments:', error);
        actions.setError('Failed to load experiments. Check the server connection.');
      } finally {
        actions.setLoading(false);
      }
    };

    loadExperiments();
  }, [navigate]); // Removed state dependencies

  return <AppContent experimentPath={null} isAnalysisRoute={false} isTrajectoryAnalysisRoute={false} isLatentVideosRoute={false} />;
};

// Main App Content Component
const AppContent = ({ experimentPath, isAnalysisRoute, isTrajectoryAnalysisRoute, isLatentVideosRoute, isAttentionBendingRoute }) => {
  const { state, actions } = useApp();
  const { clearCache } = useVideoCache();

  // Hide the initial loader once React is mounted
  useEffect(() => {
    document.body.classList.add('app-loaded');
  }, []);

  // Set up mobile responsiveness
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth <= 900) {
        // Could trigger sidebar collapse on mobile
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize(); // Check on mount

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  // Clean up cache when component unmounts
  useEffect(() => {
    return () => {
      clearCache();
    };
  }, [clearCache]);

  return (
    <div className="app-container">
        {/* Invisible hover zone for show sidebar button */}
        {state.sidebarHidden && (
          <div className="sidebar-hover-zone">
            <button
              className="show-sidebar-btn"
              onClick={actions.toggleSidebarHidden}
              aria-label="Show sidebar"
              title="Show sidebar"
            >
              <span className="show-icon">☰</span>
            </button>
          </div>
        )}

        <Sidebar />

      <div className="main-content">
        <div id="video-grid-wrapper">
          {state.isLoading && state.experiments.length === 0 && (
            <div className="skeleton-content">
              <div className="skeleton-header">
                <div className="skeleton-title"></div>
                <div className="skeleton-subtitle"></div>
              </div>
              <div className="skeleton-grid">
                {[...Array(6)].map((_, i) => (
                  <div key={i} className="skeleton-video-item">
                    <div className="skeleton-video"></div>
                    <div className="skeleton-label"></div>
                  </div>
                ))}
              </div>
              <div className="loading-text">
                <div className="loading-spinner"></div>
                <p>Loading experiments from server...</p>
              </div>
            </div>
          )}

          {state.isLoading && state.experiments.length > 0 && (
            <div className="loading">
              <div className="loading-spinner"></div>
              <p>Loading experiment data...</p>
            </div>
          )}

          {state.error && (
            <div className="error-message">
              <h3>Error</h3>
              <p>{state.error}</p>
            </div>
          )}

          {!state.isLoading && !state.error && state.currentExperiment && (
            <>
              <ExperimentHeader />
              <TabNavigation experimentPath={experimentPath} />

              {isAnalysisRoute ? (
                <AnalysisDashboard experimentPath={experimentPath} />
              ) : isTrajectoryAnalysisRoute ? (
                <TrajectoryAnalysis experimentPath={experimentPath} />
              ) : isLatentVideosRoute ? (
                <LatentVideosView experimentPath={experimentPath} />
              ) : isAttentionBendingRoute ? (
                <AttentionBendingView experimentPath={experimentPath} />
              ) : (
                <VideoGrid />
              )}
            </>
          )}

          {!state.isLoading && !state.error && !state.currentExperiment && (
            <div className="empty-state">
              <h3>No experiment selected</h3>
              <p>Select an experiment from the sidebar to view videos.</p>
            </div>
          )}
        </div>

        <div className="main-controls">
          {!isAnalysisRoute && !isTrajectoryAnalysisRoute && !isLatentVideosRoute && !isAttentionBendingRoute && <AttentionControls />}
          {isAnalysisRoute && <AnalysisControls />}
          {!isAnalysisRoute && !isTrajectoryAnalysisRoute && !isLatentVideosRoute && !isAttentionBendingRoute && <SyncControls />}
        </div>
      </div>

      {/* Similarity Metrics Modal */}
      <SimilarityMetricsModal />
      
      {/* Full Search Modal */}
      <FullSearchModal />
    </div>
  );
};

// Root App Component with Provider and Router
function App() {
  return (
    <AppProvider>
      <Router>
        <Routes>
          <Route path="/" element={<HomeRoute />} />
          <Route path="/experiment/*" element={<ExperimentRoute />} />
        </Routes>
      </Router>
    </AppProvider>
  );
}

export default App;
