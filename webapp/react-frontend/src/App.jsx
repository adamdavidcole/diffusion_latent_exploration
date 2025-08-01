import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useParams, useNavigate } from 'react-router-dom';
import { AppProvider, useApp } from './context/AppContext';
import { useVideoCache } from './hooks/useVideoCache';
import { api } from './services/api';
import Sidebar from './components/Sidebar/Sidebar';
import VideoGrid from './components/VideoGrid/VideoGrid';
import SyncControls from './components/Controls/SyncControls';
import './styles.css';

// Experiment Route Handler
const ExperimentRoute = () => {
  const { "*": experimentPath } = useParams(); // Capture full path
  const navigate = useNavigate();
  const { state, actions } = useApp();

  // Load specific experiment when experimentPath changes
  useEffect(() => {
    const loadExperiment = async () => {
      if (!experimentPath) return;

      try {
        actions.setLoading(true);
        actions.clearError();

        // Load experiments tree if not already loaded
        if (!state.experimentsTree) {
          const experimentsData = await api.getExperiments();
          actions.setExperimentsTree(experimentsData);
          
          // Also maintain flat list for backward compatibility
          const flatExperiments = api.flattenExperimentTree(experimentsData);
          actions.setExperiments(flatExperiments);

          // Try to find experiment in the tree structure
          const experiment = findExperimentInTree(experimentsData, experimentPath);
          if (!experiment) {
            console.warn(`Experiment "${experimentPath}" not found, redirecting to home`);
            navigate('/', { replace: true });
            return;
          }

          // Load experiment details
          const experimentData = await api.getExperiment(experimentPath);
          actions.setCurrentExperiment(experimentData);
        } else {
          // Try to find experiment in already loaded tree
          const experiment = findExperimentInTree(state.experimentsTree, experimentPath);
          if (!experiment) {
            console.warn(`Experiment "${experimentPath}" not found, redirecting to home`);
            navigate('/', { replace: true });
            return;
          }

          // Only load experiment details if it's not the current one
          if (!state.currentExperiment || 
              state.currentExperiment.name !== experiment.experiment_data.name) {
            const experimentData = await api.getExperiment(experimentPath);
            actions.setCurrentExperiment(experimentData);
          }
        }
      } catch (error) {
        console.error('Error loading experiment:', error);
        actions.setError(`Failed to load experiment: ${experimentPath}`);
      } finally {
        actions.setLoading(false);
      }
    };

    loadExperiment();
  }, [experimentPath, navigate]); // Removed state dependencies to prevent loops

  return <AppContent />;
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
  
  return traverse(tree);
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
        const experimentsData = await api.getExperiments();
        actions.setExperimentsTree(experimentsData);
        
        // Also maintain flat list for backward compatibility
        const flatExperiments = api.flattenExperimentTree(experimentsData);
        actions.setExperiments(flatExperiments);

        // Auto-select first experiment and redirect to it
        if (flatExperiments.length > 0 && !state.currentExperiment) {
          const firstExperiment = flatExperiments[0];
          const experimentPath = firstExperiment.path.replace(/^outputs\//, '');
          navigate(`/experiment/${experimentPath}`, { replace: true });
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

  return <AppContent />;
};

// Main App Content Component
const AppContent = () => {
  const { state } = useApp();
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

          {!state.isLoading && !state.error && (
            <VideoGrid />
          )}
        </div>

        <SyncControls />
      </div>
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
