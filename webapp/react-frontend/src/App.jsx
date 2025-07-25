import React, { useEffect } from 'react';
import { AppProvider, useApp } from './context/AppContext';
import { useVideoCache } from './hooks/useVideoCache';
import Sidebar from './components/Sidebar/Sidebar';
import VideoGrid from './components/VideoGrid/VideoGrid';
import SyncControls from './components/Controls/SyncControls';
import './styles.css';

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

// Root App Component with Provider
function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}

export default App;
