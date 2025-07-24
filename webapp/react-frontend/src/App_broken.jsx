import React, { useEffect } from 'react';
im  return (
    <div className="app-container">
      <Sidebar />
      
      <div className="main-content">
        <div className="video-grid-wrapper">{ AppProvider, useApp } from './context/AppContext';
import { useVideoCache } from './hooks/useVideoCache';
import { useVideoControls } from './hooks/useVideoControls';
import Sidebar from './components/Sidebar/Sidebar';
import VideoGrid from './components/VideoGrid/VideoGrid';
import SyncControls from './components/Controls/SyncControls';
import './styles.css';

// Main App Content Component
const AppContent = () => {
  const { state } = useApp();
  const { clearCache } = useVideoCache();
  const { updateVideoRefs } = useVideoControls();

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

  // Update video refs when videos change
  useEffect(() => {
    const videos = Array.from(document.querySelectorAll('video'));
    updateVideoRefs(videos);
  }, [state.currentExperiment, updateVideoRefs]);

  // Clean up cache when component unmounts
  useEffect(() => {
    return () => {
      clearCache();
    };
  }, [clearCache]);

  return (
    <div className="app">
      <Sidebar />
      
      <div className="main-content">
        <div className="video-grid-wrapper">
          {state.isLoading && (
            <div className="loading">
              <div className="loading-spinner"></div>
              <p>Loading...</p>
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
