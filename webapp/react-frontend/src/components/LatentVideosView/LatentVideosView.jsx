import React, { useEffect, useState, useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import LatentVideoCell from './LatentVideoCell';
import './LatentVideosView.css';

const LatentVideosView = ({ experimentPath }) => {
  const { state, actions } = useApp();
  const { currentLatentVideos, latentVideosLoading, latentVideosError } = state;
  
  // View mode state
  const [viewMode, setViewMode] = useState('across-prompts'); // 'across-prompts' or 'by-prompt'
  const [selectedPrompt, setSelectedPrompt] = useState('prompt_000');
  const [selectedSeed, setSelectedSeed] = useState('vid_001');
  const [cellSize, setCellSize] = useState(120); // Local cell size state

  // Load latent videos data when component mounts or experiment changes
  useEffect(() => {
    const loadLatentVideos = async () => {
      if (!experimentPath) return;

      try {
        actions.setLatentVideosLoading(true);
        actions.clearLatentVideosError();

        const latentVideosData = await api.getExperimentLatentVideos(experimentPath);
        actions.setCurrentLatentVideos(latentVideosData);

        // Auto-select first available prompt and seed
        if (latentVideosData?.latent_videos) {
          const firstPrompt = Object.keys(latentVideosData.latent_videos)[0];
          if (firstPrompt) {
            setSelectedPrompt(firstPrompt);
            const firstSeed = Object.keys(latentVideosData.latent_videos[firstPrompt])[0];
            if (firstSeed) {
              setSelectedSeed(firstSeed);
            }
          }
        }
      } catch (error) {
        console.error('Error loading latent videos:', error);
        actions.setLatentVideosError(error.message);
      } finally {
        actions.setLatentVideosLoading(false);
      }
    };

    loadLatentVideos();
  }, [experimentPath]); // Removed actions from dependency array to prevent infinite loop

  // Organize data based on view mode
  const gridData = useMemo(() => {
    if (!currentLatentVideos?.latent_videos) return null;

    const latentVideos = currentLatentVideos.latent_videos;
    const allPrompts = Object.keys(latentVideos).sort();
    const allSeeds = allPrompts.length > 0 ? Object.keys(latentVideos[allPrompts[0]]).sort() : [];
    
    // Get all step numbers from the first available video
    let allSteps = [];
    if (allPrompts.length > 0 && allSeeds.length > 0) {
      const firstVideo = latentVideos[allPrompts[0]][allSeeds[0]];
      allSteps = Object.keys(firstVideo).sort();
    }

    if (viewMode === 'across-prompts') {
      // Each row is a different prompt, all using the same seed
      const rows = allPrompts.map(promptId => {
        const promptData = latentVideos[promptId];
        const seedData = promptData[selectedSeed] || {};
        
        return {
          id: promptId,
          label: promptId.replace('prompt_', 'Prompt '),
          steps: allSteps.map(stepId => ({
            stepId,
            stepNumber: parseInt(stepId.replace('step_', '')),
            videoPath: seedData[stepId]?.video_path,
            imagePath: seedData[stepId]?.image_path,
          }))
        };
      });

      return {
        rows,
        columnHeaders: allSteps.map(stepId => `Step ${stepId.replace('step_', '')}`),
        rowType: 'prompt',
        selectedItem: selectedSeed.replace('vid_', 'Seed ')
      };
    } else {
      // Each row is a different seed, all using the same prompt
      const promptData = latentVideos[selectedPrompt] || {};
      const rows = allSeeds.map(seedId => {
        const seedData = promptData[seedId] || {};
        
        return {
          id: seedId,
          label: seedId.replace('vid_', 'Seed '),
          steps: allSteps.map(stepId => ({
            stepId,
            stepNumber: parseInt(stepId.replace('step_', '')),
            videoPath: seedData[stepId]?.video_path,
            imagePath: seedData[stepId]?.image_path,
          }))
        };
      });

      return {
        rows,
        columnHeaders: allSteps.map(stepId => `Step ${stepId.replace('step_', '')}`),
        rowType: 'seed',
        selectedItem: selectedPrompt.replace('prompt_', 'Prompt ')
      };
    }
  }, [currentLatentVideos, viewMode, selectedPrompt, selectedSeed]);

  // Get available options for dropdowns
  const availableOptions = useMemo(() => {
    if (!currentLatentVideos?.latent_videos) return { prompts: [], seeds: [] };

    const latentVideos = currentLatentVideos.latent_videos;
    const prompts = Object.keys(latentVideos).sort();
    const seeds = prompts.length > 0 ? Object.keys(latentVideos[prompts[0]]).sort() : [];

    return { prompts, seeds };
  }, [currentLatentVideos]);

  const handleCellClick = (cellData) => {
    // Future: could open lightbox or detailed view
    console.log('Clicked latent video cell:', cellData);
  };

  // Render loading state
  if (latentVideosLoading) {
    return (
      <div className="latent-videos-view">
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Loading latent videos...</p>
        </div>
      </div>
    );
  }

  // Render error state
  if (latentVideosError) {
    return (
      <div className="latent-videos-view">
        <div className="error-message">
          <h3>Error</h3>
          <p>{latentVideosError}</p>
        </div>
      </div>
    );
  }

  // Render no data state
  if (!currentLatentVideos?.has_latent_videos || !currentLatentVideos?.latent_videos) {
    return (
      <div className="latent-videos-view">
        <div className="empty-state">
          <h3>No latent videos available</h3>
          <p>This experiment doesn't contain latent video data.</p>
        </div>
      </div>
    );
  }

  if (!gridData) {
    return (
      <div className="latent-videos-view">
        <div className="empty-state">
          <h3>No video data available</h3>
          <p>Unable to organize the latent video data.</p>
        </div>
      </div>
    );
  }

  return (
    <div 
      className="latent-videos-view"
      style={{
        '--cell-size': `${cellSize}px`
      }}
    >
      {/* Fixed header with controls */}
      <div className="latent-videos-header-fixed">
        <div className="latent-videos-controls">
          {/* View mode toggle */}
          <div className="control-group">
            <label>View Mode:</label>
            <div className="toggle-group">
              <button
                className={`toggle-btn ${viewMode === 'across-prompts' ? 'active' : ''}`}
                onClick={() => setViewMode('across-prompts')}
              >
                Across Prompts
              </button>
              <button
                className={`toggle-btn ${viewMode === 'by-prompt' ? 'active' : ''}`}
                onClick={() => setViewMode('by-prompt')}
              >
                By Prompt
              </button>
            </div>
          </div>

          {/* Selection dropdown */}
          <div className="control-group">
            {viewMode === 'across-prompts' ? (
              <>
                <label htmlFor="seed-select">Seed:</label>
                <select 
                  id="seed-select"
                  value={selectedSeed} 
                  onChange={(e) => setSelectedSeed(e.target.value)}
                >
                  {availableOptions.seeds.map(seedId => (
                    <option key={seedId} value={seedId}>
                      {seedId.replace('vid_', 'Seed ')}
                    </option>
                  ))}
                </select>
              </>
            ) : (
              <>
                <label htmlFor="prompt-select">Prompt:</label>
                <select 
                  id="prompt-select"
                  value={selectedPrompt} 
                  onChange={(e) => setSelectedPrompt(e.target.value)}
                >
                  {availableOptions.prompts.map(promptId => (
                    <option key={promptId} value={promptId}>
                      {promptId.replace('prompt_', 'Prompt ')}
                    </option>
                  ))}
                </select>
              </>
            )}
          </div>

          {/* Cell size slider */}
          <div className="control-group">
            <label htmlFor="cell-size-slider">Size:</label>
            <input
              id="cell-size-slider"
              type="range"
              min="80"
              max="300"
              value={cellSize}
              onChange={(e) => setCellSize(parseInt(e.target.value))}
            />
            <span className="slider-value">{cellSize}px</span>
          </div>

          <div className="view-info">
            {gridData.selectedItem}
          </div>
        </div>
      </div>

      {/* Scrollable grid container */}
      <div className="latent-videos-grid-container">
        {/* Column headers */}
        <div className="grid-headers">
          <div className="row-label-header">
            {viewMode === 'across-prompts' ? 'Prompts' : 'Seeds'}
          </div>
          {gridData.columnHeaders.map(header => (
            <div key={header} className="column-header">
              {header}
            </div>
          ))}
        </div>

        {/* Grid rows */}
        <div className="grid-rows">
          {gridData.rows.map(row => (
            <div key={row.id} className="grid-row">
              <div className="row-label" title={row.label}>
                {row.label}
              </div>
              {row.steps.map(step => (
                <div key={step.stepId} className="grid-cell">
                  <LatentVideoCell
                    videoPath={step.videoPath}
                    imagePath={step.imagePath}
                    videoSize={cellSize}
                    stepNumber={step.stepNumber}
                    onClick={handleCellClick}
                  />
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default LatentVideosView;
