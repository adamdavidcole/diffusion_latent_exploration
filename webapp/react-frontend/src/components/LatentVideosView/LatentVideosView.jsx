import React, { useEffect, useState, useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import LatentVideoCell from './LatentVideoCell';
import LatentVideoLightbox from './LatentVideoLightbox';
import { getVariationTextFromPromptKey } from '../../utils/variationText';
import './LatentVideosView.css';

const LatentVideosView = ({ experimentPath }) => {
  const { state, actions } = useApp();
  const { currentLatentVideos, latentVideosLoading, latentVideosError, currentExperiment } = state;

  // View mode state
  const [viewMode, setViewMode] = useState('across-prompts'); // 'across-prompts' or 'by-prompt'
  const [selectedPrompt, setSelectedPrompt] = useState('prompt_000');
  const [selectedSeed, setSelectedSeed] = useState('vid_001');
  const [cellSize, setCellSize] = useState(120); // Local cell size state
  const [selectedAttentionToken, setSelectedAttentionToken] = useState(null); // null = 'None' (latent videos)
  const [tooltip, setTooltip] = useState({ show: false, text: '', x: 0, y: 0 });
  const [lightbox, setLightbox] = useState({ isOpen: false, position: null });

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
        } else if (latentVideosData?.attention_videos) {
          // If no latent videos but we have attention videos, use those for structure
          const firstPrompt = Object.keys(latentVideosData.attention_videos)[0];
          if (firstPrompt) {
            setSelectedPrompt(firstPrompt);
            const firstSeed = Object.keys(latentVideosData.attention_videos[firstPrompt])[0];
            if (firstSeed) {
              setSelectedSeed(firstSeed);
            }
          }
        }

        // Auto-select first attention token if no latent videos available
        if (!latentVideosData?.has_latent_videos && latentVideosData?.attention_videos) {
          const firstPrompt = Object.keys(latentVideosData.attention_videos)[0];
          if (firstPrompt) {
            const firstSeed = Object.keys(latentVideosData.attention_videos[firstPrompt])[0];
            if (firstSeed) {
              const tokens = Object.keys(latentVideosData.attention_videos[firstPrompt][firstSeed]);
              if (tokens.length > 0) {
                setSelectedAttentionToken(tokens[0]);
              }
            }
          }
        } else {
          // Reset attention token selection when loading new data with latent videos
          setSelectedAttentionToken(null);
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

  // Helper function to get display text for prompts with truncation
  const getPromptDisplayText = (promptId, maxLength = 40) => {
    const fullText = getVariationTextFromPromptKey(promptId, currentExperiment);

    if (fullText.length <= maxLength) {
      return { display: fullText, full: fullText };
    }

    // Truncate and add ellipsis
    const truncated = fullText.substring(0, maxLength - 3) + '...';
    return { display: truncated, full: fullText };
  };

  // Helper function to format video_id labels (e.g., p000_b001_s000 -> readable label)
  const getVideoIdLabel = (videoId, metadata) => {
    // Check if it's old format (vid001)
    if (videoId.startsWith('vid')) {
      return videoId.replace('vid', 'Seed ').replace(/^Seed 0+/, 'Seed ');
    }

    // New format: p000_b001_s000
    if (videoId.startsWith('p') && videoId.includes('_b') && videoId.includes('_s')) {
      // Try to get metadata for this video
      const videoMetadata = metadata?.[videoId];
      
      if (videoMetadata?.bending_metadata) {
        const bending = videoMetadata.bending_metadata;
        const parts = [];
        
        // Operation and value
        if (bending.operation === 'scale') {
          parts.push(`Scale ${bending.value}×`);
        } else if (bending.operation === 'add') {
          parts.push(`Add ${bending.value}`);
        } else if (bending.operation === 'set') {
          parts.push(`Set ${bending.value}`);
        }
        
        // Timesteps
        if (bending.timestep_spec) {
          parts.push(`T:${bending.timestep_spec}`);
        }
        
        // Layers
        if (bending.layer_spec && bending.layer_spec !== 'ALL') {
          parts.push(`L:${bending.layer_spec}`);
        } else if (bending.layer_spec === 'ALL') {
          parts.push('L:ALL');
        }
        
        // Token
        if (bending.target_token && bending.target_token !== 'ALL') {
          parts.push(`"${bending.target_token}"`);
        }
        
        return parts.join(' | ');
      } else if (videoId.includes('_b000_')) {
        // Baseline
        return 'Baseline (No Bending)';
      } else {
        // Fallback: parse the ID
        try {
          const bendingPart = videoId.split('_b')[1].split('_')[0];
          return `Bending ${bendingPart}`;
        } catch {
          return videoId;
        }
      }
    }

    return videoId;
  };

  // Organize data based on view mode
  const gridData = useMemo(() => {
    // Check if we have ANY data to display (latent videos or attention videos)
    const hasLatentVideos = currentLatentVideos?.latent_videos;
    const hasAttentionVideos = currentLatentVideos?.attention_videos;

    if (!hasLatentVideos && !hasAttentionVideos) return null;

    // Get video metadata map for labels
    const videoMetadataMap = currentLatentVideos?.video_metadata_map || {};

    // If we only have attention videos, use them to determine structure
    const dataSource = hasLatentVideos ? currentLatentVideos.latent_videos : currentLatentVideos.attention_videos;
    const attentionVideos = currentLatentVideos.attention_videos;

    let allPrompts, allSeeds, allSteps;

    if (hasLatentVideos) {
      // Use latent videos structure
      allPrompts = Object.keys(dataSource).sort();
      allSeeds = allPrompts.length > 0 ? Object.keys(dataSource[allPrompts[0]]).sort() : [];

      if (allPrompts.length > 0 && allSeeds.length > 0) {
        const firstVideo = dataSource[allPrompts[0]][allSeeds[0]];
        allSteps = Object.keys(firstVideo).sort();
      } else {
        allSteps = [];
      }
    } else {
      // Use attention videos structure (first token)
      allPrompts = Object.keys(dataSource).sort();
      allSeeds = allPrompts.length > 0 ? Object.keys(dataSource[allPrompts[0]]).sort() : [];

      if (allPrompts.length > 0 && allSeeds.length > 0) {
        // Get first token's data to determine steps
        const firstTokenData = dataSource[allPrompts[0]][allSeeds[0]];
        const firstToken = Object.keys(firstTokenData)[0];
        allSteps = Object.keys(firstTokenData[firstToken]).sort();
      } else {
        allSteps = [];
      }
    }

    // Helper function to get video/image paths with attention video fallback
    const getVideoData = (promptId, seedId, stepId) => {
      let videoPath = null;
      let imagePath = null;

      // Try attention videos first if a token is selected
      if (selectedAttentionToken && attentionVideos) {
        const attentionData = attentionVideos[promptId]?.[seedId]?.[selectedAttentionToken]?.[stepId];
        if (attentionData) {
          videoPath = attentionData.video_path;
          imagePath = attentionData.image_path;
        }
      }

      // Fallback to latent videos if no attention video found
      if (!videoPath && hasLatentVideos) {
        const latentData = dataSource[promptId]?.[seedId]?.[stepId];
        if (latentData) {
          videoPath = latentData.video_path;
          imagePath = latentData.image_path;
        }
      }

      return { videoPath, imagePath };
    };

    if (viewMode === 'across-prompts') {
      // Each row is a different prompt, all using the same seed
      const rows = allPrompts.map(promptId => {
        const { display, full } = getPromptDisplayText(promptId);
        return {
          id: promptId,
          label: display,
          fullLabel: full,
          steps: allSteps.map(stepId => {
            const { videoPath, imagePath } = getVideoData(promptId, selectedSeed, stepId);
            return {
              stepId,
              stepNumber: parseInt(stepId.replace('step_', '')),
              videoPath,
              imagePath,
            };
          })
        };
      });

      // Get readable label for selected seed
      const selectedSeedLabel = getVideoIdLabel(selectedSeed, videoMetadataMap);

      return {
        rows,
        columnHeaders: allSteps.map(stepId => `Step ${stepId.replace('step_', '')}`),
        rowType: 'prompt',
        selectedItem: selectedSeedLabel
      };
    } else {
      // Each row is a different seed, all using the same prompt
      const rows = allSeeds.map(seedId => {
        const readableLabel = getVideoIdLabel(seedId, videoMetadataMap);
        return {
          id: seedId,
          label: readableLabel,
          fullLabel: readableLabel, // Already formatted
          steps: allSteps.map(stepId => {
            const { videoPath, imagePath } = getVideoData(selectedPrompt, seedId, stepId);
            return {
              stepId,
              stepNumber: parseInt(stepId.replace('step_', '')),
              videoPath,
              imagePath,
            };
          })
        };
      });

      const { display: selectedPromptDisplay } = getPromptDisplayText(selectedPrompt);
      return {
        rows,
        columnHeaders: allSteps.map(stepId => `Step ${stepId.replace('step_', '')}`),
        rowType: 'seed',
        selectedItem: selectedPromptDisplay
      };
    }
  }, [currentLatentVideos, viewMode, selectedPrompt, selectedSeed, selectedAttentionToken]);

  // Get available options for dropdowns
  const availableOptions = useMemo(() => {
    // Check if we have ANY data to display
    const hasLatentVideos = currentLatentVideos?.latent_videos;
    const hasAttentionVideos = currentLatentVideos?.attention_videos;

    if (!hasLatentVideos && !hasAttentionVideos) {
      return { prompts: [], seeds: [], tokens: [] };
    }

    // Use whichever data source is available
    const dataSource = hasLatentVideos ? currentLatentVideos.latent_videos : currentLatentVideos.attention_videos;

    let prompts, seeds;

    if (hasLatentVideos) {
      prompts = Object.keys(dataSource).sort();
      seeds = prompts.length > 0 ? Object.keys(dataSource[prompts[0]]).sort() : [];
    } else {
      // For attention videos, need to navigate through token structure
      prompts = Object.keys(dataSource).sort();
      if (prompts.length > 0) {
        seeds = Object.keys(dataSource[prompts[0]]).sort();
      } else {
        seeds = [];
      }
    }

    // Get available attention tokens
    const attentionVideos = currentLatentVideos.attention_videos;
    const tokens = [];

    if (attentionVideos) {
      const tokenSet = new Set();
      Object.values(attentionVideos).forEach(promptData => {
        Object.values(promptData).forEach(videoData => {
          Object.keys(videoData).forEach(token => {
            tokenSet.add(token);
          });
        });
      });
      tokens.push(...Array.from(tokenSet).sort());
    }

    return { prompts, seeds, tokens };
  }, [currentLatentVideos]);

  const handleCellClick = (cellData, rowIndex, stepIndex) => {
    // Open lightbox at the clicked position
    setLightbox({
      isOpen: true,
      position: { rowIndex, stepIndex }
    });
  };

  const handleTooltipShow = (event, text) => {
    const rect = event.target.getBoundingClientRect();
    setTooltip({
      show: true,
      text: text,
      x: rect.right + 10,
      y: rect.top + rect.height / 2
    });
  };

  const handleTooltipHide = () => {
    setTooltip({ show: false, text: '', x: 0, y: 0 });
  };

  const handleLightboxClose = () => {
    setLightbox({ isOpen: false, position: null });
  };

  const handleLightboxNavigate = (newPosition) => {
    setLightbox({ isOpen: true, position: newPosition });
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
  if (!currentLatentVideos?.has_latent_videos && !currentLatentVideos?.has_attention_videos) {
    return (
      <div className="latent-videos-view">
        <div className="empty-state">
          <h3>No latent or attention videos available</h3>
          <p>This experiment doesn't contain latent video or attention video data.</p>
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
        {/* Info banner if latent videos missing but attention videos available */}
        {!currentLatentVideos?.has_latent_videos && currentLatentVideos?.has_attention_videos && (
          <div style={{
            background: '#fff3cd',
            border: '1px solid #ffc107',
            borderRadius: '4px',
            padding: '8px 12px',
            marginBottom: '12px',
            fontSize: '14px',
            color: '#856404'
          }}>
            ℹ️ Latent videos not decoded. Showing attention videos only. Select a token from the "Attention" dropdown to view.
          </div>
        )}

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
                  {availableOptions.prompts.map(promptId => {
                    const { display, full } = getPromptDisplayText(promptId);
                    return (
                      <option key={promptId} value={promptId} title={full}>
                        {display}
                      </option>
                    );
                  })}
                </select>
              </>
            )}
          </div>

          {/* Attention token dropdown */}
          {availableOptions.tokens.length > 0 && (
            <div className="control-group">
              <label htmlFor="attention-select">
                Attention:
                {!currentLatentVideos?.has_latent_videos && (
                  <span style={{ color: '#ffc107', marginLeft: '4px' }}>*</span>
                )}
              </label>
              <select
                id="attention-select"
                value={selectedAttentionToken || ''}
                onChange={(e) => setSelectedAttentionToken(e.target.value || null)}
                style={!currentLatentVideos?.has_latent_videos ? {
                  borderColor: '#ffc107',
                  borderWidth: '2px'
                } : {}}
              >
                {currentLatentVideos?.has_latent_videos && (
                  <option value="">None (Latent Videos)</option>
                )}
                {!currentLatentVideos?.has_latent_videos && (
                  <option value="" disabled>Select a token...</option>
                )}
                {availableOptions.tokens.map(token => {
                  // Token is already the clean text (without token_ prefix)
                  // Truncate long token names for display
                  const displayText = token.length > 20 ? `${token.substring(0, 20)}...` : token;
                  return (
                    <option key={token} value={token} title={token}>
                      {displayText}
                    </option>
                  );
                })}
              </select>
            </div>
          )}

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
          {gridData.rows.map((row, rowIndex) => (
            <div key={row.id} className="grid-row">
              <div
                className="row-label"
                onMouseEnter={(e) => handleTooltipShow(e, row.fullLabel)}
                onMouseLeave={handleTooltipHide}
              >
                {row.label}
              </div>
              {row.steps.map((step, stepIndex) => (
                <div key={step.stepId} className="grid-cell">
                  <LatentVideoCell
                    videoPath={step.videoPath}
                    imagePath={step.imagePath}
                    videoSize={cellSize}
                    stepNumber={step.stepNumber}
                    onClick={() => handleCellClick(step, rowIndex, stepIndex)}
                  />
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Custom tooltip */}
      {tooltip.show && (
        <div
          className="custom-tooltip"
          style={{
            position: 'fixed',
            left: `${tooltip.x}px`,
            top: `${tooltip.y}px`,
            transform: 'translateY(-50%)',
            zIndex: 10000,
            pointerEvents: 'none'
          }}
        >
          {tooltip.text}
        </div>
      )}

      {/* Lightbox */}
      <LatentVideoLightbox
        isOpen={lightbox.isOpen}
        onClose={handleLightboxClose}
        onNavigate={handleLightboxNavigate}
        currentPosition={lightbox.position}
        gridData={gridData}
        currentLatentVideos={currentLatentVideos}
        currentExperiment={currentExperiment}
        selectedAttentionToken={selectedAttentionToken}
        setSelectedAttentionToken={setSelectedAttentionToken}
        availableTokens={availableOptions.tokens}
        viewMode={viewMode}
      />
    </div>
  );
};

export default LatentVideosView;
