import React, { useEffect, useState } from 'react';
import { getVariationTextFromPromptKey } from '../../utils/variationText';
import './LatentVideoLightbox.css';

const LatentVideoLightbox = ({
  isOpen,
  onClose,
  onNavigate,
  currentPosition, // { rowIndex, stepIndex }
  gridData,
  currentLatentVideos,
  currentExperiment,
  selectedAttentionToken,
  setSelectedAttentionToken,
  availableTokens,
  viewMode
}) => {
  const [currentVideoSrc, setCurrentVideoSrc] = useState('');

  // Get current cell data
  const getCurrentCellData = () => {
    if (!gridData || !currentPosition) return null;
    
    const row = gridData.rows[currentPosition.rowIndex];
    const step = row?.steps[currentPosition.stepIndex];
    
    if (!step) return null;

    // Get video path based on view mode and selected attention token
    let videoPath = step.videoPath;
    
    // If we have attention videos and a token is selected, use attention video
    if (selectedAttentionToken && step.attentionVideos && step.attentionVideos[selectedAttentionToken]) {
      videoPath = step.attentionVideos[selectedAttentionToken];
    }

    return {
      step,
      videoPath,
      promptText: getVariationTextFromPromptKey(row.id, currentExperiment)
    };
  };

  const cellData = getCurrentCellData();

  // Update video source when position or attention token changes
  useEffect(() => {
    if (!cellData?.videoPath) {
      setCurrentVideoSrc('');
      return;
    }

    setCurrentVideoSrc(`/api/video/${cellData.videoPath}`);
  }, [cellData?.videoPath]);

  // Navigation functions
  const navigateLeft = () => {
    if (!gridData || !currentPosition) return;
    
    const newStepIndex = currentPosition.stepIndex > 0
      ? currentPosition.stepIndex - 1
      : gridData.columnHeaders.length - 1; // Wrap to last column
    
    return { ...currentPosition, stepIndex: newStepIndex };
  };

  const navigateRight = () => {
    if (!gridData || !currentPosition) return;
    
    const newStepIndex = currentPosition.stepIndex < gridData.columnHeaders.length - 1
      ? currentPosition.stepIndex + 1
      : 0; // Wrap to first column
    
    return { ...currentPosition, stepIndex: newStepIndex };
  };

  const navigateUp = () => {
    if (!gridData || !currentPosition) return;
    
    const newRowIndex = currentPosition.rowIndex > 0
      ? currentPosition.rowIndex - 1
      : gridData.rows.length - 1; // Wrap to last row
    
    return { ...currentPosition, rowIndex: newRowIndex };
  };

  const navigateDown = () => {
    if (!gridData || !currentPosition) return;
    
    const newRowIndex = currentPosition.rowIndex < gridData.rows.length - 1
      ? currentPosition.rowIndex + 1
      : 0; // Wrap to first row
    
    return { ...currentPosition, rowIndex: newRowIndex };
  };

  // Handle keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (event) => {
      let newPosition = null;

      switch (event.key) {
        case 'Escape':
          onClose();
          break;
        case 'ArrowLeft':
          event.preventDefault();
          newPosition = navigateLeft();
          break;
        case 'ArrowRight':
          event.preventDefault();
          newPosition = navigateRight();
          break;
        case 'ArrowUp':
          event.preventDefault();
          newPosition = navigateUp();
          break;
        case 'ArrowDown':
          event.preventDefault();
          newPosition = navigateDown();
          break;
        default:
          return;
      }

      if (newPosition && onNavigate) {
        onNavigate(newPosition);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, currentPosition, gridData]);

  // Handle click outside to close
  const handleOverlayClick = (event) => {
    if (event.target === event.currentTarget) {
      onClose();
    }
  };

  const handleAttentionTokenChange = (newToken) => {
    setSelectedAttentionToken(newToken || null);
  };

  if (!isOpen || !cellData) {
    return null;
  }

  return (
    <div className="latent-video-lightbox-overlay" onClick={handleOverlayClick}>
      <div className="latent-video-lightbox">
        {/* Close button */}
        <button className="lightbox-close-btn" onClick={onClose}>
          ×
        </button>

        {/* Navigation arrows */}
        <button className="lightbox-nav-btn nav-left" onClick={() => onNavigate?.(navigateLeft())}>
          ‹
        </button>
        <button className="lightbox-nav-btn nav-right" onClick={() => onNavigate?.(navigateRight())}>
          ›
        </button>
        <button className="lightbox-nav-btn nav-up" onClick={() => onNavigate?.(navigateUp())}>
          ‹
        </button>
        <button className="lightbox-nav-btn nav-down" onClick={() => onNavigate?.(navigateDown())}>
          ›
        </button>

        {/* Video container */}
        <div className="lightbox-video-container">
          {currentVideoSrc && (
            <video
              src={currentVideoSrc}
              controls
              autoPlay
              loop
              className="lightbox-video"
            />
          )}
        </div>

        {/* Metadata and controls */}
        <div className="lightbox-metadata">
          <div className="metadata-row">
            <span className="metadata-label">Prompt:</span>
            <span className="metadata-value" title={cellData.promptText}>
              {cellData.promptText.length > 60 
                ? `${cellData.promptText.substring(0, 60)}...` 
                : cellData.promptText}
            </span>
          </div>

          <div className="metadata-row">
            <span className="metadata-label">Step:</span>
            <span className="metadata-value">{cellData.step.stepNumber}</span>
          </div>

          {/* Attention token selector */}
          {availableTokens && availableTokens.length > 0 && (
            <div className="metadata-row">
              <span className="metadata-label">Attention Token:</span>
              <select 
                value={selectedAttentionToken || ''} 
                onChange={(e) => handleAttentionTokenChange(e.target.value)}
                className="attention-token-select"
              >
                <option value="">Latent Video</option>
                {availableTokens.map(token => (
                  <option key={token} value={token}>{token}</option>
                ))}
              </select>
            </div>
          )}

          <div className="metadata-row">
            <span className="metadata-label">Position:</span>
            <span className="metadata-value">
              Row {currentPosition.rowIndex + 1} of {gridData.rows.length}, 
              Step {currentPosition.stepIndex + 1} of {gridData.columnHeaders.length}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LatentVideoLightbox;