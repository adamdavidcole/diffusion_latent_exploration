import React, { useRef, useEffect, useState, useCallback } from 'react';
import { getVideoUrl, getThumbnailUrl } from '../../services/api';

const LatentVideoCell = ({ 
  videoPath, 
  imagePath, 
  videoSize, 
  stepNumber,
  onClick 
}) => {
  const videoRef = useRef(null);
  const hoverTimeoutRef = useRef(null);
  const playTimeoutRef = useRef(null);
  const [useVideoElement, setUseVideoElement] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);

  // Cleanup timeouts on unmount
  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
      }
      if (playTimeoutRef.current) {
        clearTimeout(playTimeoutRef.current);
      }
    };
  }, []);

  const handleMouseEnter = useCallback(() => {
    // Clear any pending timeouts
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }
    if (playTimeoutRef.current) {
      clearTimeout(playTimeoutRef.current);
      playTimeoutRef.current = null;
    }

    // Only switch to video after a brief delay to avoid quick hover flashes
    playTimeoutRef.current = setTimeout(() => {
      setUseVideoElement(true);
    }, 200); // 200ms delay before starting video
  }, []);

  const handleMouseLeave = useCallback(() => {
    // Clear any pending timeouts
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
    }
    if (playTimeoutRef.current) {
      clearTimeout(playTimeoutRef.current);
      playTimeoutRef.current = null;
    }

    // Switch back to image after a short delay
    hoverTimeoutRef.current = setTimeout(() => {
      setUseVideoElement(false);
      setIsLoaded(false); // Reset loaded state for next time
    }, 100); // Quick switch back to image
  }, []);

  const handleClick = useCallback(() => {
    if (onClick) {
      onClick({ videoPath, imagePath, stepNumber });
    }
  }, [onClick, videoPath, imagePath, stepNumber]);

  const handleLoadedMetadata = useCallback(() => {
    setIsLoaded(true);
  }, []);

  if (!videoPath) {
    return (
      <div
        className="latent-video-cell empty"
        style={{
          width: `${videoSize}px`,
          height: `${Math.round(videoSize * (9/16))}px`, // 16:9 aspect ratio
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'var(--bg-tertiary)',
          border: '1px dashed var(--border)',
          borderRadius: '4px',
          color: 'var(--text-muted)'
        }}
      >
        No Video
      </div>
    );
  }

  const thumbnailUrl = imagePath ? getThumbnailUrl(imagePath) : getThumbnailUrl(videoPath.replace('.mp4', '.jpg'));

  return (
    <div
      className={`latent-video-cell ${isLoaded ? 'loaded' : ''}`}
      onClick={handleClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      style={{
        position: 'relative',
        cursor: 'pointer',
        borderRadius: '4px',
        overflow: 'hidden',
        transition: 'transform 0.2s ease',
      }}
    >
      {useVideoElement ? (
        <video
          ref={videoRef}
          className="latent-video-element"
          style={{
            width: `${videoSize}px`,
            height: `${Math.round(videoSize * (9/16))}px`, // 16:9 aspect ratio
            objectFit: 'cover',
            display: 'block',
          }}
          src={getVideoUrl(videoPath)}
          muted
          loop
          autoPlay
          preload="auto"
          playsInline
          onLoadedMetadata={handleLoadedMetadata}
          poster={thumbnailUrl}
        />
      ) : (
        <img
          className="latent-video-thumbnail"
          style={{
            width: `${videoSize}px`,
            height: `${Math.round(videoSize * (9/16))}px`, // 16:9 aspect ratio
            objectFit: 'cover',
            display: 'block',
          }}
          src={thumbnailUrl}
          alt={`Step ${stepNumber} thumbnail`}
          onLoad={() => setIsLoaded(true)}
        />
      )}
      
      {/* Step number overlay */}
      <div
        className="step-overlay"
        style={{
          position: 'absolute',
          bottom: '4px',
          left: '4px',
          background: 'rgba(0, 0, 0, 0.7)',
          color: 'white',
          padding: '2px 6px',
          borderRadius: '3px',
          fontSize: '12px',
          fontWeight: '500',
        }}
      >
        {stepNumber}
      </div>
    </div>
  );
};

export default React.memo(LatentVideoCell);
