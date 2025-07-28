import { useCallback } from 'react';
import { useApp } from '../context/AppContext';

export const useVideoControls = () => {
  const { state, actions } = useApp();

  const playAllVideos = useCallback(() => {
    // Get videos directly from the DOM instead of relying on context
    const videos = Array.from(document.querySelectorAll('video'));
    console.log('useVideoControls: playAllVideos called, videos:', videos.length);
    
    // First pause ALL videos
    videos.forEach(video => {
      video.pause();
      video.playing = false;
    });

    // Reset scrubber to beginning
    document.dispatchEvent(new CustomEvent('resetScrubber'));

    // Then play only visible videos for performance
    const videoGrid = document.querySelector('#video-grid');
    if (!videoGrid) {
      console.log('useVideoControls: No video grid found');
      return;
    }

    const gridRect = videoGrid.getBoundingClientRect();
    const margin = 200;

    videos.forEach(async (video) => {
      const videoRect = video.getBoundingClientRect();

      // Check if video is in or near the viewport
      const isVisible = (
        videoRect.bottom > (gridRect.top - margin) &&
        videoRect.top < (gridRect.bottom + margin) &&
        videoRect.right > (gridRect.left - margin) &&
        videoRect.left < (gridRect.right + margin)
      );

      if (isVisible) {
        console.log('useVideoControls: Loading and playing visible video');
        
        // Load video source if not already loaded (for lazy loading support)
        if (!video.src && video.loadVideoSource) {
          try {
            await video.loadVideoSource();
          } catch (error) {
            console.warn('Failed to load video source for play all:', error);
            return;
          }
        }
        
        video.currentTime = 0;
        video.play().catch(err => {
          console.warn('Failed to play video in play all:', err);
        });
        video.playing = true;
      }
    });
  }, []);

  const pauseAllVideos = useCallback(() => {
    const videos = Array.from(document.querySelectorAll('video'));
    console.log('useVideoControls: pauseAllVideos called, videos:', videos.length);
    videos.forEach(video => {
      video.pause();
      video.playing = false;
    });
  }, []);

  const muteAllVideos = useCallback(() => {
    const videos = Array.from(document.querySelectorAll('video'));
    const anyUnmuted = videos.some(video => !video.muted);
    videos.forEach(video => {
      video.muted = anyUnmuted;
    });
    return anyUnmuted;
  }, []);

  const scrubAllVideos = useCallback(async (percentage) => {
    if (state.videoDuration === 0) return;

    actions.setScrubbingActive(true);
    const targetTime = (percentage / 100) * state.videoDuration;

    // Get videos directly from DOM
    const videos = Array.from(document.querySelectorAll('video'));
    
    // Load videos that aren't already loaded before scrubbing
    const loadPromises = videos.map(async (video) => {
      if (!video.src && video.loadVideoSource) {
        try {
          await video.loadVideoSource();
        } catch (error) {
          console.warn('Failed to load video source for scrubbing:', error);
        }
      }
    });
    
    // Wait for all videos to load, then scrub
    await Promise.all(loadPromises);
    
    videos.forEach(video => {
      if (video.duration && video.src) {
        video.currentTime = targetTime;
      }
    });

    // Clear the scrubbing flag after a short delay
    setTimeout(() => {
      actions.setScrubbingActive(false);
    }, 100);

    return targetTime;
  }, [state.videoDuration, actions]);

  return {
    playAllVideos,
    pauseAllVideos,
    muteAllVideos,
    scrubAllVideos
  };
};
