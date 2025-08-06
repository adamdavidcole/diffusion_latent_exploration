import { useCallback } from 'react';
import { useApp } from '../context/AppContext';

export const useVideoControls = () => {
  const { state, actions } = useApp();
  const { currentExperiment } = state;
  const videoDuration = currentExperiment && currentExperiment.duration_seconds || 0;

  console.log("videoDuration", videoDuration)


  const playAllVideos = useCallback(async (onLoadingChange) => {
    console.log('Play all videos called');
    
    // Notify about loading start
    onLoadingChange && onLoadingChange(true);
    
    // First, send event to all VideoCell components to force video mode
    console.log('Dispatching forceVideoMode event');
    document.dispatchEvent(new CustomEvent('forceVideoMode'));
    
    // Wait a bit for React components to re-render with video elements
    await new Promise(resolve => setTimeout(resolve, 100));
    
    // Now find all video elements that were created
    const videos = Array.from(document.querySelectorAll('video'));
    console.log('Found video elements after force mode:', videos.length);
    
    // Get the video grid for visibility checking
    const videoGrid = document.querySelector('#video-grid');
    if (!videoGrid) {
      onLoadingChange && onLoadingChange(false);
      return;
    }

    const gridRect = videoGrid.getBoundingClientRect();
    const margin = 200;

    // Filter to visible videos only
    const visibleVideos = videos.filter(video => {
      const rect = video.getBoundingClientRect();
      return (
        rect.bottom > (gridRect.top - margin) &&
        rect.top < (gridRect.bottom + margin) &&
        rect.right > (gridRect.left - margin) &&
        rect.left < (gridRect.right + margin)
      );
    });

    console.log('Visible videos for play all:', visibleVideos.length);

    // Reset all videos first
    visibleVideos.forEach(video => {
      video.pause();
      video.currentTime = 0;
    });
    
    document.dispatchEvent(new CustomEvent('resetScrubber'));

    // Wait for autoplay to settle, then play all simultaneously
    await new Promise(resolve => setTimeout(resolve, 200));
    
    // Now play all loaded videos simultaneously
    visibleVideos.forEach(video => {
      if (video && video.duration && !isNaN(video.duration)) {
        video.currentTime = 0;
        video.play().catch(err => {
          console.warn('Play failed:', err);
        });
      }
    });

    // Notify about loading end
    onLoadingChange && onLoadingChange(false);
  }, []);

  const pauseAllVideos = useCallback(() => {
    const videos = Array.from(document.querySelectorAll('video'));
    console.log('Pause all videos called, found:', videos.length);
    videos.forEach(video => {
      video.pause();
      video.currentTime = 0;
    });
    
    // Send event to exit force mode so cells can return to thumbnails
    console.log('Dispatching exitForceMode event');
    document.dispatchEvent(new CustomEvent('exitForceMode'));
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
    console.log('Scrub all videos called with percentage:', percentage);
    
    if (videoDuration === 0) {
      console.log('No video duration set, skipping scrub');
      return;
    }

    // Don't set scrubbing active to avoid re-renders
    // actions.setScrubbingActive(true);
    const targetTime = (percentage / 100) * videoDuration;

    // Get all videos and find visible ones
    const videos = Array.from(document.querySelectorAll('video'));
    const videoGrid = document.querySelector('#video-grid');
    if (!videoGrid) {
      return;
    }

    const gridRect = videoGrid.getBoundingClientRect();
    const margin = 200;

    const visibleVideos = videos.filter(video => {
      const rect = video.getBoundingClientRect();
      return (
        rect.bottom > (gridRect.top - margin) &&
        rect.top < (gridRect.bottom + margin) &&
        rect.right > (gridRect.left - margin) &&
        rect.left < (gridRect.right + margin)
      );
    });

    console.log('Scrubbing visible videos:', visibleVideos.length, 'to time:', targetTime);

    // Load all visible videos that aren't already loaded
    const loadPromises = visibleVideos.map(async (video) => {
      // If video doesn't have duration metadata, try to load it
      if (video.loadVideoSource && video.readyState < 1) { // Less than HAVE_METADATA
        try {
          console.log('Loading video for scrubbing:', video.src);
          await video.loadVideoSource();
        } catch (error) {
          console.warn('Failed to load video for scrubbing:', error);
          return null;
        }
      }
      return video;
    });

    // Wait for all videos to load, then scrub
    const loadedVideos = await Promise.all(loadPromises);
    
    // Now scrub all loaded videos
    loadedVideos.forEach(video => {
      if (video && video.duration && !isNaN(video.duration)) {
        const clampedTime = Math.min(Math.max(targetTime, 0), video.duration);
        console.log(`Scrubbing video to ${clampedTime}s (duration: ${video.duration}s)`);
        video.currentTime = clampedTime;
        video.pause(); // Make sure video is paused at the scrubbed position
      }
    });

    // Only set duration if it's not already set to avoid re-renders
    // if (videoDuration === 0) {
    //   const firstVideoWithDuration = loadedVideos.find(v => v && v.duration && !isNaN(v.duration));
    //   if (firstVideoWithDuration) {
    //     console.log('Setting video duration from first video:', firstVideoWithDuration.duration);
    //     // actions.setVideoDuration(firstVideoWithDuration.duration);
    //   }
    // }

    // Don't clear scrubbing flag since we're not setting it
    // setTimeout(() => {
    //   actions.setScrubbingActive(false);
    // }, 50);

    return targetTime;
  }, [videoDuration, actions]);

  return {
    playAllVideos,
    pauseAllVideos,
    muteAllVideos,
    scrubAllVideos
  };
};
