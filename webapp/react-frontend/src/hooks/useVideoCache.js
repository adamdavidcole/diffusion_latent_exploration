import { useRef, useCallback, useEffect } from 'react';
import { api } from '../services/api';

export const useVideoCache = () => {
  const loadingQueue = useRef(new Set());
  const cachedVideoSources = useRef(new Map());
  const loadedVideos = useRef(new Set());
  const maxConcurrentLoads = 4;

  const clearCache = useCallback(() => {
    // Clean up blob URLs to prevent memory leaks
    for (const [, blobUrl] of cachedVideoSources.current) {
      URL.revokeObjectURL(blobUrl);
    }
    cachedVideoSources.current.clear();
    loadedVideos.current.clear();
    console.log('Video cache cleared');
  }, []);

  const loadVideo = useCallback(async (videoElement, videoPath) => {
    if (videoElement.hasAttribute('data-loaded') || loadingQueue.current.has(videoElement)) {
      return;
    }

    // Check if we have a cached version
    if (cachedVideoSources.current.has(videoPath)) {
      const cachedUrl = cachedVideoSources.current.get(videoPath);
      videoElement.src = cachedUrl;
      videoElement.setAttribute('data-loaded', 'true');
      loadedVideos.current.add(videoElement);
      return;
    }

    if (loadingQueue.current.size >= maxConcurrentLoads) {
      // Add to waiting queue
      setTimeout(() => loadVideo(videoElement, videoPath), 200);
      return;
    }

    loadingQueue.current.add(videoElement);

    try {
      // Fetch and cache the video
      const blob = await api.fetchVideoBlob(videoPath);
      const blobUrl = URL.createObjectURL(blob);

      // Cache the blob URL
      cachedVideoSources.current.set(videoPath, blobUrl);

      videoElement.src = blobUrl;
      videoElement.preload = 'metadata';

      // Wait for video to be loadable
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('Video load timeout')), 10000);

        const handleLoadedMetadata = () => {
          clearTimeout(timeout);
          resolve();
        };

        const handleError = () => {
          clearTimeout(timeout);
          reject(new Error('Video load error'));
        };

        videoElement.addEventListener('loadedmetadata', handleLoadedMetadata, { once: true });
        videoElement.addEventListener('error', handleError, { once: true });

        videoElement.load();
      });

      videoElement.setAttribute('data-loaded', 'true');
      loadedVideos.current.add(videoElement);
    } catch (error) {
      console.warn('Failed to load video:', error);
      videoElement.setAttribute('data-load-error', 'true');
    } finally {
      loadingQueue.current.delete(videoElement);
    }
  }, []);

  const getCacheStats = useCallback(() => {
    return {
      cachedVideos: cachedVideoSources.current.size,
      loadedVideos: loadedVideos.current.size,
      loading: loadingQueue.current.size
    };
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearCache();
    };
  }, [clearCache]);

  return {
    loadVideo,
    clearCache,
    getCacheStats,
    cachedVideoSources: cachedVideoSources.current,
    loadedVideos: loadedVideos.current
  };
};
