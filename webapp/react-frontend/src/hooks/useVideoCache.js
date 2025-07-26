import { useRef, useCallback, useEffect } from 'react';
import { api } from '../services/api';

export const useVideoCache = () => {
  const loadingQueue = useRef(new Set());
  const cachedVideoSources = useRef(new Map());
  const loadedVideos = useRef(new Set());
  const pendingRequests = useRef(new Map()); // Track AbortControllers for cleanup
  const waitingQueue = useRef([]); // Queue for videos waiting to load
  const maxConcurrentLoads = 3; // Reduced from 4 to be more conservative
  const maxCacheSize = 50; // Limit cached videos to prevent memory issues

  const clearCache = useCallback(() => {
    // Cancel all pending requests
    for (const [, abortController] of pendingRequests.current) {
      abortController.abort();
    }
    pendingRequests.current.clear();
    
    // Clear waiting queue
    waitingQueue.current = [];
    
    // Clean up blob URLs to prevent memory leaks
    for (const [, blobUrl] of cachedVideoSources.current) {
      URL.revokeObjectURL(blobUrl);
    }
    cachedVideoSources.current.clear();
    loadedVideos.current.clear();
    loadingQueue.current.clear();
    console.log('Video cache cleared - all requests cancelled');
  }, []);

  const cancelInflightRequests = useCallback(() => {
    // Cancel all pending requests but keep cache intact
    let cancelledCount = 0;
    for (const [, abortController] of pendingRequests.current) {
      abortController.abort();
      cancelledCount++;
    }
    pendingRequests.current.clear();
    
    // Clear waiting queue
    const waitingCount = waitingQueue.current.length;
    waitingQueue.current = [];
    
    // Clear loading queue
    loadingQueue.current.clear();
    
    console.log(`Cancelled ${cancelledCount} inflight requests and ${waitingCount} queued requests`);
  }, []);

  const evictOldestCache = useCallback(() => {
    if (cachedVideoSources.current.size >= maxCacheSize) {
      // Remove oldest cached video
      const firstKey = cachedVideoSources.current.keys().next().value;
      const blobUrl = cachedVideoSources.current.get(firstKey);
      URL.revokeObjectURL(blobUrl);
      cachedVideoSources.current.delete(firstKey);
      console.log('Evicted cached video:', firstKey);
    }
  }, []);

  const processWaitingQueue = useCallback(() => {
    while (waitingQueue.current.length > 0 && loadingQueue.current.size < maxConcurrentLoads) {
      const { videoElement, videoPath } = waitingQueue.current.shift();
      
      // Check if video element is still in DOM (not cleaned up)
      if (videoElement.isConnected) {
        loadVideo(videoElement, videoPath);
      }
    }
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

    // If we're at the concurrent limit, add to waiting queue
    if (loadingQueue.current.size >= maxConcurrentLoads) {
      waitingQueue.current.push({ videoElement, videoPath });
      return;
    }

    // Evict old cache entries if needed
    evictOldestCache();

    loadingQueue.current.add(videoElement);
    const abortController = new AbortController();
    const requestKey = `${videoPath}-${Date.now()}`;
    pendingRequests.current.set(requestKey, abortController);

    try {
      // Fetch and cache the video with cancellation support
      const blob = await api.fetchVideoBlob(videoPath, abortController.signal);
      
      // Check if request was cancelled
      if (abortController.signal.aborted) {
        return;
      }
      
      const blobUrl = URL.createObjectURL(blob);

      // Cache the blob URL
      cachedVideoSources.current.set(videoPath, blobUrl);

      videoElement.src = blobUrl;
      videoElement.preload = 'metadata';

      // Wait for video to be loadable with cancellation support
      await new Promise((resolve, reject) => {
        if (abortController.signal.aborted) {
          reject(new Error('Request cancelled'));
          return;
        }

        const timeout = setTimeout(() => reject(new Error('Video load timeout')), 10000);

        const handleLoadedMetadata = () => {
          clearTimeout(timeout);
          resolve();
        };

        const handleError = () => {
          clearTimeout(timeout);
          reject(new Error('Video load error'));
        };

        const handleAbort = () => {
          clearTimeout(timeout);
          reject(new Error('Request cancelled'));
        };

        videoElement.addEventListener('loadedmetadata', handleLoadedMetadata, { once: true });
        videoElement.addEventListener('error', handleError, { once: true });
        abortController.signal.addEventListener('abort', handleAbort, { once: true });

        videoElement.load();
      });

      videoElement.setAttribute('data-loaded', 'true');
      loadedVideos.current.add(videoElement);
    } catch (error) {
      if (error.name !== 'AbortError' && !abortController.signal.aborted) {
        console.warn('Failed to load video:', error);
        videoElement.setAttribute('data-load-error', 'true');
      }
    } finally {
      loadingQueue.current.delete(videoElement);
      pendingRequests.current.delete(requestKey);
      
      // Process waiting queue
      setTimeout(processWaitingQueue, 100);
    }
  }, [evictOldestCache, processWaitingQueue]);

  const getCacheStats = useCallback(() => {
    return {
      cachedVideos: cachedVideoSources.current.size,
      loadedVideos: loadedVideos.current.size,
      loading: loadingQueue.current.size,
      waiting: waitingQueue.current.length,
      pendingRequests: pendingRequests.current.size
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
    cancelInflightRequests,
    getCacheStats,
    cachedVideoSources: cachedVideoSources.current,
    loadedVideos: loadedVideos.current
  };
};
