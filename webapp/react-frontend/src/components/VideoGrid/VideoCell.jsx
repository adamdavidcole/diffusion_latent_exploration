import React, { useRef, useEffect, useState, useCallback } from 'react';
import { getVideoUrl, getThumbnailUrl } from '../../services/api';
import { useApp } from '../../context/AppContext';
import { useVideoCache } from '../../hooks/useVideoCache';

const VideoCell = ({ video, videoSize, onVideoLoaded, onMetadataLoaded, onOpenLightbox }) => {
    const videoRef = useRef(null);
    const [isLoaded, setIsLoaded] = useState(false);
    const [isPlaying, setIsPlaying] = useState(false);
    const [isLoadingOnHover, setIsLoadingOnHover] = useState(false);
    const [preloadMode, setPreloadMode] = useState('none'); // Track preload state
    const [showPoster, setShowPoster] = useState(true); // Track poster visibility
    const { state } = useApp();
    // const { loadVideo } = useVideoCache(); // Commented out to preserve thumbnail performance

    // Calculate thumbnail path from video path
    const getThumbnailPath = useCallback((videoPath) => {
        if (!videoPath) return null;
        try {
            // Replace .mp4 extension with .jpg for thumbnail
            const thumbnailPath = videoPath.replace(/\.mp4$/, '.jpg');
            return getThumbnailUrl(thumbnailPath);
        } catch (error) {
            console.warn('Error generating thumbnail path:', error);
            return null;
        }
    }, []);

    // Function to preload video metadata (for programmatic play all)
    const loadVideoMetadata = useCallback(() => {
        const videoElement = videoRef.current;
        if (!videoElement || !video?.video_path) return Promise.resolve();

        // If video already has metadata loaded, resolve immediately
        if (videoElement.readyState >= 1) { // HAVE_METADATA or higher
            return Promise.resolve();
        }

        return new Promise((resolve, reject) => {
            const handleLoadedMetadata = () => {
                videoElement.removeEventListener('loadedmetadata', handleLoadedMetadata);
                videoElement.removeEventListener('error', handleError);
                resolve();
            };

            const handleError = (error) => {
                videoElement.removeEventListener('loadedmetadata', handleLoadedMetadata);
                videoElement.removeEventListener('error', handleError);
                console.warn('Failed to preload video metadata:', error);
                reject(error);
            };

            videoElement.addEventListener('loadedmetadata', handleLoadedMetadata);
            videoElement.addEventListener('error', handleError);

            // Set preload to metadata for basic playback
            setPreloadMode('metadata');
            videoElement.load(); // Trigger loading
        });
    }, [video?.video_path]);

    // Function to fully load video data (for scrubbing)
    const loadVideoSource = useCallback(() => {
        const videoElement = videoRef.current;
        if (!videoElement || !video?.video_path) return Promise.resolve();

        // If video is already fully loaded, resolve immediately
        if (videoElement.readyState >= 3) { // HAVE_FUTURE_DATA or higher
            return Promise.resolve();
        }

        return new Promise((resolve, reject) => {
            const handleCanPlayThrough = () => {
                videoElement.removeEventListener('canplaythrough', handleCanPlayThrough);
                videoElement.removeEventListener('error', handleError);
                console.log('Video fully loaded for scrubbing:', video.video_path);
                resolve();
            };

            const handleError = (error) => {
                videoElement.removeEventListener('canplaythrough', handleCanPlayThrough);
                videoElement.removeEventListener('error', handleError);
                console.warn('Failed to fully load video for scrubbing:', error);
                reject(error);
            };

            videoElement.addEventListener('canplaythrough', handleCanPlayThrough);
            videoElement.addEventListener('error', handleError);

            // Set preload to auto to fully load video data for scrubbing
            console.log('Fully loading video for scrubbing:', video.video_path);
            setPreloadMode('auto');
            setShowPoster(false); // Hide poster when fully loading
            videoElement.load(); // Trigger loading
        });
    }, [video?.video_path]); const handleClick = useCallback(() => {
        if (onOpenLightbox && video) {
            onOpenLightbox(video);
        }
    }, [onOpenLightbox, video]);

    const handleMouseEnter = useCallback(() => {
        const videoElement = videoRef.current;
        if (!videoElement || isPlaying || state.isScrubbingActive) return;

        // Show loading spinner immediately
        setIsLoadingOnHover(true);

        // Simply play the video - src is already set, preload="none" means it loads on demand
        videoElement.play().then(() => {
            // Hide loading spinner when playback starts
            setIsLoadingOnHover(false);
            setIsPlaying(true);
        }).catch(err => {
            console.warn('Failed to play video on hover:', err);
            setIsLoadingOnHover(false);
        });
    }, [isPlaying, state.isScrubbingActive]);

    const handleMouseLeave = useCallback(() => {
        const videoElement = videoRef.current;
        if (!videoElement || state.isScrubbingActive) return;

        // Clear loading state and stop playback
        setIsLoadingOnHover(false);
        videoElement.pause();
        videoElement.currentTime = 0;
        setIsPlaying(false);
    }, [state.isScrubbingActive]);

    const handleLoadedMetadata = useCallback(() => {
        const videoElement = videoRef.current;
        if (!videoElement) return;

        // Only set loaded state once to avoid multiple calls
        if (!isLoaded) {
            setIsLoaded(true);
            if (onMetadataLoaded && videoElement.duration) {
                onMetadataLoaded(videoElement.duration);
            }
        }
    }, [onMetadataLoaded, isLoaded]);

    // Load video when component mounts and when video prop changes
    // COMMENTED OUT: This was preloading all videos and defeating thumbnail performance
    // useEffect(() => {
    //     const videoElement = videoRef.current;
    //     if (videoElement && video?.video_path) {
    //         loadVideo(videoElement, video.video_path);
    //     }
    // }, [video?.video_path, loadVideo]);

    // Set up event listeners
    useEffect(() => {
        const videoElement = videoRef.current;
        if (!videoElement) return;

        videoElement.addEventListener('loadedmetadata', handleLoadedMetadata);

        return () => {
            videoElement.removeEventListener('loadedmetadata', handleLoadedMetadata);
        };
    }, [handleLoadedMetadata]);

    // Expose functions to parent components for play all and scrubber functionality
    useEffect(() => {
        const videoElement = videoRef.current;
        if (videoElement) {
            videoElement.loadVideoMetadata = loadVideoMetadata; // For play all
            videoElement.loadVideoSource = loadVideoSource;     // For scrubbing (full load)
            // Expose poster control functions for scrubbing
            videoElement.hidePoster = () => setShowPoster(false);
            videoElement.showPoster = () => setShowPoster(true);
        }
    }, [loadVideoMetadata, loadVideoSource]);

    // Notify parent when loaded
    useEffect(() => {
        if (isLoaded && onVideoLoaded) {
            onVideoLoaded(videoRef.current);
        }
    }, [isLoaded, onVideoLoaded]);

    if (!video) {
        return (
            <div
                className="video-placeholder"
                style={{
                    width: `${videoSize}px`,
                    height: `${Math.round(videoSize * 0.56)}px`
                }}
            >
                Missing
            </div>
        );
    }

    return (
        <div
            className={`video-cell ${isLoaded ? 'loaded' : ''}`}
            onClick={handleClick}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
        >
            <video
                ref={videoRef}
                className="video-element"
                style={{
                    width: `${videoSize}px`,
                    height: `${Math.round(videoSize * 0.56)}px`
                }}
                src={getVideoUrl(video.video_path)}
                muted
                loop
                preload={preloadMode}
                poster={showPoster ? getThumbnailPath(video.video_path) : ''}
            />
            <div className="video-overlay">
                <div>Seed: {video.seed}</div>
                <div>{video.width}x{video.height}, {video.num_frames}f</div>
                <div>Steps: {video.steps}, CFG: {video.cfg_scale}</div>
            </div>
            {isLoadingOnHover && <div className="hover-loading-spinner" />}
        </div>
    );
};

// Custom comparison function to prevent unnecessary re-renders
const areEqual = (prevProps, nextProps) => {
    // Only re-render if video object reference or essential props change
    return prevProps.video?.video_path === nextProps.video?.video_path &&
        prevProps.videoSize === nextProps.videoSize &&
        prevProps.onVideoLoaded === nextProps.onVideoLoaded &&
        prevProps.onMetadataLoaded === nextProps.onMetadataLoaded &&
        prevProps.onOpenLightbox === nextProps.onOpenLightbox;
};

export default React.memo(VideoCell, areEqual);
