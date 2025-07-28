import React, { useRef, useEffect, useState, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import { useVideoCache } from '../../hooks/useVideoCache';

const VideoCell = ({ video, videoSize, onVideoLoaded, onMetadataLoaded, onOpenLightbox }) => {
    const videoRef = useRef(null);
    const [isLoaded, setIsLoaded] = useState(false);
    const [isPlaying, setIsPlaying] = useState(false);
    const [isLoadingOnHover, setIsLoadingOnHover] = useState(false);
    const { state } = useApp();
    // const { loadVideo } = useVideoCache(); // Commented out to preserve thumbnail performance

    // Calculate thumbnail path from video path
    const getThumbnailPath = useCallback((videoPath) => {
        if (!videoPath) return null;
        try {
            // Replace .mp4 extension with .jpg for thumbnail
            const thumbnailPath = videoPath.replace(/\.mp4$/, '.jpg');
            return `/api/thumbnail/${thumbnailPath}`;
        } catch (error) {
            console.warn('Error generating thumbnail path:', error);
            return null;
        }
    }, []);

    // Simple function to ensure video can be played (for programmatic play all/scrubber)
    const loadVideoSource = useCallback(() => {
        // Since src is set on render and preload="none", just return resolved promise
        return Promise.resolve();
    }, []);

    const handleClick = useCallback(() => {
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

        setIsLoaded(true);
        if (onMetadataLoaded && videoElement.duration) {
            onMetadataLoaded(videoElement.duration);
        }
    }, [onMetadataLoaded]);

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

    // Expose loadVideoSource to parent components for play all and scrubber functionality
    useEffect(() => {
        const videoElement = videoRef.current;
        if (videoElement) {
            videoElement.loadVideoSource = loadVideoSource;
        }
    }, [loadVideoSource]);

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
                src={`/api/video/${video.video_path}`}
                muted
                loop
                preload="none"
                poster={getThumbnailPath(video.video_path)}
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
