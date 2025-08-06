import React, { useRef, useEffect, useState, useCallback } from 'react';
import { getVideoUrl, getThumbnailUrl } from '../../services/api';
import { useApp } from '../../context/AppContext';
import { useVideoCache } from '../../hooks/useVideoCache';

const VideoCell = ({ video, videoSize, onVideoLoaded, onMetadataLoaded, onOpenLightbox }) => {
    const videoRef = useRef(null);
    const imageRef = useRef(null);
    const hoverTimeoutRef = useRef(null);
    const [isLoaded, setIsLoaded] = useState(false);
    const [useVideoElement, setUseVideoElement] = useState(false); // Simple boolean switch
    const [forceVideoMode, setForceVideoMode] = useState(false); // Force video mode for playAll/scrubbing
    const { state } = useApp();

    // Get the current video source based on attention mode
    const getCurrentVideoPath = useCallback(() => {
        if (!video?.video_path) return null;

        // If attention mode is off, return normal video
        if (!state.attentionMode || !state.selectedToken || !state.currentExperiment?.attention_videos?.available) {
            return video.video_path;
        }

        // Try to find matching attention video
        const attentionVideos = state.currentExperiment.attention_videos;
        const promptNum = video.variation_num; // This should match prompt_XXX format
        const videoNum = video.video_number;

        const promptKey = `prompt_${promptNum.toString().padStart(3, '0')}`;
        const videoKey = `vid${videoNum.toString().padStart(3, '0')}`;

        const promptData = attentionVideos.prompts[promptKey];
        if (promptData && promptData.videos[videoKey]) {
            const tokenData = promptData.videos[videoKey].tokens[state.selectedToken];
            if (tokenData && tokenData.aggregate_overlay_path) {
                return tokenData.aggregate_overlay_path;
            }
        }

        // Fallback to normal video if attention video not found
        return video.video_path;
    }, [video, state.attentionMode, state.selectedToken, state.currentExperiment?.attention_videos]);

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

    // Simple cleanup function for timeouts
    useEffect(() => {
        return () => {
            if (hoverTimeoutRef.current) {
                clearTimeout(hoverTimeoutRef.current);
            }
        };
    }, []);

    // Listen for global playAll and pauseAll events
    useEffect(() => {
        const handleForceVideoMode = (event) => {
            console.log('VideoCell received forceVideoMode event');
            setForceVideoMode(true);
            setUseVideoElement(true);
        };

        const handleExitForceMode = (event) => {
            console.log('VideoCell received exitForceMode event');
            setForceVideoMode(false);
        };

        document.addEventListener('forceVideoMode', handleForceVideoMode);
        document.addEventListener('exitForceMode', handleExitForceMode);

        return () => {
            document.removeEventListener('forceVideoMode', handleForceVideoMode);
            document.removeEventListener('exitForceMode', handleExitForceMode);
        };
    }, []);

    const handleClick = useCallback(() => {
        if (onOpenLightbox && video) {
            onOpenLightbox(video);
        }
    }, [onOpenLightbox, video]);

    const handleMouseEnter = useCallback(() => {
        if (state.isScrubbingActive) return;

        // Clear any pending timeout
        if (hoverTimeoutRef.current) {
            clearTimeout(hoverTimeoutRef.current);
            hoverTimeoutRef.current = null;
        }

        // Simply switch to video element - autoplay will handle the rest
        setUseVideoElement(true);
    }, [state.isScrubbingActive]);

    const handleMouseLeave = useCallback(() => {
        if (state.isScrubbingActive || forceVideoMode) return;

        // Clear any pending timeout
        if (hoverTimeoutRef.current) {
            clearTimeout(hoverTimeoutRef.current);
        }

        // Switch back to image after a short delay (only if not forced)
        hoverTimeoutRef.current = setTimeout(() => {
            if (!state.isScrubbingActive && !forceVideoMode) {
                setUseVideoElement(false);
            }
        }, 300);
    }, [state.isScrubbingActive, forceVideoMode]);

    // Simple metadata handler
    const handleLoadedMetadata = useCallback(() => {
        if (!isLoaded) {
            setIsLoaded(true);
            if (onMetadataLoaded && videoRef.current?.duration) {
                onMetadataLoaded(videoRef.current.duration);
            }
        }
    }, [onMetadataLoaded, isLoaded]);

    // Expose loadVideoMetadata for play all functionality
    useEffect(() => {
        const videoElement = videoRef.current;
        if (videoElement && useVideoElement) {
            // Simple function to switch to video for play all
            videoElement.loadVideoMetadata = () => {
                setUseVideoElement(true);
                return Promise.resolve();
            };

            if (onVideoLoaded) {
                onVideoLoaded(videoElement);
            }
        }
    }, [useVideoElement, onVideoLoaded]);

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
            {(useVideoElement || forceVideoMode) ? (
                <video
                    ref={videoRef}
                    className="video-element"
                    style={{
                        width: `${videoSize}px`,
                        height: `${Math.round(videoSize * 0.56)}px`
                    }}
                    src={getVideoUrl(getCurrentVideoPath())}
                    muted
                    loop
                    autoPlay
                    preload="auto"
                    playsInline
                    onLoadedMetadata={handleLoadedMetadata}
                    poster={getThumbnailPath(getCurrentVideoPath())}
                />
            ) : (
                <img
                    ref={imageRef}
                    className="video-element video-thumbnail"
                    style={{
                        width: `${videoSize}px`,
                        height: `${Math.round(videoSize * 0.56)}px`,
                        objectFit: 'cover'
                    }}
                    src={getThumbnailPath(getCurrentVideoPath())}
                    alt={`Video thumbnail - Seed ${video.seed}`}
                />
            )}
            <div className="video-overlay">
                <div>Seed: {video.seed}</div>
                <div>{video.width}x{video.height}, {video.num_frames}f</div>
                <div>Steps: {video.steps}, CFG: {video.cfg_scale}</div>
                {state.attentionMode && state.selectedToken && (
                    <div>🎯 {state.selectedToken}</div>
                )}
            </div>
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
