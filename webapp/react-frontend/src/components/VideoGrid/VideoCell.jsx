import React, { useRef, useEffect, useState, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import { useVideoCache } from '../../hooks/useVideoCache';

const VideoCell = ({ video, videoSize, onVideoLoaded, onMetadataLoaded, onOpenLightbox }) => {
    const videoRef = useRef(null);
    const [isLoaded, setIsLoaded] = useState(false);
    const [isPlaying, setIsPlaying] = useState(false);
    const { state } = useApp();
    const { loadVideo } = useVideoCache();

    const handleClick = useCallback(() => {
        if (onOpenLightbox && video) {
            onOpenLightbox(video);
        }
    }, [onOpenLightbox, video]);

    const handleMouseEnter = useCallback(() => {
        const videoElement = videoRef.current;
        if (!videoElement || isPlaying || state.isScrubbingActive) return;

        videoElement.play();
        setIsPlaying(true);
    }, [isPlaying, state.isScrubbingActive]);

    const handleMouseLeave = useCallback(() => {
        const videoElement = videoRef.current;
        if (!videoElement || state.isScrubbingActive) return;

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
    useEffect(() => {
        const videoElement = videoRef.current;
        if (videoElement && video?.video_path) {
            loadVideo(videoElement, video.video_path);
        }
    }, [video?.video_path, loadVideo]);

    // Set up event listeners
    useEffect(() => {
        const videoElement = videoRef.current;
        if (!videoElement) return;

        videoElement.addEventListener('loadedmetadata', handleLoadedMetadata);

        return () => {
            videoElement.removeEventListener('loadedmetadata', handleLoadedMetadata);
        };
    }, [handleLoadedMetadata]);

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
                muted
                loop
                preload="none"
            />
            <div className="video-overlay">
                <div>Seed: {video.seed}</div>
                <div>{video.width}x{video.height}, {video.num_frames}f</div>
                <div>Steps: {video.steps}, CFG: {video.cfg_scale}</div>
            </div>
            {!isLoaded && <div className="loading-spinner" />}
        </div>
    );
};

export default React.memo(VideoCell);
