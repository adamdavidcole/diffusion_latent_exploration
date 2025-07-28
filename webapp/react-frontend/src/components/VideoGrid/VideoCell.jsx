import React, { useRef, useEffect, useState, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import { useVideoCache } from '../../hooks/useVideoCache';

const VideoCell = ({ video, videoSize, onVideoLoaded, onMetadataLoaded, onOpenLightbox }) => {
    const videoRef = useRef(null);
    const [isLoaded, setIsLoaded] = useState(false);
    const [isPlaying, setIsPlaying] = useState(false);
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

    const handleClick = useCallback(() => {
        if (onOpenLightbox && video) {
            onOpenLightbox(video);
        }
    }, [onOpenLightbox, video]);

    const handleMouseEnter = useCallback(() => {
        try {
            const videoElement = videoRef.current;
            if (!videoElement || isPlaying || state.isScrubbingActive) return;

            // Load the video source if not already loaded
            if (!videoElement.src && video?.video_path) {
                videoElement.src = `/api/video/${video.video_path}`;
            }

            // Play the video (will load first if needed)
            videoElement.play().catch(err => {
                console.warn('Failed to play video on hover:', err);
            });
            setIsPlaying(true);
        } catch (error) {
            console.error('Error in handleMouseEnter:', error);
        }
    }, [isPlaying, state.isScrubbingActive, video?.video_path]);

    const handleMouseLeave = useCallback(() => {
        try {
            const videoElement = videoRef.current;
            if (!videoElement || state.isScrubbingActive) return;

            videoElement.pause();
            videoElement.currentTime = 0;
            setIsPlaying(false);
        } catch (error) {
            console.error('Error in handleMouseLeave:', error);
        }
    }, [state.isScrubbingActive]);

    const handleLoadedMetadata = useCallback(() => {
        try {
            const videoElement = videoRef.current;
            if (!videoElement) return;

            setIsLoaded(true);
            if (onMetadataLoaded && videoElement.duration) {
                onMetadataLoaded(videoElement.duration);
            }
        } catch (error) {
            console.error('Error in handleLoadedMetadata:', error);
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
                poster={getThumbnailPath(video.video_path)}
            />
                        <div className="video-overlay">
                <div>Seed: {video.seed}</div>
                <div>{video.width}x{video.height}, {video.num_frames}f</div>
                <div>Steps: {video.steps}, CFG: {video.cfg_scale}</div>
            </div>
            {/* Loading spinner hidden for thumbnail-based loading */}
            {/* {!isLoaded && <div className="loading-spinner" />} */}
        </div>
    );
};

export default React.memo(VideoCell);
