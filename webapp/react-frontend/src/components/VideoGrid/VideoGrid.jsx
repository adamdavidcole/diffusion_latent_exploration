import React, { useCallback, useRef, useEffect, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { useIntersectionObserver } from '../../hooks/useIntersectionObserver';
import { useVideoCache } from '../../hooks/useVideoCache';
import VideoCell from './VideoCell';
import VideoLightbox from './VideoLightbox';

const VideoGrid = () => {
    const context = useApp();

    // Guard against context not being ready
    if (!context || !context.actions) {
        return <div>Loading...</div>;
    }

    const { state, actions } = context;
    const { currentExperiment, videoSize, showLabels } = state;
    const allVideosRef = useRef([]);
    const videoGridRef = useRef(null);
    const [lightboxVideo, setLightboxVideo] = useState(null);
    const currentExperimentNameRef = useRef(null);

    // Use video cache hook
    const { cancelInflightRequests, getCacheStats } = useVideoCache();

    // Handle lightbox open/close
    const handleOpenLightbox = useCallback((video) => {
        setLightboxVideo(video);
    }, []);

    const handleCloseLightbox = useCallback(() => {
        setLightboxVideo(null);
    }, []);

    // Handle navigation in lightbox
    const handleLightboxNavigation = useCallback((direction) => {
        if (!lightboxVideo || !currentExperiment?.video_grid) return;
        
        const grid = currentExperiment.video_grid;
        let currentRowIndex = -1;
        let currentVideoIndex = -1;
        
        // Find current video position in grid
        for (let rowIndex = 0; rowIndex < grid.length; rowIndex++) {
            const videoIndex = grid[rowIndex].videos.findIndex(v => 
                v.video_path === lightboxVideo.video_path
            );
            if (videoIndex !== -1) {
                currentRowIndex = rowIndex;
                currentVideoIndex = videoIndex;
                break;
            }
        }
        
        if (currentRowIndex === -1) return;
        
        let newRowIndex = currentRowIndex;
        let newVideoIndex = currentVideoIndex;
        
        switch (direction) {
            case 'left':
                newVideoIndex = currentVideoIndex - 1;
                if (newVideoIndex < 0) {
                    newVideoIndex = grid[currentRowIndex].videos.length - 1;
                }
                break;
            case 'right':
                newVideoIndex = currentVideoIndex + 1;
                if (newVideoIndex >= grid[currentRowIndex].videos.length) {
                    newVideoIndex = 0;
                }
                break;
            case 'up':
                newRowIndex = currentRowIndex - 1;
                if (newRowIndex < 0) {
                    newRowIndex = grid.length - 1;
                }
                // Ensure video index exists in new row
                if (newVideoIndex >= grid[newRowIndex].videos.length) {
                    newVideoIndex = grid[newRowIndex].videos.length - 1;
                }
                break;
            case 'down':
                newRowIndex = currentRowIndex + 1;
                if (newRowIndex >= grid.length) {
                    newRowIndex = 0;
                }
                // Ensure video index exists in new row
                if (newVideoIndex >= grid[newRowIndex].videos.length) {
                    newVideoIndex = grid[newRowIndex].videos.length - 1;
                }
                break;
            default:
                return;
        }
        
        const newVideo = grid[newRowIndex].videos[newVideoIndex];
        if (newVideo) {
            setLightboxVideo(newVideo);
        }
    }, [lightboxVideo, currentExperiment?.video_grid]);

    // Get preview information for navigation
    const getNavigationPreview = useCallback((direction) => {
        if (!lightboxVideo || !currentExperiment?.video_grid) return null;
        
        const grid = currentExperiment.video_grid;
        let currentRowIndex = -1;
        let currentVideoIndex = -1;
        
        // Find current video position in grid
        for (let rowIndex = 0; rowIndex < grid.length; rowIndex++) {
            const videoIndex = grid[rowIndex].videos.findIndex(v => 
                v.video_path === lightboxVideo.video_path
            );
            if (videoIndex !== -1) {
                currentRowIndex = rowIndex;
                currentVideoIndex = videoIndex;
                break;
            }
        }
        
        if (currentRowIndex === -1) return null;
        
        let newRowIndex = currentRowIndex;
        let newVideoIndex = currentVideoIndex;
        
        switch (direction) {
            case 'left':
                newVideoIndex = currentVideoIndex - 1;
                if (newVideoIndex < 0) {
                    newVideoIndex = grid[currentRowIndex].videos.length - 1;
                }
                return `Seed ${grid[currentRowIndex].videos[newVideoIndex]?.seed}`;
            case 'right':
                newVideoIndex = currentVideoIndex + 1;
                if (newVideoIndex >= grid[currentRowIndex].videos.length) {
                    newVideoIndex = 0;
                }
                return `Seed ${grid[currentRowIndex].videos[newVideoIndex]?.seed}`;
            case 'up':
                newRowIndex = currentRowIndex - 1;
                if (newRowIndex < 0) {
                    newRowIndex = grid.length - 1;
                }
                if (newVideoIndex >= grid[newRowIndex].videos.length) {
                    newVideoIndex = grid[newRowIndex].videos.length - 1;
                }
                return grid[newRowIndex].variation.length > 40 
                    ? grid[newRowIndex].variation.substring(0, 40) + '...'
                    : grid[newRowIndex].variation;
            case 'down':
                newRowIndex = currentRowIndex + 1;
                if (newRowIndex >= grid.length) {
                    newRowIndex = 0;
                }
                if (newVideoIndex >= grid[newRowIndex].videos.length) {
                    newVideoIndex = grid[newRowIndex].videos.length - 1;
                }
                return grid[newRowIndex].variation.length > 40 
                    ? grid[newRowIndex].variation.substring(0, 40) + '...'
                    : grid[newRowIndex].variation;
            default:
                return null;
        }
    }, [lightboxVideo, currentExperiment?.video_grid]);

    // Calculate proportional gap
    const calculateGap = useCallback((size) => {
        const minGap = 0.05;
        const maxGap = 2;
        const minSize = 25;
        const maxSize = 1000;
        const gapSize = minGap + (maxGap - minGap) * ((size - minSize) / (maxSize - minSize));
        return Math.max(minGap, Math.min(maxGap, gapSize));
    }, []);

    const gapRem = calculateGap(videoSize);

    // Handle video loading via intersection observer
    const handleIntersection = useCallback((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const videoElement = entry.target;
                // VideoCell component handles the actual loading
                videoElement.dispatchEvent(new CustomEvent('enterViewport'));
            }
        });
    }, []);

    const { observe, disconnect } = useIntersectionObserver(handleIntersection, {
        root: null, // Use viewport instead of grid container
        rootMargin: '100px',
        threshold: 0.1
    });

    // Handle when a video is loaded
    const handleVideoLoaded = useCallback((videoElement) => {
        if (videoElement && !allVideosRef.current.includes(videoElement)) {
            allVideosRef.current.push(videoElement);
            observe(videoElement);
            // Removed console.log to reduce noise
        }
    }, [observe]);

    // Handle video metadata loaded (for duration)
    const handleMetadataLoaded = useCallback((duration) => {
        if (state?.videoDuration === 0 && duration && actions && typeof actions.setVideoDuration === 'function') {
            actions.setVideoDuration(duration);
        }
    }, [state?.videoDuration, actions]);

    // Clean up videos array when experiment changes
    useEffect(() => {
        // Check if experiment actually changed
        const newExperimentName = currentExperiment?.name;
        const hasExperimentChanged = currentExperimentNameRef.current !== newExperimentName;

        if (hasExperimentChanged) {
            console.log('Experiment changed, cancelling inflight video requests');

            // Cancel inflight requests but keep cache intact
            cancelInflightRequests();

            // Clear local state
            allVideosRef.current = [];
            disconnect();

            // Close lightbox if open
            setLightboxVideo(null);

            // Update current experiment reference
            currentExperimentNameRef.current = newExperimentName;

            // Log cache stats for debugging
            const stats = getCacheStats();
            console.log('Cache stats after cleanup:', stats);
        }
    }, [currentExperiment?.name, disconnect, cancelInflightRequests, getCacheStats]);

    if (!currentExperiment) {
        return (
            <div className="empty-state">
                <h3>No experiment selected</h3>
                <p>Select an experiment from the sidebar to view videos.</p>
            </div>
        );
    }

    if (currentExperiment.video_grid.length === 0) {
        return (
            <div className="empty-state">
                <h3>No videos found</h3>
                <p>This experiment doesn't contain any videos.</p>
            </div>
        );
    }

    return (
        <>
            {/* Header */}
            <div className="experiment-header">
                <h2 id="experiment-title">{currentExperiment.name}</h2>
                <div className="experiment-stats">
                    {currentExperiment.model_id && (
                        <span className="stat-item">{currentExperiment.model_id.split('/').pop().replace('-Diffusers', '').replace('Wan2.1-T2V-', '')}</span>
                    )}
                    <span className="stat-item">{currentExperiment.videos_count} videos</span>
                    <span className="stat-item">{currentExperiment.variations_count} variations</span>
                    <span className="stat-item">{currentExperiment.seeds_count} seeds</span>
                </div>
                <p
                    id="base-prompt"
                    className="base-prompt"
                    title={currentExperiment.base_prompt}
                >
                    {currentExperiment.base_prompt}
                </p>
            </div>

            {/* Seeds Header */}
            <div
                className="seeds-header"
                id="seeds-header"
                style={{ gap: `${gapRem}rem` }}
            >
                <div className={`row-label ${showLabels ? '' : 'hidden'}`}></div>
                {currentExperiment.seeds.map(seed => (
                    <div
                        key={seed}
                        className="seed-label"
                        style={{ width: `${videoSize}px` }}
                    >
                        Seed {seed}
                    </div>
                ))}
            </div>

            {/* Scrollable Video Grid */}
            <div
                ref={videoGridRef}
                className="video-grid"
                id="video-grid"
                style={{ gap: `${gapRem}rem` }}
            >
                {currentExperiment.video_grid.map((row, rowIndex) => (
                    <div
                        key={rowIndex}
                        className="grid-row"
                        data-row-index={rowIndex}
                        style={{ gap: `${gapRem}rem` }}
                    >
                        <div className={`row-label ${showLabels ? '' : 'hidden'}`}>
                            {row.variation}
                        </div>
                        <div
                            className="videos-row"
                            style={{ gap: `${gapRem}rem` }}
                        >
                            {currentExperiment.seeds.map(seed => {
                                const video = row.videos.find(v => v && v.seed === seed);
                                return (
                                    <VideoCell
                                        key={`${rowIndex}-${seed}`}
                                        video={video}
                                        videoSize={videoSize}
                                        onVideoLoaded={handleVideoLoaded}
                                        onMetadataLoaded={handleMetadataLoaded}
                                        onOpenLightbox={handleOpenLightbox}
                                    />
                                );
                            })}
                        </div>
                    </div>
                ))}
            </div>

            {/* Video Lightbox */}
            <VideoLightbox
                video={lightboxVideo}
                isOpen={!!lightboxVideo}
                onClose={handleCloseLightbox}
                onNavigate={handleLightboxNavigation}
                getPreviewInfo={getNavigationPreview}
            />
        </>
    );
};

export default VideoGrid;
