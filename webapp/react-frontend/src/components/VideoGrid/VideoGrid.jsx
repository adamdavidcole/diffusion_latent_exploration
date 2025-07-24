import React, { useCallback, useRef, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { useIntersectionObserver } from '../../hooks/useIntersectionObserver';
import VideoCell from './VideoCell';

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
        allVideosRef.current = [];
        disconnect();
        // Removed console.log to reduce noise
    }, [currentExperiment, disconnect]);

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
                                    />
                                );
                            })}
                        </div>
                    </div>
                ))}
            </div>
        </>
    );
};

export default VideoGrid;
