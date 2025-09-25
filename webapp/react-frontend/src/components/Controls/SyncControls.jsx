import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useApp } from '../../context/AppContext';
import { useVideoControls } from '../../hooks/useVideoControls';

const SyncControls = () => {
    const { state, actions } = useApp();
    const { currentExperiment } = state;
    const { playAllVideos, pauseAllVideos, muteAllVideos, scrubAllVideos } = useVideoControls();
    const videoDuration = currentExperiment && currentExperiment.duration_seconds || 0;


    const [isMuted, setIsMuted] = useState(true);
    const [scrubberValue, setScrubberValue] = useState(0);
    const [scrubberTime, setScrubberTime] = useState('0:00');
    const [isPlayAllLoading, setIsPlayAllLoading] = useState(false);
    const sizeSliderTimeoutRef = useRef(null);



    // Throttled video size update
    const handleVideoSizeChange = useCallback((e) => {
        const newSize = parseInt(e.target.value);

        // Clear previous timeout
        if (sizeSliderTimeoutRef.current) {
            clearTimeout(sizeSliderTimeoutRef.current);
        }

        // Set new timeout for throttled update
        sizeSliderTimeoutRef.current = setTimeout(() => {
            actions.setVideoSize(newSize);
        }, 100); // 100ms throttle
    }, [actions]);

    // Update scrubber time display
    const updateScrubberTime = useCallback((currentTime) => {
        const minutes = Math.floor(currentTime / 60);
        const seconds = Math.floor(currentTime % 60);
        const timeString = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        setScrubberTime(timeString);
    }, []);

    // Handle scrubber change
    const handleScrubberChange = useCallback((e) => {
        const percentage = parseFloat(e.target.value);
        setScrubberValue(percentage);
        scrubAllVideos(percentage);

        const currentTime = (percentage / 100) * videoDuration;
        updateScrubberTime(currentTime);
    }, [scrubAllVideos, updateScrubberTime, videoDuration]);

    // Handle play all
    const handlePlayAll = useCallback(() => {
        console.log('SyncControls: Play All clicked');
        playAllVideos(setIsPlayAllLoading);
        setScrubberValue(0);
        updateScrubberTime(0);
    }, [playAllVideos, updateScrubberTime]);

    // Handle mute toggle
    const handleMuteToggle = useCallback(() => {
        console.log('SyncControls: Mute toggle clicked');
        const newMuteState = muteAllVideos();
        setIsMuted(newMuteState);
    }, [muteAllVideos]);

    // Reset scrubber when experiment changes
    useEffect(() => {
        setScrubberValue(0);
        updateScrubberTime(0);
    }, [currentExperiment, updateScrubberTime]);

    // Listen for reset scrubber events
    useEffect(() => {
        const handleResetScrubber = () => {
            setScrubberValue(0);
            updateScrubberTime(0);
        };

        document.addEventListener('resetScrubber', handleResetScrubber);
        return () => {
            document.removeEventListener('resetScrubber', handleResetScrubber);
        };
    }, [updateScrubberTime]);

    // Cleanup timeout on unmount
    useEffect(() => {
        return () => {
            if (sizeSliderTimeoutRef.current) {
                clearTimeout(sizeSliderTimeoutRef.current);
            }
        };
    }, []);

    // Don't show controls if no experiment is loaded or if video_grid is not available yet
    if (!currentExperiment || !currentExperiment.video_grid) {
        return null;
    }

    const hasVideos = currentExperiment.video_grid.length > 0;

    return (
        <div className={`sync-controls ${hasVideos ? 'visible' : ''}`}>
            {/* First row: Play/Pause and Labels buttons */}
            <div className="controls-row">
                <label>Playback:</label>

                <button
                    className={`control-btn ${isPlayAllLoading ? 'loading' : ''}`}
                    onClick={handlePlayAll}
                    disabled={!hasVideos || isPlayAllLoading}
                >
                    {isPlayAllLoading ? '⏳ Loading...' : '▶️ Play All'}
                </button>

                <button
                    className="control-btn"
                    onClick={pauseAllVideos}
                    disabled={!hasVideos}
                >
                    ⏸️ Pause All
                </button>

                {/* <button
                    className="control-btn"
                    onClick={actions.toggleLabels}
                >
                    {state.showLabels ? '🏷️ Hide Labels' : '🏷️ Show Labels'}
                </button> */}


            </div>

            {/* Second row: Video Scrubber */}
            {videoDuration > 0 && (
                <div className="scrubber-row">
                    <label htmlFor="global-scrubber">Video Time:</label>
                    <input
                        id="global-scrubber"
                        type="range"
                        min="0"
                        max="100"
                        step="0.1"
                        value={scrubberValue}
                        onChange={handleScrubberChange}
                        className="scrubber"
                    />
                    <span className="scrubber-time">{scrubberTime}</span>
                </div>
            )}

            {/* Third row: Video Size */}
            <div className="size-row">
                <label htmlFor="size-slider">Video Size:</label>
                <input
                    id="size-slider"
                    type="range"
                    min="25"
                    max="1000"
                    value={state.videoSize}
                    onChange={handleVideoSizeChange}
                    className="size-slider"
                />
                <span className="size-value">{state.videoSize}px</span>
            </div>

            {/* Fourth row: Similarity Ranking */}
            {state.similarityAnalysis && (
                <div className="similarity-row">
                    <label htmlFor="similarity-sort">Rank by:</label>
                    <select
                        id="similarity-sort"
                        value={state.similaritySortBy}
                        onChange={(e) => actions.setSimilaritySortBy(e.target.value)}
                        className="similarity-sort-select"
                    >
                        <option value="default">Default (generation order)</option>
                        <option value="weighted_similarity_distance">Most Different (weighted)</option>
                        {state.similarityAnalysis.metrics_used?.map(metric => (
                            <option key={metric} value={metric}>
                                {metric === 'clip' ? 'CLIP Similarity' :
                                    metric === 'lpips' ? 'LPIPS Similarity' :
                                        metric === 'ssim' ? 'SSIM Similarity' :
                                            metric === 'mse' ? 'MSE Similarity' :
                                                metric === 'phash' ? 'Perceptual Hash' :
                                                    metric.toUpperCase() + ' Similarity'}
                            </option>
                        ))}
                    </select>
                    <button
                        className="control-btn"
                        onClick={actions.toggleSimilarityMetrics}
                    >
                        📊
                    </button>
                </div>
            )}
        </div>
    );
};

export default SyncControls;
