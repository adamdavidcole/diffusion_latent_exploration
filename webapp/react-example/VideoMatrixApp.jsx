// React Component Example - Video Matrix App
// This shows what the main app would look like converted to React

import React, { useState, useEffect, useCallback } from 'react';
import './VideoMatrix.css'; // Same CSS file

const VideoMatrixApp = () => {
    const [experiments, setExperiments] = useState([]);
    const [currentExperiment, setCurrentExperiment] = useState(null);
    const [allVideos, setAllVideos] = useState([]);
    const [videoSize, setVideoSize] = useState(200);
    const [showLabels, setShowLabels] = useState(true);
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const [loading, setLoading] = useState(false);

    // Load experiments
    const loadExperiments = useCallback(async () => {
        try {
            const response = await fetch('/api/experiments');
            const data = await response.json();
            setExperiments(data);
        } catch (error) {
            console.error('Error loading experiments:', error);
        }
    }, []);

    // Select experiment
    const selectExperiment = useCallback(async (experimentName) => {
        setLoading(true);
        try {
            const response = await fetch(`/api/experiment/${experimentName}`);
            const experiment = await response.json();

            if (experiment.error) {
                throw new Error(experiment.error);
            }

            setCurrentExperiment(experiment);
            // Extract all video elements after render
            setTimeout(() => {
                setAllVideos(Array.from(document.querySelectorAll('video')));
            }, 100);

        } catch (error) {
            console.error('Error loading experiment:', error);
        } finally {
            setLoading(false);
        }
    }, []);

    // Video controls
    const playAllVideos = useCallback(() => {
        allVideos.forEach(video => {
            video.currentTime = 0;
            video.play();
        });
    }, [allVideos]);

    const pauseAllVideos = useCallback(() => {
        allVideos.forEach(video => video.pause());
    }, [allVideos]);

    const muteAllVideos = useCallback(() => {
        const anyUnmuted = allVideos.some(video => !video.muted);
        allVideos.forEach(video => {
            video.muted = anyUnmuted;
        });
        return anyUnmuted;
    }, [allVideos]);

    // Initialize
    useEffect(() => {
        loadExperiments();
    }, [loadExperiments]);

    return (
        <div className="app-container">
            {/* Sidebar Component */}
            <Sidebar
                experiments={experiments}
                collapsed={sidebarCollapsed}
                onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
                onSelectExperiment={selectExperiment}
                onRescan={loadExperiments}
            />

            {/* Main Content */}
            <main className="main-content">
                {/* Header Component */}
                <Header
                    experiment={currentExperiment}
                    videoSize={videoSize}
                    showLabels={showLabels}
                    onVideoSizeChange={setVideoSize}
                    onToggleLabels={() => setShowLabels(!showLabels)}
                    onRescan={loadExperiments}
                />

                {/* Video Grid Component */}
                <VideoGrid
                    experiment={currentExperiment}
                    videoSize={videoSize}
                    showLabels={showLabels}
                    loading={loading}
                />
            </main>

            {/* Sync Controls Component */}
            <SyncControls
                visible={allVideos.length > 0}
                onPlayAll={playAllVideos}
                onPauseAll={pauseAllVideos}
                onMuteAll={muteAllVideos}
            />
        </div>
    );
};

// Individual Components (would be in separate files)

const Sidebar = ({ experiments, collapsed, onToggleCollapse, onSelectExperiment, onRescan }) => (
    <nav className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
        <div className="sidebar-header">
            <h1 className="logo">🎬 WAN Viewer</h1>
            <button className="collapse-btn" onClick={onToggleCollapse}>
                <span>{collapsed ? '→' : '←'}</span>
            </button>
        </div>

        <div className="experiments-list">
            {experiments.map(exp => (
                <ExperimentItem
                    key={exp.name}
                    experiment={exp}
                    onClick={() => onSelectExperiment(exp.name)}
                />
            ))}
        </div>
    </nav>
);

const ExperimentItem = ({ experiment, onClick, active }) => (
    <div
        className={`experiment-item ${active ? 'active' : ''}`}
        onClick={onClick}
    >
        <div className="experiment-header">
            <div className="experiment-name">{experiment.name}</div>
            <div className="experiment-meta">
                <span>{experiment.videos_count} videos</span>
                <span>{experiment.variations_count} variations</span>
                <span>{experiment.seeds_count} seeds</span>
            </div>
            <div className="experiment-prompt">{experiment.base_prompt}</div>
        </div>
    </div>
);

const VideoGrid = ({ experiment, videoSize, showLabels, loading }) => {
    if (loading) {
        return (
            <div className="loading">
                <div className="loading-spinner"></div>
                Loading experiment...
            </div>
        );
    }

    if (!experiment) {
        return (
            <div className="loading">
                Select an experiment to view videos
            </div>
        );
    }

    return (
        <div className="video-container">
            <div className="seeds-header">
                {experiment.seeds.map(seed => (
                    <div key={seed} className="seed-label" style={{ width: `${videoSize}px` }}>
                        Seed {seed}
                    </div>
                ))}
            </div>

            <div className="video-grid">
                {experiment.video_grid.map((row, idx) => (
                    <VideoRow
                        key={idx}
                        row={row}
                        seeds={experiment.seeds}
                        videoSize={videoSize}
                        showLabels={showLabels}
                    />
                ))}
            </div>
        </div>
    );
};

const VideoRow = ({ row, seeds, videoSize, showLabels }) => (
    <div className="grid-row">
        <div className={`row-label ${showLabels ? '' : 'hidden'}`}>
            {row.variation}
        </div>
        <div className="videos-row">
            {seeds.map(seed => {
                const video = row.videos.find(v => v && v.seed === seed);
                return (
                    <VideoCell
                        key={seed}
                        video={video}
                        videoSize={videoSize}
                    />
                );
            })}
        </div>
    </div>
);

const VideoCell = ({ video, videoSize }) => {
    const handleVideoHover = useCallback((video) => {
        video.play();
    }, []);

    const handleVideoLeave = useCallback((video) => {
        video.pause();
        video.currentTime = 0;
    }, []);

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
        <div className="video-cell">
            <video
                className="video-element"
                style={{
                    width: `${videoSize}px`,
                    height: `${Math.round(videoSize * 0.56)}px`
                }}
                muted
                loop
                preload="metadata"
                onMouseEnter={(e) => handleVideoHover(e.target)}
                onMouseLeave={(e) => handleVideoLeave(e.target)}
            >
                <source src={`/api/video/${video.video_path}`} type="video/mp4" />
            </video>
            <div className="video-overlay">
                <div>Seed: {video.seed}</div>
                <div>{video.width}x{video.height}, {video.num_frames}f</div>
                <div>Steps: {video.steps}, CFG: {video.cfg_scale}</div>
            </div>
        </div>
    );
};

export default VideoMatrixApp;
