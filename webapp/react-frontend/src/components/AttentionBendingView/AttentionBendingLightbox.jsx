import React, { useRef, useEffect, useCallback, useState } from 'react';
import { getVideoUrl } from '../../services/api';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import './AttentionBendingLightbox.css';

const AttentionBendingLightbox = ({
    video,
    baselineVideo,
    isOpen,
    onClose,
    onNavigate,
    experimentPath,
    cachedAttentionData,
    attentionDataLoading,
    attentionDataError
}) => {
    const mainVideoRef = useRef(null);
    const baselineVideoRef = useRef(null);
    const lightboxRef = useRef(null);
    const { state } = useApp();

    // Feature toggles
    const [compareToBaseline, setCompareToBaseline] = useState(false);

    // Attention maps data - extracted from cached data
    const [attentionData, setAttentionData] = useState(null);
    const [baselineAttentionData, setBaselineAttentionData] = useState(null);
    const [localError, setLocalError] = useState(null);

    // Format operation details for display
    const formatOperationDetails = useCallback((bending) => {
        if (!bending) return 'Baseline (No Bending)';

        const parts = [];
        const type = bending.transformation_type;
        const params = bending.transformation_params || {};

        // Operation type and parameters
        if (type === 'scale') {
            const scale = params.scale_x || params.scale || 1;
            parts.push(`Scale: ${scale.toFixed(2)}×`);
        } else if (type === 'rotate') {
            parts.push(`Rotate: ${(params.angle || 0).toFixed(1)}°`);
        } else if (type === 'translate') {
            if (params.translate_x !== undefined) {
                parts.push(`Translate X: ${params.translate_x.toFixed(2)}`);
            } else if (params.translate_y !== undefined) {
                parts.push(`Translate Y: ${params.translate_y.toFixed(2)}`);
            }
        } else if (type === 'amplify') {
            parts.push(`Amplify: ${(params.amplify_factor || 1).toFixed(2)}×`);
        } else if (type === 'sharpen') {
            parts.push(`Sharpen: ${(params.sharpen_amount || 1).toFixed(2)}×`);
        } else if (type === 'blur') {
            parts.push(`Blur: ${(params.blur_amount || 0).toFixed(2)}×`);
        } else if (type === 'flip') {
            if (params.flip_horizontal) parts.push('Flip: Horizontal');
            if (params.flip_vertical) parts.push('Flip: Vertical');
        }

        // Layers
        if (bending.layer_range) {
            const [start, end] = bending.layer_range;
            if (start === end) {
                parts.push(`Layer: ${start}`);
            } else {
                parts.push(`Layers: ${start}-${end}`);
            }
        }

        // Timesteps
        if (bending.timestep_range) {
            const [start, end] = bending.timestep_range;
            if (start === end) {
                parts.push(`Step: ${start}`);
            } else {
                parts.push(`Steps: ${start}-${end}`);
            }
        }

        // Tokens
        const targetToken = bending.target_token || 'ALL';
        if (targetToken !== 'ALL') {
            const resolvedTokens = bending.resolved_tokens?.[targetToken] || [targetToken];
            if (resolvedTokens.length === 1) {
                parts.push(`Token: "${resolvedTokens[0]}"`);
            } else if (resolvedTokens.length > 1) {
                parts.push(`Tokens: ${resolvedTokens.length} (${resolvedTokens.slice(0, 2).join(', ')}...)`);
            }
        }

        return parts.join(' | ');
    }, []);

    // Synchronize playback between main and baseline videos
    const syncVideos = useCallback(() => {
        if (!compareToBaseline || !mainVideoRef.current || !baselineVideoRef.current) return;

        const mainVideo = mainVideoRef.current;
        const baselineVideo = baselineVideoRef.current;

        // Sync playback state
        if (mainVideo.paused !== baselineVideo.paused) {
            if (mainVideo.paused) {
                baselineVideo.pause();
            } else {
                baselineVideo.play().catch(console.error);
            }
        }

        // Sync time (only if difference is significant)
        if (Math.abs(mainVideo.currentTime - baselineVideo.currentTime) > 0.1) {
            baselineVideo.currentTime = mainVideo.currentTime;
        }
    }, [compareToBaseline]);

    // Handle keyboard navigation
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (!isOpen) return;

            switch (e.key) {
                case 'Escape':
                    onClose();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    onNavigate?.('left');
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    onNavigate?.('right');
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    onNavigate?.('up');
                    break;
                case 'ArrowDown':
                    e.preventDefault();
                    onNavigate?.('down');
                    break;
                default:
                    break;
            }
        };

        if (isOpen) {
            document.addEventListener('keydown', handleKeyDown);
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = 'unset';
        }

        return () => {
            document.removeEventListener('keydown', handleKeyDown);
            document.body.style.overflow = 'unset';
        };
    }, [isOpen, onClose, onNavigate]);

    // Handle click outside
    const handleBackdropClick = useCallback((e) => {
        if (e.target === lightboxRef.current) {
            onClose();
        }
    }, [onClose]);

    // Auto-play videos when lightbox opens
    useEffect(() => {
        if (isOpen && mainVideoRef.current) {
            mainVideoRef.current.play().catch(console.error);
        }
        if (isOpen && compareToBaseline && baselineVideoRef.current) {
            baselineVideoRef.current.play().catch(console.error);
        }
    }, [isOpen, compareToBaseline]);

    // Extract attention data from cached data when lightbox opens
    useEffect(() => {
        if (!isOpen || !video || !cachedAttentionData) {
            setAttentionData(null);
            setBaselineAttentionData(null);
            return;
        }

        try {
            setLocalError(null);

            // Get prompt key
            const promptNum = video.prompt_variation?.index || 0;
            const promptKey = `prompt_${promptNum.toString().padStart(3, '0')}`;

            console.log('Extracting attention data for video:', video);
            console.log('Video ID:', video.video_id);
            console.log('Prompt key:', promptKey);

            // Check if attention videos exist
            if (!cachedAttentionData.attention_videos) {
                throw new Error('No attention videos data available in cached data');
            }

            // Get videos for this prompt
            const promptData = cachedAttentionData.attention_videos[promptKey];
            if (!promptData) {
                throw new Error(`No attention videos for ${promptKey}`);
            }

            // Use video.video_id directly - this is the correct folder ID
            const videoId = video.video_id;

            if (!videoId) {
                throw new Error('Video object missing video_id property');
            }

            console.log('Using video_id:', videoId);
            console.log('Available IDs in promptData:', Object.keys(promptData));

            // Extract attention videos for this video
            const attentionVideos = promptData[videoId];

            if (!attentionVideos) {
                throw new Error(`No attention videos available for video ID: ${videoId}`);
            }

            setAttentionData({
                videos: attentionVideos,
                tokens: Object.keys(attentionVideos).sort(),
                videoId
            });

            // Also extract baseline attention data if comparing
            if (baselineVideo && baselineVideo.video_id) {
                console.log('Extracting baseline attention data');
                console.log('Baseline video ID:', baselineVideo.video_id);

                const baselineAttentionVideos = promptData[baselineVideo.video_id];

                if (baselineAttentionVideos) {
                    console.log('Found baseline attention data!');
                    setBaselineAttentionData({
                        videos: baselineAttentionVideos,
                        tokens: Object.keys(baselineAttentionVideos).sort(),
                        videoId: baselineVideo.video_id
                    });
                } else {
                    console.warn('Could not find baseline attention data for ID:', baselineVideo.video_id);
                    setBaselineAttentionData(null);
                }
            } else {
                setBaselineAttentionData(null);
            }

        } catch (error) {
            console.error('Error extracting attention data:', error);
            setLocalError(error.message);
            setAttentionData(null);
            setBaselineAttentionData(null);
        }
    }, [isOpen, video, baselineVideo, cachedAttentionData]);

    if (!isOpen || !video) {
        return null;
    }

    return (
        <div
            ref={lightboxRef}
            className="attention-bending-lightbox"
            onClick={handleBackdropClick}
        >
            <div className="lightbox-content">
                {/* Header row with title, toggles, and close button */}
                <div className="lightbox-header">
                    <h2 className="lightbox-title">Video Details</h2>

                    <div className="lightbox-toggles">
                        <button
                            className={`toggle-button ${compareToBaseline ? 'active' : ''}`}
                            onClick={() => setCompareToBaseline(!compareToBaseline)}
                        >
                            Compare to Baseline
                        </button>
                    </div>

                    <div>
                        <button
                            className="attention-lightbox-close-btn"
                            onClick={onClose}
                            aria-label="Close lightbox"
                        >
                            ×
                        </button>
                    </div>
                </div>

                {/* Navigation buttons */}
                {onNavigate && (
                    <div className="lightbox-nav-container">
                        <button
                            className="lightbox-nav lightbox-nav-left"
                            onClick={() => onNavigate('left')}
                            aria-label="Previous seed (left)"
                        >
                            ◀
                        </button>
                        <button
                            className="lightbox-nav lightbox-nav-right"
                            onClick={() => onNavigate('right')}
                            aria-label="Next seed (right)"
                        >
                            ▶
                        </button>
                        <button
                            className="lightbox-nav lightbox-nav-up"
                            onClick={() => onNavigate('up')}
                            aria-label="Previous operation (up)"
                        >
                            ▲
                        </button>
                        <button
                            className="lightbox-nav lightbox-nav-down"
                            onClick={() => onNavigate('down')}
                            aria-label="Next operation (down)"
                        >
                            ▼
                        </button>
                    </div>
                )}

                <div className="lightbox-main-content">
                    {/* Video display */}
                    <div className={`video-container ${compareToBaseline ? 'split' : 'single'}`}>
                        {/* Baseline video on left when comparing */}
                        {compareToBaseline && baselineVideo && (
                            <div className="video-section">
                                <video
                                    ref={baselineVideoRef}
                                    className="lightbox-video"
                                    controls
                                    muted
                                    loop
                                    preload="metadata"
                                    src={getVideoUrl(baselineVideo.video_path)}
                                />
                                {/* Condensed metadata below baseline video */}
                                <div className="video-metadata-compact">
                                    <div className="metadata-line">
                                        {baselineVideo.prompt_variation?.text || 'N/A'} • Seed: {baselineVideo.seed} • Steps: {baselineVideo.steps} • CFG: {baselineVideo.cfg_scale}
                                    </div>
                                    <div className="metadata-line">
                                        Baseline (No Bending)
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Main/Variation video on right when comparing, centered when alone */}
                        <div className="video-section">
                            <video
                                ref={mainVideoRef}
                                className="lightbox-video"
                                controls
                                muted
                                loop
                                preload="metadata"
                                src={getVideoUrl(video.video_path)}
                                onTimeUpdate={syncVideos}
                            />
                            {/* Condensed metadata below video */}
                            <div className="video-metadata-compact">
                                <div className="metadata-line">
                                    {video.prompt_variation?.text || 'N/A'} • Seed: {video.seed} • Steps: {video.steps} • CFG: {video.cfg_scale}
                                </div>
                                <div className="metadata-line">
                                    {formatOperationDetails(video.bending_metadata)}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Attention Maps Section - Always show */}
                    <div className="attention-maps-section">
                        <h3>Attention Maps by Timestep</h3>
                        {attentionDataLoading && (
                            <div className="loading-state">Loading attention data...</div>
                        )}
                        {(attentionDataError || localError) && (
                            <div className="error-state">Error: {attentionDataError || localError}</div>
                        )}
                        {attentionData && (
                            <div className="attention-maps-grid">
                                {/* Show per-token comparison rows when baseline is active */}
                                {compareToBaseline && baselineVideo && baselineAttentionData ? (
                                    // Per-token baseline comparisons
                                    attentionData.tokens.map(token => {
                                        const bentTokenData = attentionData.videos[token];
                                        const baselineTokenData = baselineAttentionData.videos[token];
                                        const bentSteps = Object.keys(bentTokenData).sort();

                                        // Only show if baseline has this token too
                                        if (!baselineTokenData) return null;

                                        return (
                                            <React.Fragment key={token}>
                                                {/* Baseline row for this token */}
                                                <div className="attention-row baseline-row">
                                                    <div className="row-label">Baseline {token}</div>
                                                    <div className="row-videos">
                                                        {bentSteps.map(stepId => {
                                                            const stepData = baselineTokenData[stepId];
                                                            if (!stepData) return null;
                                                            return (
                                                                <div key={stepId} className="attention-cell">
                                                                    <video
                                                                        className="attention-video"
                                                                        src={getVideoUrl(stepData.video_path)}
                                                                        muted
                                                                        loop
                                                                        preload="metadata"
                                                                        onMouseEnter={(e) => e.target.play()}
                                                                        onMouseLeave={(e) => {
                                                                            e.target.pause();
                                                                            e.target.currentTime = 0;
                                                                        }}
                                                                    />
                                                                    <div className="step-label">
                                                                        {stepId.replace('step_', 'S')}
                                                                    </div>
                                                                </div>
                                                            );
                                                        })}
                                                    </div>
                                                </div>

                                                {/* Bent row for this token */}
                                                <div className="attention-row bent-row">
                                                    <div className="row-label">Bent {token}</div>
                                                    <div className="row-videos">
                                                        {bentSteps.map(stepId => {
                                                            const stepData = bentTokenData[stepId];
                                                            return (
                                                                <div key={stepId} className="attention-cell">
                                                                    <video
                                                                        className="attention-video"
                                                                        src={getVideoUrl(stepData.video_path)}
                                                                        muted
                                                                        loop
                                                                        preload="metadata"
                                                                        onMouseEnter={(e) => e.target.play()}
                                                                        onMouseLeave={(e) => {
                                                                            e.target.pause();
                                                                            e.target.currentTime = 0;
                                                                        }}
                                                                    />
                                                                    <div className="step-label">
                                                                        {stepId.replace('step_', 'S')}
                                                                    </div>
                                                                </div>
                                                            );
                                                        })}
                                                    </div>
                                                </div>
                                            </React.Fragment>
                                        );
                                    })
                                ) : (
                                    // No baseline comparison - show bent token rows only
                                    attentionData.tokens.map(token => {
                                        const tokenData = attentionData.videos[token];
                                        const steps = Object.keys(tokenData).sort();

                                        return (
                                            <div key={token} className="attention-row">
                                                <div className="row-label">{token}</div>
                                                <div className="row-videos">
                                                    {steps.map(stepId => {
                                                        const stepData = tokenData[stepId];
                                                        return (
                                                            <div key={stepId} className="attention-cell">
                                                                <video
                                                                    className="attention-video"
                                                                    src={getVideoUrl(stepData.video_path)}
                                                                    muted
                                                                    loop
                                                                    preload="metadata"
                                                                    onMouseEnter={(e) => e.target.play()}
                                                                    onMouseLeave={(e) => {
                                                                        e.target.pause();
                                                                        e.target.currentTime = 0;
                                                                    }}
                                                                />
                                                                <div className="step-label">
                                                                    {stepId.replace('step_', 'S')}
                                                                </div>
                                                            </div>
                                                        );
                                                    })}
                                                </div>
                                            </div>
                                        );
                                    })
                                )}
                            </div>
                        )}
                    </div>

                    {/* Navigation hint */}
                    {onNavigate && (
                        <div className="navigation-hint">
                            Use arrow keys to navigate: ← → (videos) ↑ ↓ (operations) | ESC to close
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default AttentionBendingLightbox;
