import React, { useRef, useEffect, useCallback, useState } from 'react';
import { getVideoUrl } from '../../services/api';
import { useApp } from '../../context/AppContext';
import VideoLightboxAnalysis from './VideoLightboxAnalysis';

const VideoLightbox = ({ video, isOpen, onClose, onNavigate, getPreviewInfo }) => {
    const videoRef = useRef(null);
    const lightboxRef = useRef(null);
    const { state, actions } = useApp();
    const [lightboxAttentionMode, setLightboxAttentionMode] = useState(false);
    const [lightboxSelectedToken, setLightboxSelectedToken] = useState(null);

    // Get the current video source based on attention mode (either global or lightbox-specific)
    const getCurrentVideoPath = useCallback(() => {
        if (!video?.video_path) return null;

        const attentionMode = lightboxAttentionMode || state.attentionMode;
        const selectedToken = lightboxSelectedToken || state.selectedToken;

        // If attention mode is off, return normal video
        if (!attentionMode || !selectedToken || !state.currentExperiment?.attention_videos?.available) {
            return video.video_path;
        }

        // Try to find matching attention video
        const attentionVideos = state.currentExperiment.attention_videos;
        const promptNum = video.variation_num;
        const videoNum = video.video_number;

        const promptKey = `prompt_${promptNum.toString().padStart(3, '0')}`;
        const videoKey = `vid${videoNum.toString().padStart(3, '0')}`;

        const promptData = attentionVideos.prompts[promptKey];
        if (promptData && promptData.videos[videoKey]) {
            const tokenData = promptData.videos[videoKey].tokens[selectedToken];
            if (tokenData && tokenData.aggregate_overlay_path) {
                return tokenData.aggregate_overlay_path;
            }
        }

        // Fallback to normal video if attention video not found
        return video.video_path;
    }, [video, lightboxAttentionMode, lightboxSelectedToken, state.attentionMode, state.selectedToken, state.currentExperiment?.attention_videos]);

    // Initialize lightbox attention state from global state when opening
    useEffect(() => {
        if (isOpen) {
            setLightboxAttentionMode(state.attentionMode);
            setLightboxSelectedToken(state.selectedToken);
        }
    }, [isOpen, state.attentionMode, state.selectedToken]);

    // Get preview information for navigation
    const getNavigationPreview = (direction) => {
        if (!onNavigate || !getPreviewInfo) return null;
        return getPreviewInfo(direction);
    };

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
            document.body.style.overflow = 'hidden'; // Prevent background scrolling
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

    // Auto-play video when lightbox opens or video source changes
    useEffect(() => {
        const videoElement = videoRef.current;
        if (isOpen && videoElement) {
            const currentVideoPath = getCurrentVideoPath();
            if (currentVideoPath) {
                // Set video source
                const videoUrl = getVideoUrl(currentVideoPath);
                if (videoElement.src !== videoUrl) {
                    videoElement.src = videoUrl;
                }

                // Play video
                videoElement.play().catch(console.error);
            }
        }
    }, [isOpen, getCurrentVideoPath]);

    // Clean up when closing
    useEffect(() => {
        const videoElement = videoRef.current;
        if (!isOpen && videoElement) {
            videoElement.pause();
            videoElement.currentTime = 0;
        }
    }, [isOpen]);

    if (!isOpen || !video) {
        return null;
    }

    return (
        <div
            ref={lightboxRef}
            className="video-lightbox"
            onClick={handleBackdropClick}
        >
            <div className="lightbox-content">
                <button
                    className="lightbox-close"
                    onClick={onClose}
                    aria-label="Close lightbox"
                >
                    ×
                </button>

                {/* Navigation buttons */}
                {onNavigate && (
                    <div className="lightbox-nav-container">
                        <button
                            className="lightbox-nav lightbox-nav-left"
                            onClick={() => onNavigate('left')}
                            aria-label="Previous video (left)"
                        >
                            ◀
                            {getPreviewInfo && (
                                <div className="lightbox-nav-preview">
                                    {getNavigationPreview('left') || 'Previous seed'}
                                </div>
                            )}
                        </button>
                        <button
                            className="lightbox-nav lightbox-nav-right"
                            onClick={() => onNavigate('right')}
                            aria-label="Next video (right)"
                        >
                            ▶
                            {getPreviewInfo && (
                                <div className="lightbox-nav-preview">
                                    {getNavigationPreview('right') || 'Next seed'}
                                </div>
                            )}
                        </button>
                        <button
                            className="lightbox-nav lightbox-nav-up"
                            onClick={() => onNavigate('up')}
                            aria-label="Previous variation (up)"
                        >
                            ▲
                            {getPreviewInfo && (
                                <div className="lightbox-nav-preview">
                                    {getNavigationPreview('up') || 'Previous variation'}
                                </div>
                            )}
                        </button>
                        <button
                            className="lightbox-nav lightbox-nav-down"
                            onClick={() => onNavigate('down')}
                            aria-label="Next variation (down)"
                        >
                            ▼
                            {getPreviewInfo && (
                                <div className="lightbox-nav-preview">
                                    {getNavigationPreview('down') || 'Next variation'}
                                </div>
                            )}
                        </button>
                    </div>
                )}

                <div className="lightbox-video-and-details-container">
                    <div className="lightbox-video-container">
                        {(lightboxAttentionMode || state.attentionMode) && (lightboxSelectedToken || state.selectedToken) && (
                            <div className="attention-video-indicator">
                                🎯 Attention: {lightboxSelectedToken || state.selectedToken}
                            </div>
                        )}
                        <video
                            ref={videoRef}
                            className="lightbox-video"
                            controls
                            muted
                            loop
                            preload="metadata"
                        />
                    </div>

                    {/* Attention Controls for Lightbox */}
                    {state.currentExperiment?.attention_videos?.available && state.availableTokens?.length > 0 && (
                        <div className="lightbox-attention-controls">
                            <div className="attention-toggle">
                                <label>
                                    <input
                                        type="checkbox"
                                        checked={lightboxAttentionMode}
                                        onChange={(e) => {
                                            setLightboxAttentionMode(e.target.checked);
                                            if (e.target.checked && !lightboxSelectedToken && state.availableTokens?.length > 0) {
                                                setLightboxSelectedToken(state.availableTokens[0]);
                                            }
                                        }}
                                    />
                                    🎯 Attention Mode
                                </label>
                            </div>

                            {lightboxAttentionMode && (
                                <div className="token-selector">
                                    <label>Focus Token:</label>
                                    <div className="token-buttons">
                                        {state.availableTokens.map(token => (
                                            <button
                                                key={token}
                                                className={`token-button ${lightboxSelectedToken === token ? 'selected' : ''}`}
                                                onClick={() => setLightboxSelectedToken(token)}
                                                title={`Show attention for "${token}"`}
                                            >
                                                {token}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    <div className="lightbox-info">
                        <div className="video-details">
                            <span><strong>Variation:</strong> {video.variation || 'Unknown'}</span>
                            <span><strong>Seed:</strong> {video.seed}</span>
                            <span><strong>Resolution:</strong> {video.width}×{video.height}</span>
                            <span><strong>Frames:</strong> {video.num_frames}</span>
                            <span><strong>Steps:</strong> {video.steps}</span>
                            <span><strong>CFG Scale:</strong> {video.cfg_scale}</span>
                            {(lightboxAttentionMode || state.attentionMode) && (lightboxSelectedToken || state.selectedToken) && (
                                <span><strong>Attention Token:</strong> {lightboxSelectedToken || state.selectedToken}</span>
                            )}
                        </div>
                        {onNavigate && (
                            <div className="lightbox-controls-hint">
                                <span>Use arrow keys or buttons to navigate: ← → (videos) ↑ ↓ (variations) | ESC to close</span>
                            </div>
                        )}
                    </div>
                </div>

                {/* VLM Analysis Section */}
                <VideoLightboxAnalysis video={video} />
            </div>
        </div>
    );
};

export default VideoLightbox;
