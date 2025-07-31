import React, { useRef, useEffect, useCallback } from 'react';

const VideoLightbox = ({ video, isOpen, onClose, onNavigate, getPreviewInfo }) => {
    const videoRef = useRef(null);
    const lightboxRef = useRef(null);

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

    // Auto-play video when lightbox opens
    useEffect(() => {
        const videoElement = videoRef.current;
        if (isOpen && videoElement && video?.video_path) {
            // Set video source
            const videoUrl = `/media/${video.video_path}`;
            if (videoElement.src !== videoUrl) {
                videoElement.src = videoUrl;
            }

            // Play video
            videoElement.play().catch(console.error);
        }
    }, [isOpen, video?.video_path]);

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
                    <>
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
                    </>
                )}

                <div className="lightbox-video-container">
                    <video
                        ref={videoRef}
                        className="lightbox-video"
                        controls
                        muted
                        loop
                        preload="metadata"
                    />
                </div>

                <div className="lightbox-info">
                    <div className="video-details">
                        <span><strong>Variation:</strong> {video.variation || 'Unknown'}</span>
                        <span><strong>Seed:</strong> {video.seed}</span>
                        <span><strong>Resolution:</strong> {video.width}×{video.height}</span>
                        <span><strong>Frames:</strong> {video.num_frames}</span>
                        <span><strong>Steps:</strong> {video.steps}</span>
                        <span><strong>CFG Scale:</strong> {video.cfg_scale}</span>
                    </div>
                    {onNavigate && (
                        <div className="lightbox-controls-hint">
                            <span>Use arrow keys or buttons to navigate: ← → (videos) ↑ ↓ (variations) | ESC to close</span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default VideoLightbox;
