import React, { useRef, useEffect, useCallback } from 'react';

const VideoLightbox = ({ video, isOpen, onClose }) => {
    const videoRef = useRef(null);
    const lightboxRef = useRef(null);

    // Handle ESC key press
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'Escape' && isOpen) {
                onClose();
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
    }, [isOpen, onClose]);

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
            const videoUrl = `/api/video/${video.video_path}`;
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
                </div>
            </div>
        </div>
    );
};

export default VideoLightbox;
