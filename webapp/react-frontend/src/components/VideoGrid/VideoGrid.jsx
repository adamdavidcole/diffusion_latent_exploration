import React, { useCallback, useRef, useEffect, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { useIntersectionObserver } from '../../hooks/useIntersectionObserver';
import { useVideoCache } from '../../hooks/useVideoCache';
import VideoCell from './VideoCell';
import VideoLightbox from './VideoLightbox';
import { draggable, dropTargetForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { extractClosestEdge, attachClosestEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';

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

    // Drag and drop state
    const [reorderedVideoGrid, setReorderedVideoGrid] = useState(null);
    const [dragState, setDragState] = useState({ isDragging: false, draggedRowIndex: null });

    // Use video cache hook
    const { cancelInflightRequests, getCacheStats } = useVideoCache();

    // Get the video grid to render (reordered or original)
    const videoGridToRender = reorderedVideoGrid || currentExperiment?.video_grid || [];

    // Reset reordered grid when experiment changes
    const resetRowOrder = useCallback(() => {
        setReorderedVideoGrid(null);
    }, []);

    // Handle drag and drop
    const handleReorderRows = useCallback((startIndex, finishIndex) => {
        const items = Array.from(videoGridToRender);
        const [reorderedItem] = items.splice(startIndex, 1);
        items.splice(finishIndex, 0, reorderedItem);
        setReorderedVideoGrid(items);
    }, [videoGridToRender]);

    // Handle lightbox open/close
    const handleOpenLightbox = useCallback((video) => {
        setLightboxVideo(video);
    }, []);

    const handleCloseLightbox = useCallback(() => {
        setLightboxVideo(null);
    }, []);

    // Handle navigation in lightbox
    const handleLightboxNavigation = useCallback((direction) => {
        if (!lightboxVideo || !videoGridToRender.length) return;

        const grid = videoGridToRender;
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
    }, [lightboxVideo, videoGridToRender]);

    // Get preview information for navigation
    const getNavigationPreview = useCallback((direction) => {
        if (!lightboxVideo || !videoGridToRender.length) return null;

        const grid = videoGridToRender;
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
    }, [lightboxVideo, videoGridToRender]);

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
        // if (state?.videoDuration === 0 && duration && actions && typeof actions.setVideoDuration === 'function') {
        //     actions.setVideoDuration(duration);
        // }
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

            // Reset row reordering
            setReorderedVideoGrid(null);
            setDragState({ isDragging: false, draggedRowIndex: null });

            // Update current experiment reference
            currentExperimentNameRef.current = newExperimentName;

            // Log cache stats for debugging
            const stats = getCacheStats();
            console.log('Cache stats after cleanup:', stats);
        }
    }, [currentExperiment?.name, disconnect, cancelInflightRequests, getCacheStats]);

    // Draggable Row Component
    const DraggableRow = ({ row, rowIndex, gapRem }) => {
        const rowRef = useRef(null);
        const [isDragging, setIsDragging] = useState(false);
        const [dragOverState, setDragOverState] = useState(null);

        useEffect(() => {
            const element = rowRef.current;
            if (!element) return;

            return combine(
                draggable({
                    element,
                    getInitialData: () => ({ rowIndex, type: 'grid-row' }),
                    onDragStart: () => {
                        setIsDragging(true);
                        setDragState({ isDragging: true, draggedRowIndex: rowIndex });
                    },
                    onDrop: () => {
                        setIsDragging(false);
                        setDragState({ isDragging: false, draggedRowIndex: null });
                    },
                }),
                dropTargetForElements({
                    element,
                    canDrop: ({ source }) => source.data.type === 'grid-row',
                    getIsSticky: () => true,
                    getData: ({ input, element }) => {
                        return attachClosestEdge({ type: 'grid-row', rowIndex }, {
                            input,
                            element,
                            allowedEdges: ['top', 'bottom'],
                        });
                    },
                    onDragEnter: ({ self, source }) => {
                        const sourceIndex = source.data.rowIndex;
                        const targetIndex = self.data.rowIndex;
                        const edge = extractClosestEdge(self.data);

                        if (sourceIndex !== targetIndex) {
                            setDragOverState({ edge, isTarget: true });
                        }
                    },
                    onDragLeave: () => {
                        setDragOverState(null);
                    },
                    onDrop: ({ self, source }) => {
                        const sourceIndex = source.data.rowIndex;
                        const targetIndex = self.data.rowIndex;
                        const edge = extractClosestEdge(self.data);

                        setDragOverState(null);

                        if (sourceIndex === targetIndex) return;

                        let finishIndex = targetIndex;
                        if (edge === 'bottom') {
                            finishIndex = targetIndex + 1;
                        }

                        // Adjust for removal of source
                        if (sourceIndex < finishIndex) {
                            finishIndex -= 1;
                        }

                        handleReorderRows(sourceIndex, finishIndex);
                    },
                })
            );
        }, [rowIndex, handleReorderRows]);

        const dragHandleStyle = {
            display: 'flex',
            alignItems: 'center',
            padding: '0 8px',
            cursor: isDragging ? 'grabbing' : 'grab',
            color: '#666',
            fontSize: '16px',
            userSelect: 'none',
        };

        const rowClasses = [
            'grid-row',
            isDragging && 'dragging',
            dragOverState?.isTarget && 'drag-over',
            dragOverState?.edge === 'top' && 'drag-over-top',
            dragOverState?.edge === 'bottom' && 'drag-over-bottom',
        ].filter(Boolean).join(' ');

        return (
            <div
                ref={rowRef}
                className={rowClasses}
                data-row-index={rowIndex}
                style={{ gap: `${gapRem}rem` }}
            >
                <div className="drag-handle" style={dragHandleStyle} title="Drag to reorder">
                    ⋮⋮
                </div>
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
        );
    };

    if (!currentExperiment) {
        return (
            <div className="empty-state">
                <h3>No experiment selected</h3>
                <p>Select an experiment from the sidebar to view videos.</p>
            </div>
        );
    }

    if (videoGridToRender.length === 0) {
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
                <div className="experiment-title-row">
                    <h2 id="experiment-title">{currentExperiment.name}</h2>
                    {reorderedVideoGrid && (
                        <button
                            className="reset-order-btn"
                            onClick={resetRowOrder}
                            title="Reset row order to original"
                        >
                            Reset Order
                        </button>
                    )}
                </div>
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
                <div className="drag-handle-spacer"></div>
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
                {videoGridToRender.map((row, rowIndex) => (
                    <DraggableRow
                        key={`row-${row.variation}-${rowIndex}`}
                        row={row}
                        rowIndex={rowIndex}
                        gapRem={gapRem}
                    />
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
