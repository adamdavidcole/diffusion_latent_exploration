import React, { useCallback, useRef, useEffect, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { useIntersectionObserver } from '../../hooks/useIntersectionObserver';
import { useVideoCache } from '../../hooks/useVideoCache';
import VideoCell from './VideoCell';
import VideoLightbox from './VideoLightbox';
import { draggable, dropTargetForElements } from '@atlaskit/pragmatic-drag-and-drop/element/adapter';
import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { extractClosestEdge, attachClosestEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';
import { getVariationTextFromPromptKey } from '../../utils/variationText';

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

    // Calculate similarity-sorted grid based on user selection
    const getSimilaritySortedGrid = useCallback(() => {
        if (!currentExperiment?.video_grid || !state.similarityAnalysis || state.similaritySortBy === 'default') {
            return currentExperiment?.video_grid || [];
        }

        const { rankings, baseline_prompt } = state.similarityAnalysis;
        const originalGrid = currentExperiment.video_grid;

        // Find baseline row and separate it from others
        let baselineRow = null;
        let otherRows = [];

        originalGrid.forEach(row => {
            // Match row to prompt using variation number or prompt key
            const promptKey = row.prompt_key || `prompt_${String(row.variation_num).padStart(3, '0')}`;
            
            if (promptKey === baseline_prompt) {
                baselineRow = row;
            } else {
                otherRows.push(row);
            }
        });

        // Sort other rows based on selected similarity metric
        if (rankings && rankings.final_scores) {
            otherRows.sort((a, b) => {
                const aPromptKey = a.prompt_key || `prompt_${String(a.variation_num).padStart(3, '0')}`;
                const bPromptKey = b.prompt_key || `prompt_${String(b.variation_num).padStart(3, '0')}`;
                
                const aScore = rankings.final_scores[aPromptKey];
                const bScore = rankings.final_scores[bPromptKey];

                if (!aScore || !bScore) return 0;

                let aValue, bValue;

                if (state.similaritySortBy === 'weighted_similarity_distance') {
                    aValue = aScore.weighted_similarity_distance;
                    bValue = bScore.weighted_similarity_distance;
                } else {
                    // Individual metric - use z-score
                    const metricKey = `${state.similaritySortBy}_distance`;
                    aValue = aScore.individual_z_scores[metricKey];
                    bValue = bScore.individual_z_scores[metricKey];
                }

                if (aValue === undefined || bValue === undefined) return 0;

                // Sort ascending (most similar first - lowest distance values)
                return aValue - bValue;
            });
        }

        // Return grid with baseline first, then sorted others
        return baselineRow ? [baselineRow, ...otherRows] : otherRows;
    }, [currentExperiment?.video_grid, state.similarityAnalysis, state.similaritySortBy]);

    // Get the video grid to render (manual reorder takes precedence, then similarity sort, then original)
    const videoGridToRender = reorderedVideoGrid || getSimilaritySortedGrid();

    // Reset reordered grid when experiment changes
    const resetRowOrder = useCallback(() => {
        setReorderedVideoGrid(null);
    }, []);

    // Reset manual reordering when similarity sort changes
    useEffect(() => {
        setReorderedVideoGrid(null);
    }, [state.similaritySortBy]);

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

    // Helper function to extract just the variation part from the full prompt
    const extractVariationFromPrompt = useCallback((fullPrompt, basePrompt) => {
        if (!basePrompt || !fullPrompt) return fullPrompt;
        
        console.log('Extracting variation from:', { fullPrompt, basePrompt });
        
        // Look for patterns like [variation] in base prompt
        const bracketMatch = basePrompt.match(/\[(.*?)\]/);
        if (bracketMatch) {
            console.log('Found bracket pattern:', bracketMatch);
            // Base prompt has [placeholder], find what replaced it
            const placeholder = bracketMatch[0]; // e.g., "[...] family"
            const beforePlaceholder = basePrompt.split(placeholder)[0];
            const afterPlaceholder = basePrompt.split(placeholder)[1];
            
            console.log('Placeholder parts:', { placeholder, beforePlaceholder, afterPlaceholder });
            
            // Extract the variation by finding what's between the before/after parts
            const beforeIndex = fullPrompt.indexOf(beforePlaceholder);
            const afterIndex = fullPrompt.lastIndexOf(afterPlaceholder);
            
            if (beforeIndex !== -1 && afterIndex !== -1) {
                const startIndex = beforeIndex + beforePlaceholder.length;
                const variation = fullPrompt.substring(startIndex, afterIndex).trim();
                console.log('Extracted variation:', variation);
                
                // Handle empty variation case
                if (variation === '') {
                    return '[empty]';
                }
                
                return variation || fullPrompt;
            }
        }
        
        // Fallback: try to find differences by comparing word by word
        const baseWords = basePrompt.toLowerCase().split(/\s+/);
        const fullWords = fullPrompt.toLowerCase().split(/\s+/);
        
        // Find the differing parts
        const variations = [];
        fullWords.forEach((word, index) => {
            if (baseWords[index] && baseWords[index] !== word) {
                variations.push(fullPrompt.split(/\s+/)[index]); // Keep original case
            } else if (!baseWords[index]) {
                variations.push(fullPrompt.split(/\s+/)[index]); // Additional words
            }
        });
        
        return variations.length > 0 ? variations.join(' ') : fullPrompt;
    }, []);

    // Helper function to get display text for variation with intelligent truncation
    const getVariationDisplayText = useCallback((row, maxLength = 40) => {
        const fullText = row.variation || '';
        const basePrompt = currentExperiment?.base_prompt || '';
        
        // Extract just the variation part
        const variationOnly = extractVariationFromPrompt(fullText, basePrompt);
        
        // Handle special cases
        if (variationOnly === '[empty]' || variationOnly.length <= maxLength) {
            return { display: variationOnly, full: fullText };
        }
        
        // If variation is still too long, truncate it
        const truncated = variationOnly.substring(0, maxLength - 3) + '...';
        return { display: truncated, full: fullText };
    }, [currentExperiment?.base_prompt, extractVariationFromPrompt]);

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
                return getVariationDisplayText(grid[newRowIndex]).display;
            case 'down':
                newRowIndex = currentRowIndex + 1;
                if (newRowIndex >= grid.length) {
                    newRowIndex = 0;
                }
                if (newVideoIndex >= grid[newRowIndex].videos.length) {
                    newVideoIndex = grid[newRowIndex].videos.length - 1;
                }
                return getVariationDisplayText(grid[newRowIndex]).display;
            default:
                return null;
        }
    }, [lightboxVideo, videoGridToRender, getVariationDisplayText]);

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
                <div 
                    className={`row-label ${showLabels ? '' : 'hidden'}`}
                    title={getVariationDisplayText(row).full}
                >
                    {getVariationDisplayText(row).display}
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
            {/* Reset Order Button (if needed) */}
            {reorderedVideoGrid && (
                <div className="grid-controls">
                    <button
                        className="reset-order-btn"
                        onClick={resetRowOrder}
                        title="Reset row order to original"
                    >
                        Reset Order
                    </button>
                </div>
            )}

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
