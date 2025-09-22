import React, { useState, useEffect, useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import VideoCell from '../VideoGrid/VideoCell';
import './FullSearchModal.css';

const FullSearchModal = () => {
    const { state, actions } = useApp();
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [videoSize, setVideoSize] = useState(220); // Increased default from 180 to 220

    // Search across all experiments for matching rows
    const performSearch = useMemo(() => {
        if (!searchQuery || searchQuery.length < 3) {
            return [];
        }

        const results = [];
        const query = searchQuery.toLowerCase();

        // Get all experiments from the tree
        const getAllExperiments = (node) => {
            const experiments = [];
            if (node.type === 'experiment') {
                experiments.push(node);
            } else if (node.children) {
                node.children.forEach(child => {
                    experiments.push(...getAllExperiments(child));
                });
            }
            return experiments;
        };

        const allExperiments = state.experimentsTree ? getAllExperiments(state.experimentsTree) : [];

        // Search through each experiment's video grid
        allExperiments.forEach((experiment) => {
            // Try to access video_grid from both locations
            let videoGrid = experiment.video_grid;
            if (!videoGrid && experiment.experiment_data) {
                videoGrid = experiment.experiment_data.video_grid;
            }
            
            if (videoGrid && videoGrid.length > 0) {
                videoGrid.forEach((row) => {
                    // Search in multiple possible text fields
                    const variation = row.variation || '';
                    const prompt = row.prompt || '';
                    const fullPrompt = row.full_prompt || '';
                    
                    if (variation.toLowerCase().includes(query) || 
                        prompt.toLowerCase().includes(query) || 
                        fullPrompt.toLowerCase().includes(query)) {
                        results.push({
                            ...row,
                            experimentName: experiment.name || experiment.experiment_data?.name,
                            experimentPath: experiment.path,
                            basePrompt: experiment.base_prompt || experiment.experiment_data?.base_prompt,
                            seeds: experiment.seeds || experiment.experiment_data?.seeds || []
                        });
                    }
                });
            }
        });

        // Limit to 120 results
        return results.slice(0, 120);
    }, [searchQuery, state.experimentsTree]);

    // Update search results when query changes
    useEffect(() => {
        setSearchResults(performSearch);
    }, [performSearch]);

    // Initialize search query from context if provided
    useEffect(() => {
        if (state.fullSearchQuery) {
            setSearchQuery(state.fullSearchQuery);
            // Clear the query from context after using it
            actions.setFullSearchQuery('');
        }
    }, [state.fullSearchQuery, actions]);

    if (!state.showFullSearch) {
        return null;
    }

    const handleSearchChange = (e) => {
        setSearchQuery(e.target.value);
    };

    const handleExperimentClick = (experimentPath) => {
        // Close the modal and navigate to the experiment
        actions.toggleFullSearch();
        actions.setCurrentExperiment(experimentPath);
    };

    return (
        <div className="full-search-modal">
            <div className="modal-overlay" onClick={actions.toggleFullSearch} />
            <div className="modal-content">
                <div className="modal-header">
                    <h2>Search Across All Experiments</h2>
                    <button 
                        className="close-btn"
                        onClick={actions.toggleFullSearch}
                        aria-label="Close"
                    >
                        ×
                    </button>
                </div>

                <div className="search-section">
                    <input
                        type="text"
                        className="search-input"
                        placeholder="Search for prompts across all experiments (min 3 characters)..."
                        value={searchQuery}
                        onChange={handleSearchChange}
                        autoFocus
                    />
                    <div className="search-controls">
                        <div className="search-info">
                            {searchQuery.length < 3 && searchQuery.length > 0 && (
                                <span className="search-hint">Enter at least 3 characters to search</span>
                            )}
                            {searchResults.length > 0 && (
                                <span className="results-count">
                                    Found {searchResults.length} results{searchResults.length === 120 ? ' (limited to 120)' : ''}
                                </span>
                            )}
                            {searchQuery.length >= 3 && searchResults.length === 0 && (
                                <span className="no-results">No matching prompts found</span>
                            )}
                        </div>
                        <div className="video-size-control">
                            <label className="video-size-label">
                                Video Size: {videoSize}px
                            </label>
                            <input
                                type="range"
                                min="120"
                                max="300"
                                step="20"
                                value={videoSize}
                                onChange={(e) => setVideoSize(parseInt(e.target.value))}
                                className="video-size-slider"
                                title={`Video size: ${videoSize}px`}
                            />
                        </div>
                    </div>
                </div>

                <div className="results-section">
                    {searchResults.map((row, index) => (
                        <div key={`${row.experimentPath}-${row.variation_num}-${index}`} className="result-row">
                            <div className="row-header">
                                <button 
                                    className="experiment-link"
                                    onClick={() => handleExperimentClick(row.experimentPath)}
                                    title={`Go to experiment: ${row.experimentName}`}
                                >
                                    {row.experimentName}
                                </button>
                                <span className="prompt-separator">—</span>
                                <div className="prompt-text" title={row.variation}>
                                    {row.variation}
                                </div>
                            </div>
                            <div className="videos-row">
                                {row.videos && row.videos.map((video, videoIndex) => (
                                    <VideoCell
                                        key={`${row.experimentPath}-${row.variation_num}-${video.seed}-${videoIndex}`}
                                        video={video}
                                        videoSize={videoSize} // Use dynamic video size
                                        onVideoLoaded={() => {}} // No-op for now
                                        onMetadataLoaded={() => {}} // No-op for now
                                        onOpenLightbox={() => {}} // Disabled for now
                                        disableLightbox={true}
                                    />
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default FullSearchModal;