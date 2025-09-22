import React, { useState, useEffect, useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import VideoCell from '../VideoGrid/VideoCell';
import './FullSearchModal.css';

const FullSearchModal = () => {
    const { state, actions } = useApp();
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState([]);
    const [isLoading, setIsLoading] = useState(false);

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
        
        console.log('🔍 Full Search Debug:');
        console.log('Query:', query);
        console.log('Total experiments found in tree:', allExperiments.length);
        console.log('Experiments tree structure:', state.experimentsTree);

        // Search through each experiment's video grid
        allExperiments.forEach((experiment, expIndex) => {
            console.log(`\n--- Experiment ${expIndex + 1}/${allExperiments.length}: ${experiment.name} ---`);
            console.log('Experiment object keys:', Object.keys(experiment));
            console.log('Has video_grid?', !!experiment.video_grid);
            console.log('Has experiment_data?', !!experiment.experiment_data);
            
            // Try to access video_grid from both locations
            let videoGrid = experiment.video_grid;
            if (!videoGrid && experiment.experiment_data) {
                console.log('Checking experiment_data for video_grid...');
                console.log('experiment_data keys:', Object.keys(experiment.experiment_data));
                videoGrid = experiment.experiment_data.video_grid;
            }
            
            console.log('Video grid found:', !!videoGrid);
            if (videoGrid) {
                console.log('Video grid length:', videoGrid.length);
                console.log('First row structure:', videoGrid[0] ? Object.keys(videoGrid[0]) : 'No rows');
                
                videoGrid.forEach((row, rowIndex) => {
                    console.log(`Row ${rowIndex + 1}:`, {
                        variation: row.variation,
                        prompt: row.prompt,
                        full_prompt: row.full_prompt,
                        hasVariation: !!row.variation,
                        hasPrompt: !!row.prompt,
                        hasFullPrompt: !!row.full_prompt
                    });
                    
                    // Search in multiple possible text fields
                    const variation = row.variation || '';
                    const prompt = row.prompt || '';
                    const fullPrompt = row.full_prompt || '';
                    
                    console.log(`Searching in row ${rowIndex + 1}:`, {
                        variation: variation.substring(0, 50) + '...',
                        prompt: prompt.substring(0, 50) + '...',
                        fullPrompt: fullPrompt.substring(0, 50) + '...'
                    });
                    
                    if (variation.toLowerCase().includes(query) || 
                        prompt.toLowerCase().includes(query) || 
                        fullPrompt.toLowerCase().includes(query)) {
                        console.log(`✅ MATCH found in row ${rowIndex + 1}!`);
                        results.push({
                            ...row,
                            experimentName: experiment.name || experiment.experiment_data?.name,
                            experimentPath: experiment.path,
                            basePrompt: experiment.base_prompt || experiment.experiment_data?.base_prompt,
                            seeds: experiment.seeds || experiment.experiment_data?.seeds || []
                        });
                    } else {
                        console.log(`❌ No match in row ${rowIndex + 1}`);
                    }
                });
            } else {
                console.log('❌ No video_grid found for this experiment');
            }
        });

        console.log(`\n🎯 Search completed. Found ${results.length} total matches.`);
        console.log('First few results:', results.slice(0, 3));

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
                </div>

                <div className="results-section">
                    {searchResults.map((row, index) => (
                        <div key={`${row.experimentPath}-${row.variation_num}-${index}`} className="result-row">
                            <div className="row-header">
                                <div className="experiment-info">
                                    <button 
                                        className="experiment-link"
                                        onClick={() => handleExperimentClick(row.experimentPath)}
                                        title={`Go to experiment: ${row.experimentName}`}
                                    >
                                        {row.experimentName}
                                    </button>
                                </div>
                                <div className="prompt-text" title={row.variation}>
                                    {row.variation}
                                </div>
                            </div>
                            <div className="videos-row">
                                {row.videos && row.videos.map((video, videoIndex) => (
                                    <VideoCell
                                        key={`${row.experimentPath}-${row.variation_num}-${video.seed}-${videoIndex}`}
                                        video={video}
                                        videoSize={120} // Fixed size for search results
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