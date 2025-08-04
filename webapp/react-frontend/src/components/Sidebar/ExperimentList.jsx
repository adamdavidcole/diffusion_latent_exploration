import React, { useEffect, useCallback, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import ExperimentItem from './ExperimentItem';

const ExperimentList = ({ onRescan }) => {
    const navigate = useNavigate();
    const { state, actions } = useApp();
    const { experiments, currentExperiment, isLoading, error } = state;
    const [searchTerm, setSearchTerm] = useState('');
    const [modelFilter, setModelFilter] = useState('all'); // 'all', '14b', '1.3b'
    const [minDuration, setMinDuration] = useState(0); // Minimum duration filter in seconds

    const handleSelectExperiment = useCallback((experimentName) => {
        // Navigate to the experiment URL instead of loading directly
        navigate(`/experiment/${experimentName}`);
    }, [navigate]);

    const handleRescan = useCallback(async () => {
        try {
            actions.setLoading(true);
            await api.scanExperiments();
            const experimentsData = await api.getExperiments();
            actions.setExperiments(experimentsData);
        } catch (error) {
            console.error('Error rescanning:', error);
            actions.setError('Failed to rescan experiments.');
        } finally {
            actions.setLoading(false);
        }
    }, [actions]);

    // Expose handleRescan to parent
    useEffect(() => {
        if (onRescan) {
            onRescan(handleRescan);
        }
    }, [onRescan, handleRescan]);

    // Filter experiments based on search term and model filter
    const filteredExperiments = useMemo(() => {
        let filtered = experiments;

        // Apply model filter
        if (modelFilter !== 'all') {
            filtered = filtered.filter(experiment => {
                const modelId = experiment.model_id.toLowerCase();
                if (modelFilter === '14b') {
                    return modelId.includes('14b') || modelId.includes('2.1');
                } else if (modelFilter === '1.3b') {
                    return modelId.includes('1.3b');
                }
                return true;
            });
        }

        // Apply duration filter
        if (minDuration > 0) {
            filtered = filtered.filter(experiment => {
                const duration = experiment.duration_seconds || 0;
                return duration >= minDuration;
            });
        }

        // Apply search filter
        if (searchTerm.trim()) {
            const term = searchTerm.toLowerCase();
            filtered = filtered.filter(experiment =>
                experiment.name.toLowerCase().includes(term) ||
                experiment.base_prompt.toLowerCase().includes(term) ||
                experiment.model_id.toLowerCase().includes(term)
            );
        }

        return filtered;
    }, [experiments, searchTerm, modelFilter, minDuration]);

    // Calculate maximum duration for slider range
    const maxDuration = useMemo(() => {
        if (experiments.length === 0) return 10;
        const durations = experiments
            .map(exp => exp.duration_seconds || 0)
            .filter(duration => duration > 0);
        return durations.length > 0 ? Math.ceil(Math.max(...durations)) : 10;
    }, [experiments]);

    const handleSearchChange = useCallback((e) => {
        setSearchTerm(e.target.value);
    }, []);

    const clearSearch = useCallback(() => {
        setSearchTerm('');
    }, []);

    const handleModelFilterChange = useCallback((filter) => {
        setModelFilter(filter);
    }, []);

    if (error && experiments.length === 0) {
        return (
            <div className="empty-state">
                <h3>Error loading experiments</h3>
                <p>{error}</p>
                <button onClick={handleRescan} disabled={isLoading}>
                    {isLoading ? 'Rescanning...' : 'Retry'}
                </button>
            </div>
        );
    }

    if (experiments.length === 0 && isLoading) {
        return (
            <div className="experiments-list">
                <div className="sidebar-skeleton">
                    {[...Array(4)].map((_, i) => (
                        <div key={i} className="skeleton-experiment-item">
                            <div className="skeleton-experiment-name"></div>
                            <div className="skeleton-experiment-details"></div>
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    if (experiments.length === 0 && !isLoading) {
        return (
            <div className="empty-state">
                <h3>No experiments found</h3>
                <p>Generate some videos first, then rescan.</p>
                <button onClick={handleRescan} disabled={isLoading}>
                    {isLoading ? 'Rescanning...' : 'Rescan'}
                </button>
            </div>
        );
    }

    return (
        <div className="experiments-list">
            {/* Search Filter */}
            <div className="search-container">
                <div className="search-input-wrapper">
                    <input
                        type="text"
                        placeholder="Search experiments..."
                        value={searchTerm}
                        onChange={handleSearchChange}
                        className="search-input"
                    />
                    {searchTerm && (
                        <button
                            onClick={clearSearch}
                            className="clear-search-btn"
                            title="Clear search"
                        >
                            ×
                        </button>
                    )}
                </div>

                {/* Model Filter Toggles */}
                <div className="model-filter-container">
                    <button
                        className={`model-filter-btn ${modelFilter === 'all' ? 'active' : ''}`}
                        onClick={() => handleModelFilterChange('all')}
                    >
                        All
                    </button>
                    <button
                        className={`model-filter-btn ${modelFilter === '14b' ? 'active' : ''}`}
                        onClick={() => handleModelFilterChange('14b')}
                    >
                        14B
                    </button>
                    <button
                        className={`model-filter-btn ${modelFilter === '1.3b' ? 'active' : ''}`}
                        onClick={() => handleModelFilterChange('1.3b')}
                    >
                        1.3B
                    </button>
                </div>

                {/* Duration Filter */}
                <div className="duration-filter">
                    <label className="duration-label">
                        Min Duration: {minDuration}s
                    </label>
                    <div className="duration-slider-container">
                        <span className="duration-range-start">0s</span>
                        <input
                            type="range"
                            min="0"
                            max={maxDuration}
                            step="0.5"
                            value={minDuration}
                            onChange={(e) => setMinDuration(parseFloat(e.target.value))}
                            className="duration-slider"
                            title={`Filter experiments with at least ${minDuration}s duration`}
                        />
                        <span className="duration-range-end">{maxDuration}s</span>
                    </div>
                </div>

                {(searchTerm || modelFilter !== 'all' || minDuration > 0) && (
                    <div className="search-results-info">
                        {filteredExperiments.length} of {experiments.length} experiments
                        {modelFilter !== 'all' && ` (${modelFilter.toUpperCase()} model)`}
                        {minDuration > 0 && ` (≥${minDuration}s duration)`}
                    </div>
                )}
            </div>

            {/* Experiments List */}
            {filteredExperiments.length === 0 && (searchTerm || modelFilter !== 'all' || minDuration > 0) ? (
                <div className="empty-state">
                    <h3>No experiments found</h3>
                    <p>
                        No experiments match
                        {searchTerm && ` "${searchTerm}"`}
                        {searchTerm && (modelFilter !== 'all' || minDuration > 0) && ' and'}
                        {modelFilter !== 'all' && ` ${modelFilter.toUpperCase()} model`}
                        {modelFilter !== 'all' && minDuration > 0 && ' and'}
                        {minDuration > 0 && ` ≥${minDuration}s duration`}
                    </p>
                    <div className="filter-clear-buttons">
                        {searchTerm && (
                            <button onClick={clearSearch} className="clear-search-btn-large">
                                Clear search
                            </button>
                        )}
                        {modelFilter !== 'all' && (
                            <button
                                onClick={() => handleModelFilterChange('all')}
                                className="clear-search-btn-large"
                            >
                                Show all models
                            </button>
                        )}
                        {minDuration > 0 && (
                            <button
                                onClick={() => setMinDuration(0)}
                                className="clear-search-btn-large"
                            >
                                Clear duration filter
                            </button>
                        )}
                    </div>
                </div>
            ) : (
                filteredExperiments
                    .sort((a, b) => {
                        // Sort by creation timestamp, newest first
                        // Use the created_timestamp field from the API (Unix timestamp)
                        const timestampA = a.created_timestamp || 0;
                        const timestampB = b.created_timestamp || 0;

                        return timestampB - timestampA; // Newest first (descending)
                    })
                    .map(experiment => (
                        <ExperimentItem
                            key={experiment.name}
                            experiment={experiment}
                            isActive={currentExperiment?.name === experiment.name}
                            onSelect={handleSelectExperiment}
                        />
                    ))
            )}
        </div>
    );
};

export default ExperimentList;
