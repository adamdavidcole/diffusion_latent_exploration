import React, { useEffect, useCallback, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import ExperimentItem from './ExperimentItem';

const TreeNode = ({ node, level = 0, onSelect, currentExperiment, searchTerm, modelFilter }) => {
    const [isExpanded, setIsExpanded] = useState(level < 2); // Auto-expand first two levels
    const navigate = useNavigate();

    const toggleExpanded = useCallback(() => {
        setIsExpanded(prev => !prev);
    }, []);

    const handleExperimentSelect = useCallback((experimentPath) => {
        navigate(`/experiment/${experimentPath}`);
    }, [navigate]);

    // For folder nodes
    if (node.type === 'folder') {
        const hasVisibleChildren = node.children?.some(child => 
            child.type === 'experiment' ? 
                isExperimentVisible(child.experiment_data, searchTerm, modelFilter) :
                hasVisibleExperiments(child, searchTerm, modelFilter)
        );

        if (!hasVisibleChildren) return null;

        return (
            <div className="tree-folder" style={{ marginLeft: `${level * 12}px` }}>
                <div 
                    className="folder-header"
                    onClick={toggleExpanded}
                >
                    <span className={`folder-icon ${isExpanded ? 'expanded' : ''}`}>
                        {isExpanded ? '▼' : '▶'}
                    </span>
                    <span className="folder-name">{node.name}</span>
                    <span className="folder-count">
                        ({getExperimentCount(node)})
                    </span>
                </div>
                {isExpanded && (
                    <div className="folder-children">
                        {node.children
                            .sort((a, b) => {
                                // Folders first, then by creation time
                                if (a.type !== b.type) {
                                    return a.type === 'folder' ? -1 : 1;
                                }
                                return (b.created_timestamp || 0) - (a.created_timestamp || 0);
                            })
                            .map(child => (
                                <TreeNode
                                    key={child.path}
                                    node={child}
                                    level={level + 1}
                                    onSelect={onSelect}
                                    currentExperiment={currentExperiment}
                                    searchTerm={searchTerm}
                                    modelFilter={modelFilter}
                                />
                            ))}
                    </div>
                )}
            </div>
        );
    }

    // For experiment nodes
    if (node.type === 'experiment') {
        if (!isExperimentVisible(node.experiment_data, searchTerm, modelFilter)) {
            return null;
        }

        const experimentPath = node.path.replace(/^outputs\//, '');
        
        return (
            <div style={{ marginLeft: `${level * 12}px` }}>
                <ExperimentItem
                    experiment={node.experiment_data}
                    isActive={currentExperiment?.name === node.experiment_data.name}
                    onSelect={() => handleExperimentSelect(experimentPath)}
                />
            </div>
        );
    }

    return null;
};

// Helper function to check if experiment matches filters
const isExperimentVisible = (experiment, searchTerm, modelFilter) => {
    // Model filter
    if (modelFilter !== 'all') {
        const modelId = experiment.model_id.toLowerCase();
        if (modelFilter === '14b') {
            if (!modelId.includes('14b') && !modelId.includes('2.1')) return false;
        } else if (modelFilter === '1.3b') {
            if (!modelId.includes('1.3b')) return false;
        }
    }

    // Search filter
    if (searchTerm.trim()) {
        const term = searchTerm.toLowerCase();
        return (
            experiment.name.toLowerCase().includes(term) ||
            experiment.base_prompt.toLowerCase().includes(term) ||
            experiment.model_id.toLowerCase().includes(term)
        );
    }

    return true;
};

// Helper function to check if folder has visible experiments
const hasVisibleExperiments = (node, searchTerm, modelFilter) => {
    if (node.type === 'experiment') {
        return isExperimentVisible(node.experiment_data, searchTerm, modelFilter);
    }
    if (node.type === 'folder' && node.children) {
        return node.children.some(child => hasVisibleExperiments(child, searchTerm, modelFilter));
    }
    return false;
};

// Helper function to count experiments in a folder
const getExperimentCount = (node) => {
    if (node.type === 'experiment') return 1;
    if (node.type === 'folder' && node.children) {
        return node.children.reduce((count, child) => count + getExperimentCount(child), 0);
    }
    return 0;
};

const TreeExperimentList = ({ onRescan }) => {
    const navigate = useNavigate();
    const { state, actions } = useApp();
    const { experimentsTree, currentExperiment, isLoading, error } = state;
    const [searchTerm, setSearchTerm] = useState('');
    const [modelFilter, setModelFilter] = useState('all'); // 'all', '14b', '1.3b'

    const handleRescan = useCallback(async () => {
        try {
            actions.setLoading(true);
            await api.scanExperiments();
            const experimentsData = await api.getExperiments();
            actions.setExperimentsTree(experimentsData);
            
            // Also maintain backward compatibility by setting flat experiments list
            const flatExperiments = api.flattenExperimentTree(experimentsData);
            actions.setExperiments(flatExperiments);
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

    // Calculate total visible experiments for search results
    const totalExperiments = useMemo(() => {
        if (!experimentsTree) return 0;
        return getExperimentCount(experimentsTree);
    }, [experimentsTree]);

    const visibleExperiments = useMemo(() => {
        if (!experimentsTree || (!searchTerm && modelFilter === 'all')) return totalExperiments;
        return countVisibleExperiments(experimentsTree, searchTerm, modelFilter);
    }, [experimentsTree, searchTerm, modelFilter, totalExperiments]);

    const handleSearchChange = useCallback((e) => {
        setSearchTerm(e.target.value);
    }, []);

    const clearSearch = useCallback(() => {
        setSearchTerm('');
    }, []);

    const handleModelFilterChange = useCallback((filter) => {
        setModelFilter(filter);
    }, []);

    if (error && !experimentsTree) {
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

    if (!experimentsTree && isLoading) {
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

    if (!experimentsTree && !isLoading) {
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

                {(searchTerm || modelFilter !== 'all') && (
                    <div className="search-results-info">
                        {visibleExperiments} of {totalExperiments} experiments
                        {modelFilter !== 'all' && ` (${modelFilter.toUpperCase()} model)`}
                    </div>
                )}
            </div>

            {/* Tree View */}
            {visibleExperiments === 0 && (searchTerm || modelFilter !== 'all') ? (
                <div className="empty-state">
                    <h3>No experiments found</h3>
                    <p>
                        No experiments match
                        {searchTerm && ` "${searchTerm}"`}
                        {searchTerm && modelFilter !== 'all' && ' and'}
                        {modelFilter !== 'all' && ` ${modelFilter.toUpperCase()} model`}
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
                    </div>
                </div>
            ) : (
                <div className="tree-container">
                    {experimentsTree && (
                        <div>
                            {/* Skip the top-level "outputs" folder and render its children directly */}
                            {experimentsTree.name === 'outputs' && experimentsTree.children ? (
                                experimentsTree.children
                                    .sort((a, b) => {
                                        // Folders first, then by creation time
                                        if (a.type !== b.type) {
                                            return a.type === 'folder' ? -1 : 1;
                                        }
                                        return (b.created_timestamp || 0) - (a.created_timestamp || 0);
                                    })
                                    .map(child => (
                                        <TreeNode
                                            key={child.path}
                                            node={child}
                                            level={0} // Start at level 0 since we're skipping outputs folder
                                            currentExperiment={currentExperiment}
                                            searchTerm={searchTerm}
                                            modelFilter={modelFilter}
                                        />
                                    ))
                            ) : (
                                /* Fallback: render the tree normally if it's not the expected structure */
                                <TreeNode
                                    node={experimentsTree}
                                    level={0}
                                    currentExperiment={currentExperiment}
                                    searchTerm={searchTerm}
                                    modelFilter={modelFilter}
                                />
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

// Helper function to count visible experiments in tree
const countVisibleExperiments = (node, searchTerm, modelFilter) => {
    if (node.type === 'experiment') {
        return isExperimentVisible(node.experiment_data, searchTerm, modelFilter) ? 1 : 0;
    }
    if (node.type === 'folder' && node.children) {
        return node.children.reduce((count, child) => 
            count + countVisibleExperiments(child, searchTerm, modelFilter), 0
        );
    }
    return 0;
};

export default TreeExperimentList;
