import React, { useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import ExperimentItem from './ExperimentItem';

const ExperimentList = ({ onRescan }) => {
    const navigate = useNavigate();
    const { state, actions } = useApp();
    const { experiments, currentExperiment, isLoading, error } = state;

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
            {experiments
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
            }
        </div>
    );
};

export default ExperimentList;
