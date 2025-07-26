import React, { useEffect, useRef, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import ExperimentItem from './ExperimentItem';

const ExperimentList = ({ onRescan }) => {
    const { state, actions } = useApp();
    const { experiments, currentExperiment, isLoading, error } = state;
    const hasLoadedRef = useRef(false);

    const handleSelectExperiment = async (experimentName) => {
        try {
            actions.setLoading(true);
            actions.clearError();

            const experiment = await api.getExperiment(experimentName);
            actions.setCurrentExperiment(experiment);
        } catch (error) {
            console.error('Error loading experiment:', error);
            actions.setError(`Failed to load experiment: ${experimentName}`);
        } finally {
            actions.setLoading(false);
        }
    };

    // Load experiments on mount
    useEffect(() => {
        const loadExperiments = async () => {
            console.log('ExperimentList: Loading experiments...');
            try {
                actions.setLoading(true);
                const experimentsData = await api.getExperiments();
                console.log('ExperimentList: Got experiments:', experimentsData);
                actions.setExperiments(experimentsData);

                // Auto-select first experiment if none selected (only on first load)
                if (experimentsData.length > 0 && !currentExperiment && !hasLoadedRef.current) {
                    console.log('ExperimentList: Auto-selecting first experiment:', experimentsData[0].name);
                    hasLoadedRef.current = true;
                    setTimeout(() => {
                        handleSelectExperiment(experimentsData[0].name);
                    }, 100);
                }
            } catch (error) {
                console.error('ExperimentList: Error loading experiments:', error);
                actions.setError('Failed to load experiments. Check the server connection.');
            } finally {
                actions.setLoading(false);
            }
        };

        // Only load experiments if we don't have any yet and haven't loaded before
        if (experiments.length === 0 && !hasLoadedRef.current) {
            console.log('ExperimentList: No experiments in state, loading...');
            loadExperiments();
        } else {
            console.log('ExperimentList: Already have experiments:', experiments.length);
        }
    }, []); // Empty dependency array - only run on mount

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
