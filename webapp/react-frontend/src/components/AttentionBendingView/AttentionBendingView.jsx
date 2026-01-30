import React, { useEffect, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import AttentionBendingFilters from './AttentionBendingFilters';
import AttentionBendingGrid from './AttentionBendingGrid';
import './AttentionBendingView.css';

const AttentionBendingView = ({ experimentPath }) => {
    const { state } = useApp();
    const { currentExperiment } = state;
    const [bendingData, setBendingData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [activeFilters, setActiveFilters] = useState(null);
    const [videoSize, setVideoSize] = useState(180); // Default 180px width
    const [pinBaseline, setPinBaseline] = useState(true); // Pin baseline by default

    useEffect(() => {
        const loadBendingData = async () => {
            if (!experimentPath) return;

            try {
                setLoading(true);
                setError(null);
                const data = await api.getExperimentAttentionBending(experimentPath);
                setBendingData(data);
            } catch (err) {
                console.error('Error loading attention bending data:', err);
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        loadBendingData();
    }, [experimentPath]);

    if (loading) {
        return (
            <div className="attention-bending-view">
                <div className="loading-state">
                    <div className="spinner"></div>
                    <p>Loading attention bending data...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="attention-bending-view">
                <div className="error-state">
                    <h3>⚠️ Error Loading Attention Bending Data</h3>
                    <p>{error}</p>
                </div>
            </div>
        );
    }

    if (!bendingData || !bendingData.available) {
        return (
            <div className="attention-bending-view">
                <div className="empty-state">
                    <h3>📊 No Attention Bending Data</h3>
                    <p>This experiment doesn't have attention bending visualizations available.</p>
                    <p className="hint">
                        Enable attention bending in your generation config to create these visualizations.
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="attention-bending-view">
            {/* Phase 3: Filters */}
            <AttentionBendingFilters 
                filterOptions={bendingData.filter_options}
                onFiltersChange={setActiveFilters}
                videoSize={videoSize}
                onVideoSizeChange={setVideoSize}
                pinBaseline={pinBaseline}
                onPinBaselineChange={setPinBaseline}
            />

            {/* Phase 4: Grid View */}
            {activeFilters && (
                <AttentionBendingGrid
                    baselineVideos={bendingData.baseline_videos}
                    bendingVideos={bendingData.bending_videos}
                    activeFilters={activeFilters}
                    videoSize={videoSize}
                    pinBaseline={pinBaseline}
                />
            )}
        </div>
    );
};

export default AttentionBendingView;
