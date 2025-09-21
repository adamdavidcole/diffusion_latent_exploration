import React, { useState, useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import BarChart from '../Charts/BarChart';
import './SimilarityMetricsModal.css';

const SimilarityMetricsModal = () => {
    const { state, actions } = useApp();
    const [selectedMetric, setSelectedMetric] = useState('clip');
    const [useZScore, setUseZScore] = useState(false);

    // Helper to extract variation from full prompt (same as VideoGrid)
    const extractVariationFromPrompt = (fullPrompt, basePrompt) => {
        if (!fullPrompt || !basePrompt) return fullPrompt || '[empty]';
        
        const trimmedFull = fullPrompt.trim();
        const trimmedBase = basePrompt.trim();
        
        if (trimmedFull === trimmedBase) return '[empty]';
        
        // If the full prompt starts with the base prompt, extract the variation
        if (trimmedFull.startsWith(trimmedBase)) {
            const variation = trimmedFull.substring(trimmedBase.length).trim();
            // Remove leading comma or semicolon if present
            const cleanVariation = variation.replace(/^[,;]\s*/, '');
            return cleanVariation || '[empty]';
        }
        
        return trimmedFull;
    };

    // Helper to get display text for variation with intelligent truncation (same as VideoGrid)
    const getVariationDisplayText = (promptKey, maxLength = 30) => {
        // Convert prompt_XXX to XXX format for matching with variation_num
        const variationNum = promptKey.replace('prompt_', '');
        
        // Find the matching video_grid row - video_grid contains the variation text
        const matchingRow = state.currentExperiment?.video_grid?.find(row => row.variation_num === variationNum);
        
        if (!matchingRow) {
            console.warn(`No matching row found for promptKey: ${promptKey}, variationNum: ${variationNum}`);
            return promptKey; // fallback to prompt key
        }
        
        const fullText = matchingRow.variation || promptKey;
        const basePrompt = state.currentExperiment?.base_prompt || '';
        
        // Extract just the variation part
        const variationOnly = extractVariationFromPrompt(fullText, basePrompt);
        
        // Handle special cases
        if (variationOnly === '[empty]' || variationOnly.length <= maxLength) {
            return variationOnly;
        }
        
        // If variation is still too long, truncate it
        return variationOnly.substring(0, maxLength - 3) + '...';
    };

    const chartData = useMemo(() => {
        if (!state.similarityAnalysis?.rankings?.final_scores || !state.currentExperiment) {
            return { counts: {} };
        }

        const promptEntries = [];
        
        Object.entries(state.similarityAnalysis.rankings.final_scores).forEach(([promptKey, scoreData]) => {
            let value;
            
            if (selectedMetric === 'weighted_similarity_distance') {
                // Weighted similarity distance is always available in rankings
                value = scoreData.weighted_similarity_distance;
            } else if (useZScore) {
                // Use Z-score normalized values from rankings
                const metricKey = `${selectedMetric}_distance`;
                value = scoreData.individual_z_scores?.[metricKey];
            } else {
                // Use raw similarity values from detailed_similarities
                const detailedData = state.similarityAnalysis.detailed_similarities?.[promptKey];
                const metricKey = `${selectedMetric}_distance`;
                value = detailedData?.aggregated_similarities?.[metricKey]?.mean;
                
                // Fallback to z-score if raw data not available
                if (value === undefined) {
                    value = scoreData.individual_z_scores?.[metricKey];
                }
            }

            if (value !== undefined) {
                // For now, just use prompt key to keep chart working
                promptEntries.push({
                    label: promptKey,
                    value: parseFloat(value.toFixed(4))
                });
            }
        });

        // Sort by increasing values (most similar to least similar)
        promptEntries.sort((a, b) => a.value - b.value);

        // Convert to the format expected by BarChart
        const data = { counts: {} };
        promptEntries.forEach(entry => {
            data.counts[entry.label] = entry.value;
        });

        return data;
    }, [state.similarityAnalysis, state.currentExperiment, selectedMetric, useZScore]);

    // Early return AFTER all hooks are called
    if (!state.showSimilarityMetrics || !state.similarityAnalysis) {
        return null;
    }

    const getChartTitle = () => {
        if (selectedMetric === 'weighted_similarity_distance') {
            return 'Weighted Similarity Distance from Baseline';
        }
        const metricName = selectedMetric.toUpperCase();
        if (useZScore) {
            return `${metricName} Distance from Baseline (Z-Score Normalized)`;
        } else {
            return `${metricName} Distance from Baseline (Raw Values)`;
        }
    };

    const availableMetrics = state.similarityAnalysis.metrics_used || [];

    return (
        <div className="similarity-metrics-modal">
            <div className="modal-overlay" onClick={actions.toggleSimilarityMetrics} />
            <div className="modal-content">
                <div className="modal-header">
                    <h2>Similarity Analysis Metrics</h2>
                    <button 
                        className="close-btn"
                        onClick={actions.toggleSimilarityMetrics}
                        aria-label="Close metrics modal"
                    >
                        ✕
                    </button>
                </div>

                <div className="modal-controls">
                    <div className="control-group">
                        <label htmlFor="metric-select">Metric:</label>
                        <select
                            id="metric-select"
                            value={selectedMetric}
                            onChange={(e) => setSelectedMetric(e.target.value)}
                            className="metric-select"
                        >
                            <option value="weighted_similarity_distance">Weighted Similarity (Combined)</option>
                            {availableMetrics.map(metric => (
                                <option key={metric} value={metric}>
                                    {metric === 'clip' ? 'CLIP Semantic' :
                                     metric === 'lpips' ? 'LPIPS Perceptual' :
                                     metric === 'ssim' ? 'SSIM Structural' :
                                     metric === 'mse' ? 'MSE' :
                                     metric === 'phash' ? 'Perceptual Hash' :
                                     metric.toUpperCase()}
                                </option>
                            ))}
                        </select>
                    </div>

                    {selectedMetric !== 'weighted_similarity_distance' && (
                        <div className="control-group">
                            <label>
                                <input
                                    type="checkbox"
                                    checked={useZScore}
                                    onChange={(e) => setUseZScore(e.target.checked)}
                                />
                                Use Z-Score Normalized Values
                            </label>
                        </div>
                    )}
                </div>

                <div className="chart-container-wrapper">
                    {chartData && (
                        <BarChart 
                            data={chartData}
                            title={getChartTitle()}
                            size={Math.min(1200, window.innerWidth - 100)}
                        />
                    )}
                </div>

                <div className="modal-info">
                    <p><strong>Baseline:</strong> {state.similarityAnalysis.baseline_prompt}</p>
                    <p><strong>Analysis File:</strong> {state.similarityAnalysis.analysis_file}</p>
                    <p><strong>Total Prompts:</strong> {Object.keys(state.similarityAnalysis.rankings.final_scores || {}).length}</p>
                </div>
            </div>
        </div>
    );
};

export default SimilarityMetricsModal;