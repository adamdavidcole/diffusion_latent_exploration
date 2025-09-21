import React, { useState, useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import BarChart from '../Charts/BarChart';
import './SimilarityMetricsModal.css';

const SimilarityMetricsModal = () => {
    const { state, actions } = useApp();
    const [selectedMetric, setSelectedMetric] = useState('weighted_similarity_distance');
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
        // Find the prompt that matches this key
        const matchingPrompt = state.currentExperiment?.prompts?.find(p => p.id === promptKey);
        const fullText = matchingPrompt?.variation || promptKey;
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
        console.log('SimilarityMetricsModal - Debug data:', {
            similarityAnalysis: state.similarityAnalysis,
            currentExperiment: state.currentExperiment,
            selectedMetric,
            useZScore
        });

        if (!state.similarityAnalysis?.rankings?.final_scores || !state.currentExperiment) {
            console.log('SimilarityMetricsModal - Missing data, returning empty');
            return { counts: {} };
        }

        const promptEntries = [];
        
        Object.entries(state.similarityAnalysis.rankings.final_scores).forEach(([promptKey, scoreData]) => {
            console.log('SimilarityMetricsModal - Processing prompt:', promptKey, scoreData);
            
            let value;
            if (selectedMetric === 'weighted_similarity_distance') {
                value = scoreData.weighted_similarity_distance;
                console.log('SimilarityMetricsModal - Using weighted distance:', value);
            } else if (useZScore && scoreData.individual_z_scores) {
                const metricKey = `${selectedMetric}_distance`;
                value = scoreData.individual_z_scores[metricKey];
                console.log('SimilarityMetricsModal - Using z-score:', metricKey, value);
            } else {
                // For individual metrics without z-score, we can also use the individual_z_scores
                // since that's what's available in the data structure
                const metricKey = `${selectedMetric}_distance`;
                value = scoreData.individual_z_scores?.[metricKey];
                console.log('SimilarityMetricsModal - Using individual z-score as fallback:', metricKey, value);
            }

            if (value !== undefined) {
                // Use the same variation text logic as VideoGrid
                const displayText = getVariationDisplayText(promptKey, 30);
                console.log('SimilarityMetricsModal - Adding entry:', { label: displayText, value });

                promptEntries.push({
                    label: displayText,
                    value: parseFloat(value.toFixed(4))
                });
            }
        });

        console.log('SimilarityMetricsModal - All prompt entries:', promptEntries);

        // Sort by increasing values (most similar to least similar)
        promptEntries.sort((a, b) => a.value - b.value);

        // Convert to the format expected by BarChart
        const data = { counts: {} };
        promptEntries.forEach(entry => {
            data.counts[entry.label] = entry.value;
        });

        console.log('SimilarityMetricsModal - Final chart data:', data);
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
        return `${metricName} Distance from Baseline${useZScore ? ' (Z-Score)' : ''}`;
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