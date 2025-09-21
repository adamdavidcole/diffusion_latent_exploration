import React, { useState, useMemo, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import BarChart from '../Charts/BarChart';
import './SimilarityMetricsModal.css';

const SimilarityMetricsModal = () => {
    const { state, actions } = useApp();
    const [selectedMetric, setSelectedMetric] = useState('clip');
    const [useZScore, setUseZScore] = useState(false);

    // Helper function to extract just the variation part from the full prompt (COPIED FROM VideoGrid)
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

    // Helper function to get display text for variation with intelligent truncation (COPIED FROM VideoGrid)
    const getVariationDisplayText = useCallback((row, maxLength = 30) => {
        const fullText = row.variation || '';
        const basePrompt = state.currentExperiment?.base_prompt || '';
        
        // Extract just the variation part
        const variationOnly = extractVariationFromPrompt(fullText, basePrompt);
        
        // Handle special cases
        if (variationOnly === '[empty]' || variationOnly.length <= maxLength) {
            return { display: variationOnly, full: fullText };
        }
        
        // If variation is still too long, truncate it
        const truncated = variationOnly.substring(0, maxLength - 3) + '...';
        return { display: truncated, full: fullText };
    }, [state.currentExperiment?.base_prompt, extractVariationFromPrompt]);

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
                // Convert prompt_XXX to XXX format for matching with variation_num
                const variationNum = promptKey.replace('prompt_', '');
                
                // Find the matching video_grid row - video_grid contains the variation text
                const matchingRow = state.currentExperiment?.video_grid?.find(row => row.variation_num === variationNum);
                
                if (matchingRow) {
                    // Use the EXACT same logic as VideoGrid for getting variation text
                    const displayText = getVariationDisplayText(matchingRow, 30).display;
                    
                    promptEntries.push({
                        label: displayText,
                        value: parseFloat(value.toFixed(4))
                    });
                } else {
                    console.warn(`No matching row found for promptKey: ${promptKey}, variationNum: ${variationNum}`);
                    // Fallback to prompt key
                    promptEntries.push({
                        label: promptKey,
                        value: parseFloat(value.toFixed(4))
                    });
                }
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