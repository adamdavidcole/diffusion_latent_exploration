import React from 'react';
import { useApp } from '../../context/AppContext';

const AttentionControls = () => {
    const { state, actions } = useApp();
    
    // Check if attention videos are available for the current experiment
    const hasAttentionVideos = state.currentExperiment?.attention_videos?.available;
    const availableTokens = state.availableTokens || [];
    
    if (!hasAttentionVideos || availableTokens.length === 0) {
        return null; // Don't show controls if no attention videos
    }

    const handleTokenSelect = (token) => {
        if (state.selectedToken === token) {
            // If clicking the same token, deselect it and turn off attention mode
            actions.setSelectedToken(null);
            actions.toggleAttentionMode();
        } else {
            // Select new token and ensure attention mode is on
            actions.setSelectedToken(token);
            if (!state.attentionMode) {
                actions.toggleAttentionMode();
            }
        }
    };

    const handleToggleAttentionMode = () => {
        actions.toggleAttentionMode();
        // If turning off attention mode, clear selected token
        if (state.attentionMode) {
            actions.setSelectedToken(null);
        } else if (availableTokens.length > 0 && !state.selectedToken) {
            // If turning on attention mode and no token selected, select first one
            actions.setSelectedToken(availableTokens[0]);
        }
    };

    return (
        <div className="attention-controls">
            <div className="attention-toggle">
                <label>
                    <input
                        type="checkbox"
                        checked={state.attentionMode}
                        onChange={handleToggleAttentionMode}
                    />
                    🎯 Attention Mode
                </label>
            </div>

            {state.attentionMode && (
                <div className="token-selector">
                    <label>Focus Token:</label>
                    <div className="token-buttons">
                        {availableTokens.map(token => (
                            <button
                                key={token}
                                className={`token-button ${state.selectedToken === token ? 'selected' : ''}`}
                                onClick={() => handleTokenSelect(token)}
                                title={`Show attention for "${token}"`}
                            >
                                {token}
                            </button>
                        ))}
                    </div>
                </div>
            )}

            {state.attentionMode && state.currentExperiment?.attention_videos && (
                <div className="attention-info">
                    <small>
                        {state.currentExperiment.attention_videos.total_count} attention videos available
                    </small>
                </div>
            )}
        </div>
    );
};

export default AttentionControls;
