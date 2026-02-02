/**
 * Utility functions for extracting and formatting prompt variation text
 * This centralizes the logic used across VideoGrid, TrajectoryAnalysis, and SimilarityMetricsModal
 */

/**
 * Helper function to extract just the variation part from the full prompt
 * @param {string} fullPrompt - The complete prompt text
 * @param {string} basePrompt - The base/template prompt with [placeholder] markers
 * @returns {string} - Just the variation part, or '[empty]' if empty, or full prompt if extraction fails
 */
export const extractVariationFromPrompt = (fullPrompt, basePrompt) => {
    if (!basePrompt || !fullPrompt) return fullPrompt;
        
    // Look for patterns like [variation] in base prompt
    const bracketMatch = basePrompt.match(/\[(.*?)\]/);
    if (bracketMatch) {
        // Base prompt has [placeholder], find what replaced it
        const placeholder = bracketMatch[0]; // e.g., "[variation]"
        const beforePlaceholder = basePrompt.split(placeholder)[0];
        const afterPlaceholder = basePrompt.split(placeholder)[1];
        
        
        // Extract the variation by finding what's between the before/after parts
        const beforeIndex = fullPrompt.indexOf(beforePlaceholder);
        const afterIndex = fullPrompt.lastIndexOf(afterPlaceholder);
        
        if (beforeIndex !== -1 && afterIndex !== -1) {
            const startIndex = beforeIndex + beforePlaceholder.length;
            const variation = fullPrompt.substring(startIndex, afterIndex).trim();
            
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
    
    const result = variations.length > 0 ? variations.join(' ') : fullPrompt;
    return result;
};

/**
 * Helper function to get display text for variation with intelligent truncation
 * @param {Object} row - Video grid row object with variation property
 * @param {string} basePrompt - The base/template prompt 
 * @param {number} maxLength - Maximum length for truncation
 * @returns {Object} - { display: string, full: string }
 */
export const getVariationDisplayText = (row, basePrompt, maxLength = 40) => {
    if (!row) return { display: '', full: '' };
    
    const fullText = row.variation || '';
    
    // Extract just the variation part
    const variationOnly = extractVariationFromPrompt(fullText, basePrompt);
    
    // Handle special cases
    if (variationOnly === '[empty]' || variationOnly.length <= maxLength) {
        return { display: variationOnly, full: fullText };
    }
    
    // If variation is still too long, truncate it
    const truncated = variationOnly.substring(0, maxLength - 3) + '...';
    return { display: truncated, full: fullText };
};

/**
 * Find a video grid row by prompt key using multiple matching strategies
 * @param {string} promptKey - The prompt key (e.g., "prompt_001")
 * @param {Array} videoGrid - Array of video grid rows
 * @returns {Object|null} - Matching row or null
 */
export const findVideoGridRowByPromptKey = (promptKey, videoGrid) => {
    if (!promptKey || !videoGrid) return null;
    
    // Extract the prompt number from the key (e.g., "prompt_000" -> "000")
    const promptMatch = promptKey.match(/prompt_(\d+)/);
    if (!promptMatch) return null;
    
    const promptNum = promptMatch[1];
    
    // Try different matching strategies
    const strategies = [
        // Strategy 1: Match by variation_id
        (row) => row.variation_id === `variation_${promptNum}`,
        // Strategy 2: Match by variation_id exact
        (row) => row.variation_id === promptKey,
        // Strategy 3: Match by variation_num
        (row) => row.variation_num === promptNum,
        // Strategy 4: Match by variation_num as number
        (row) => row.variation_num === parseInt(promptNum),
        // Strategy 5: Match by index
        (row, index) => index === parseInt(promptNum)
    ];
    
    for (let i = 0; i < strategies.length; i++) {
        const strategy = strategies[i];
        const matchingRow = videoGrid.find((row, index) => strategy(row, index));
        
        if (matchingRow) {
            return matchingRow;
        }
    }
    
    console.warn(`No matching row found for promptKey: ${promptKey}`);
    return null;
};

/**
 * Main function to get variation text from a prompt key
 * @param {string} promptKey - The prompt key (e.g., "prompt_001")
 * @param {Object} currentExperiment - Current experiment with video_grid and base_prompt
 * @param {boolean} showFullText - Whether to show full prompt or just variation
 * @returns {string} - The variation text or fallback
 */
export const getVariationTextFromPromptKey = (promptKey, currentExperiment, showFullText = false) => {
    if (!promptKey) {
        return 'Unknown Variation';
    }
    
    if (promptKey === 'combined') {
        return 'Combined';
    }
    
    if (!currentExperiment?.video_grid) {
        return promptKey.replace('prompt_', 'Prompt ');
    }
    
    const matchingRow = findVideoGridRowByPromptKey(promptKey, currentExperiment.video_grid);
    
    if (matchingRow) {
        const fullText = matchingRow.variation || '';
        
        // If showFullText is true, return the complete prompt
        if (showFullText) {
            return fullText;
        }
        
        // Otherwise, extract just the variation part using the base prompt
        const basePrompt = currentExperiment?.base_prompt || '';
        const variationOnly = extractVariationFromPrompt(fullText, basePrompt);
        
        return variationOnly;
    }
    
    // Fallback
    return promptKey.replace('prompt_', 'Prompt ');
};