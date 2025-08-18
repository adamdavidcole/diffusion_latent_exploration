// Shared color utilities for consistent chart coloring across components

export const DEFAULT_CHART_COLORS = [
    '#4A90E2', '#7ED321', '#F5A623', '#D0021B', '#9013FE', '#50E3C2',
    '#B8E986', '#4BD5EE', '#9AA0A6', '#F8E71C'
];

/**
 * Get consistent colors for prompt groups
 * @param {Array} promptGroups - Array of prompt group keys (e.g., ['prompt_000', 'prompt_001'])
 * @param {Array} customColors - Optional custom color array
 * @returns {Array} Array of colors matching the prompt groups
 */
export const getPromptGroupColors = (promptGroups, customColors = null) => {
    const colors = customColors || DEFAULT_CHART_COLORS;
    const sortedGroups = [...promptGroups].sort();
    
    // Create a mapping from prompt group to color index based on sorted order
    const colorMapping = {};
    sortedGroups.forEach((group, index) => {
        colorMapping[group] = colors[index % colors.length];
    });
    
    // Return colors in the original order of promptGroups
    return promptGroups.map(group => colorMapping[group]);
};

/**
 * Get color for a specific prompt group with consistent assignment
 * @param {string} promptGroup - Prompt group key (e.g., 'prompt_000')
 * @param {Array} allPromptGroups - All prompt groups for consistent indexing
 * @param {Array} customColors - Optional custom color array
 * @returns {string} Color for the prompt group
 */
export const getPromptGroupColor = (promptGroup, allPromptGroups, customColors = null) => {
    const colors = getPromptGroupColors(allPromptGroups, customColors);
    const index = allPromptGroups.indexOf(promptGroup);
    return index >= 0 ? colors[index] : (customColors || DEFAULT_CHART_COLORS)[0];
};
