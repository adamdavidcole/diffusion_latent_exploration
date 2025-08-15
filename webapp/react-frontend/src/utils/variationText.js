// Utility function to extract variation text from prompt keys
// This helps debug and ensure consistent variation text mapping

export const getVariationTextFromPromptKey = (promptKey, currentExperiment) => {
  console.log('Getting variation text for:', promptKey, "from ", currentExperiment);
  console.log('Current experiment video_grid:', currentExperiment?.video_grid);
  
  if (promptKey === 'combined') {
    return 'Combined';
  }
  
  if (!currentExperiment?.video_grid) {
    console.log('No video grid available, using fallback');
    return promptKey.replace('prompt_', 'Prompt ');
  }

  // Extract the prompt number from the key (e.g., "prompt_000" -> "000")
  const promptMatch = promptKey.match(/prompt_(\d+)/);
  console.log('Prompt match:', promptMatch);
  
  if (!promptMatch) {
    console.log('No prompt match found, using fallback');
    return promptKey.replace('prompt_', 'Prompt ');
  }

  const promptNum = promptMatch[1];
  console.log('Extracted prompt number:', promptNum);
  
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
    const matchingRow = currentExperiment.video_grid.find((row, index) => strategy(row, index));
    
    console.log(`Strategy ${i + 1}:`, strategy.toString());
    console.log(`Strategy ${i + 1} result:`, matchingRow);
    
    if (matchingRow) {
      console.log('Found matching row with variation:', matchingRow.variation);
      return matchingRow.variation;
    }
  }

  // Log all available rows for debugging
  console.log('All video grid rows:');
  currentExperiment.video_grid.forEach((row, index) => {
    console.log(`Row ${index}:`, {
      variation: row.variation,
      variation_id: row.variation_id,
      variation_num: row.variation_num
    });
  });

  // Fallback
  console.log('No match found, using fallback');
  return promptKey.replace('prompt_', 'Prompt ');
};

export default getVariationTextFromPromptKey;
