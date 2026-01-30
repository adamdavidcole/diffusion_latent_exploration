import React, { useState, useMemo, useEffect } from 'react';
import './AttentionBendingFilters.css';

const AttentionBendingFilters = ({ filterOptions, onFiltersChange, videoSize, onVideoSizeChange }) => {
  // Collapsible state - collapsed by default
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Helper to get default selection for tokens/timesteps/layers
  const getDefaultSelection = (options) => {
    // If "ALL" exists, select only "ALL"
    if (options.includes('ALL')) {
      return new Set(['ALL']);
    }
    // Otherwise select the first option
    return new Set(options.length > 0 ? [options[0]] : []);
  };

  // Initialize with ALL selected by default for tokens/timesteps/layers, everything else selected
  const [selectedOperations, setSelectedOperations] = useState(new Set(filterOptions.operations));
  const [selectedTokens, setSelectedTokens] = useState(getDefaultSelection(filterOptions.tokens));
  const [selectedTimesteps, setSelectedTimesteps] = useState(getDefaultSelection(filterOptions.timestep_ranges));
  const [selectedLayers, setSelectedLayers] = useState(getDefaultSelection(filterOptions.layer_ranges));
  const [selectedPrompts, setSelectedPrompts] = useState(new Set(filterOptions.prompts.map(p => p.id)));
  const [selectedSeeds, setSelectedSeeds] = useState(new Set(filterOptions.seeds));

  // Initialize filters on mount
  useEffect(() => {
    onFiltersChange({
      operations: selectedOperations,
      tokens: selectedTokens,
      timesteps: selectedTimesteps,
      layers: selectedLayers,
      prompts: selectedPrompts,
      seeds: selectedSeeds,
    });
  }, []); // Only run once on mount

  // Group operations by type
  const operationGroups = useMemo(() => {
    const groups = {};
    filterOptions.operations.forEach(op => {
      const type = op.split('_')[0]; // e.g., "SCALE" from "SCALE_1.5"
      if (!groups[type]) {
        groups[type] = [];
      }
      groups[type].push(op);
    });
    return groups;
  }, [filterOptions.operations]);

  // Handle filter changes
  const handleOperationToggle = (operation) => {
    const newSelected = new Set(selectedOperations);
    if (newSelected.has(operation)) {
      newSelected.delete(operation);
    } else {
      newSelected.add(operation);
    }
    setSelectedOperations(newSelected);
    notifyChange({ operations: newSelected });
  };

  const handleTokenToggle = (token) => {
    const newSelected = new Set(selectedTokens);
    if (newSelected.has(token)) {
      newSelected.delete(token);
    } else {
      newSelected.add(token);
    }
    setSelectedTokens(newSelected);
    notifyChange({ tokens: newSelected });
  };

  const handleTimestepToggle = (range) => {
    const newSelected = new Set(selectedTimesteps);
    if (newSelected.has(range)) {
      newSelected.delete(range);
    } else {
      newSelected.add(range);
    }
    setSelectedTimesteps(newSelected);
    notifyChange({ timesteps: newSelected });
  };

  const handleLayerToggle = (range) => {
    const newSelected = new Set(selectedLayers);
    if (newSelected.has(range)) {
      newSelected.delete(range);
    } else {
      newSelected.add(range);
    }
    setSelectedLayers(newSelected);
    notifyChange({ layers: newSelected });
  };

  const handlePromptToggle = (promptId) => {
    const newSelected = new Set(selectedPrompts);
    if (newSelected.has(promptId)) {
      newSelected.delete(promptId);
    } else {
      newSelected.add(promptId);
    }
    setSelectedPrompts(newSelected);
    notifyChange({ prompts: newSelected });
  };

  const handleSeedToggle = (seed) => {
    const newSelected = new Set(selectedSeeds);
    if (newSelected.has(seed)) {
      newSelected.delete(seed);
    } else {
      newSelected.add(seed);
    }
    setSelectedSeeds(newSelected);
    notifyChange({ seeds: newSelected });
  };

  const notifyChange = (updates) => {
    onFiltersChange({
      operations: updates.operations || selectedOperations,
      tokens: updates.tokens || selectedTokens,
      timesteps: updates.timesteps || selectedTimesteps,
      layers: updates.layers || selectedLayers,
      prompts: updates.prompts || selectedPrompts,
      seeds: updates.seeds || selectedSeeds,
    });
  };

  const handleSelectAll = (type) => {
    switch(type) {
      case 'operations':
        setSelectedOperations(new Set(filterOptions.operations));
        notifyChange({ operations: new Set(filterOptions.operations) });
        break;
      case 'tokens':
        setSelectedTokens(new Set(filterOptions.tokens));
        notifyChange({ tokens: new Set(filterOptions.tokens) });
        break;
      case 'timesteps':
        setSelectedTimesteps(new Set(filterOptions.timestep_ranges));
        notifyChange({ timesteps: new Set(filterOptions.timestep_ranges) });
        break;
      case 'layers':
        setSelectedLayers(new Set(filterOptions.layer_ranges));
        notifyChange({ layers: new Set(filterOptions.layer_ranges) });
        break;
      case 'prompts':
        setSelectedPrompts(new Set(filterOptions.prompts.map(p => p.id)));
        notifyChange({ prompts: new Set(filterOptions.prompts.map(p => p.id)) });
        break;
      case 'seeds':
        setSelectedSeeds(new Set(filterOptions.seeds));
        notifyChange({ seeds: new Set(filterOptions.seeds) });
        break;
    }
  };

  const handleClearAll = (type) => {
    switch(type) {
      case 'operations':
        setSelectedOperations(new Set());
        notifyChange({ operations: new Set() });
        break;
      case 'tokens':
        setSelectedTokens(new Set());
        notifyChange({ tokens: new Set() });
        break;
      case 'timesteps':
        setSelectedTimesteps(new Set());
        notifyChange({ timesteps: new Set() });
        break;
      case 'layers':
        setSelectedLayers(new Set());
        notifyChange({ layers: new Set() });
        break;
      case 'prompts':
        setSelectedPrompts(new Set());
        notifyChange({ prompts: new Set() });
        break;
      case 'seeds':
        setSelectedSeeds(new Set());
        notifyChange({ seeds: new Set() });
        break;
    }
  };

  const handleResetAll = () => {
    const defaultTokens = getDefaultSelection(filterOptions.tokens);
    const defaultTimesteps = getDefaultSelection(filterOptions.timestep_ranges);
    const defaultLayers = getDefaultSelection(filterOptions.layer_ranges);
    
    setSelectedOperations(new Set(filterOptions.operations));
    setSelectedTokens(defaultTokens);
    setSelectedTimesteps(defaultTimesteps);
    setSelectedLayers(defaultLayers);
    setSelectedPrompts(new Set(filterOptions.prompts.map(p => p.id)));
    setSelectedSeeds(new Set(filterOptions.seeds));
    
    onFiltersChange({
      operations: new Set(filterOptions.operations),
      tokens: defaultTokens,
      timesteps: defaultTimesteps,
      layers: defaultLayers,
      prompts: new Set(filterOptions.prompts.map(p => p.id)),
      seeds: new Set(filterOptions.seeds),
    });
  };

  // Generate compact filter summary
  const getFilterSummary = () => {
    const parts = [];
    
    // Operations
    const opText = selectedOperations.size === filterOptions.operations.length 
      ? 'All' 
      : selectedOperations.size === 0 
        ? 'None' 
        : `${selectedOperations.size} selected`;
    parts.push(`operations: ${opText}`);
    
    // Tokens
    const tokenList = Array.from(selectedTokens).slice(0, 3).join(', ');
    const tokenSuffix = selectedTokens.size > 3 ? '...' : '';
    parts.push(`tokens: ${tokenList}${tokenSuffix}`);
    
    // Timesteps
    const timestepList = Array.from(selectedTimesteps).slice(0, 2).join(', ');
    const timestepSuffix = selectedTimesteps.size > 2 ? '...' : '';
    parts.push(`timesteps: ${timestepList}${timestepSuffix}`);
    
    // Layers
    const layerList = Array.from(selectedLayers).slice(0, 2).join(', ');
    const layerSuffix = selectedLayers.size > 2 ? '...' : '';
    parts.push(`layers: ${layerList}${layerSuffix}`);
    
    // Prompts & Seeds (compact counts)
    parts.push(`${selectedPrompts.size} prompts`);
    parts.push(`${selectedSeeds.size} seeds`);
    
    return parts.join(', ');
  };

  return (
    <div className="attention-bending-filters">
      {!isExpanded ? (
        <div className="filters-collapsed">
          <span className="filters-summary">
            <strong>Filters:</strong> {getFilterSummary()}
          </span>
          <div className="collapsed-actions">
            <button 
              className="reset-button-icon" 
              onClick={handleResetAll}
              title="Reset filters to default"
            >
              ↻
            </button>
            <button className="toggle-filters-button" onClick={() => setIsExpanded(true)}>
              Show Filters
            </button>
          </div>
        </div>
      ) : (
        <>
          <div className="filters-header">
            <h3>Filters</h3>
            <div className="filters-header-actions">
              <button className="reset-button" onClick={handleResetAll}>
                Reset All to Default
              </button>
              <button className="toggle-filters-button" onClick={() => setIsExpanded(false)}>
                Hide Filters
              </button>
            </div>
          </div>

          {/* Video Size Control */}
          <div className="size-control">
            <label htmlFor="video-size-slider">Video Size: {videoSize}px</label>
            <input
              id="video-size-slider"
              type="range"
              min="100"
              max="400"
              value={videoSize}
              onChange={(e) => onVideoSizeChange(Number(e.target.value))}
              className="size-slider"
            />
          </div>

      <div className="filters-grid">
        {/* Operations Filter */}
        <div className="filter-section">
          <div className="filter-section-header">
            <h4>Operations ({selectedOperations.size}/{filterOptions.operations.length})</h4>
            <div className="filter-actions">
              <button onClick={() => handleSelectAll('operations')}>All</button>
              <button onClick={() => handleClearAll('operations')}>None</button>
            </div>
          </div>
          <div className="filter-options-grouped">
            {Object.entries(operationGroups).map(([groupType, operations]) => (
              <div key={groupType} className="operation-group">
                <div className="group-label">{groupType}</div>
                {operations.map(op => (
                  <label key={op} className="filter-checkbox">
                    <input
                      type="checkbox"
                      checked={selectedOperations.has(op)}
                      onChange={() => handleOperationToggle(op)}
                    />
                    <span>{op.replace(`${groupType}_`, '')}</span>
                  </label>
                ))}
              </div>
            ))}
          </div>
        </div>

        {/* Tokens Filter */}
        <div className="filter-section">
          <div className="filter-section-header">
            <h4>Tokens ({selectedTokens.size}/{filterOptions.tokens.length})</h4>
            <div className="filter-actions">
              <button onClick={() => handleSelectAll('tokens')}>All</button>
              <button onClick={() => handleClearAll('tokens')}>None</button>
            </div>
          </div>
          <div className="filter-options">
            {filterOptions.tokens.map(token => (
              <label key={token} className="filter-checkbox">
                <input
                  type="checkbox"
                  checked={selectedTokens.has(token)}
                  onChange={() => handleTokenToggle(token)}
                />
                <span>{token}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Timesteps Filter */}
        <div className="filter-section">
          <div className="filter-section-header">
            <h4>Timestep Ranges ({selectedTimesteps.size}/{filterOptions.timestep_ranges.length})</h4>
            <div className="filter-actions">
              <button onClick={() => handleSelectAll('timesteps')}>All</button>
              <button onClick={() => handleClearAll('timesteps')}>None</button>
            </div>
          </div>
          <div className="filter-options">
            {filterOptions.timestep_ranges.map(range => (
              <label key={range} className="filter-checkbox">
                <input
                  type="checkbox"
                  checked={selectedTimesteps.has(range)}
                  onChange={() => handleTimestepToggle(range)}
                />
                <span>{range}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Layers Filter */}
        <div className="filter-section">
          <div className="filter-section-header">
            <h4>Layer Ranges ({selectedLayers.size}/{filterOptions.layer_ranges.length})</h4>
            <div className="filter-actions">
              <button onClick={() => handleSelectAll('layers')}>All</button>
              <button onClick={() => handleClearAll('layers')}>None</button>
            </div>
          </div>
          <div className="filter-options">
            {filterOptions.layer_ranges.map(range => (
              <label key={range} className="filter-checkbox">
                <input
                  type="checkbox"
                  checked={selectedLayers.has(range)}
                  onChange={() => handleLayerToggle(range)}
                />
                <span>{range}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Prompts Filter */}
        <div className="filter-section">
          <div className="filter-section-header">
            <h4>Prompts ({selectedPrompts.size}/{filterOptions.prompts.length})</h4>
            <div className="filter-actions">
              <button onClick={() => handleSelectAll('prompts')}>All</button>
              <button onClick={() => handleClearAll('prompts')}>None</button>
            </div>
          </div>
          <div className="filter-options">
            {filterOptions.prompts.map(prompt => (
              <label key={prompt.id} className="filter-checkbox">
                <input
                  type="checkbox"
                  checked={selectedPrompts.has(prompt.id)}
                  onChange={() => handlePromptToggle(prompt.id)}
                />
                <span className="prompt-text">{prompt.text}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Seeds Filter */}
        <div className="filter-section">
          <div className="filter-section-header">
            <h4>Seeds ({selectedSeeds.size}/{filterOptions.seeds.length})</h4>
            <div className="filter-actions">
              <button onClick={() => handleSelectAll('seeds')}>All</button>
              <button onClick={() => handleClearAll('seeds')}>None</button>
            </div>
          </div>
          <div className="filter-options">
            {filterOptions.seeds.map(seed => (
              <label key={seed} className="filter-checkbox">
                <input
                  type="checkbox"
                  checked={selectedSeeds.has(seed)}
                  onChange={() => handleSeedToggle(seed)}
                />
                <span>{seed}</span>
              </label>
            ))}
          </div>
        </div>
      </div>
        </>
      )}
    </div>
  );
};

export default AttentionBendingFilters;
