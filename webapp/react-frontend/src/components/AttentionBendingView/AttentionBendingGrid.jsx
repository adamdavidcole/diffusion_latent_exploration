import React, { useMemo, useState, useCallback, useRef } from 'react';
import { getThumbnailUrl, getVideoUrl } from '../../services/api';
import './AttentionBendingGrid.css';

const AttentionBendingGrid = ({ baselineVideos, bendingVideos, activeFilters, videoSize = 180, pinBaseline = true }) => {
  // Calculate aspect ratio from first available video metadata
  const aspectRatio = useMemo(() => {
    const firstVideo = bendingVideos[0] || baselineVideos[0];
    if (firstVideo?.width && firstVideo?.height) {
      return firstVideo.width / firstVideo.height;
    }
    return 16 / 9; // Default to 16:9
  }, [bendingVideos, baselineVideos]);

  const videoHeight = Math.round(videoSize / aspectRatio);
  // Filter and organize videos
  const { filteredVideos, promptSeedCombos } = useMemo(() => {
    if (!activeFilters) {
      return { filteredVideos: bendingVideos, promptSeedCombos: [] };
    }

    // Filter bending videos based on active filters
    const filtered = bendingVideos.filter(video => {
      const bending = video.bending_metadata;
      if (!bending) return false;

      // Check operation
      if (activeFilters.operations.size > 0 && 
          !activeFilters.operations.has(bending.transformation_type)) {
        return false;
      }

      // Check tokens - need to match at least one resolved token
      if (activeFilters.tokens.size > 0) {
        const targetToken = bending.target_token || 'ALL';
        const resolvedTokens = bending.resolved_tokens?.[targetToken] || [targetToken];
        const hasMatchingToken = resolvedTokens.some(token => activeFilters.tokens.has(token));
        if (!hasMatchingToken) return false;
      }

      // Check timestep range - must match backend format exactly
      if (activeFilters.timesteps.size > 0) {
        const timestepRange = bending.timestep_range;
        let rangeStr;
        if (timestepRange && Array.isArray(timestepRange) && timestepRange.length === 2) {
          rangeStr = `${timestepRange[0]}-${timestepRange[1]}`;
        } else {
          rangeStr = "ALL";
        }
        if (!activeFilters.timesteps.has(rangeStr)) return false;
      }

      // Check layer range - must match backend format exactly
      if (activeFilters.layers.size > 0) {
        const layerIndices = bending.layer_indices || bending.apply_to_layers;
        let rangeStr;
        if (layerIndices && Array.isArray(layerIndices) && layerIndices.length >= 2) {
          rangeStr = `${Math.min(...layerIndices)}-${Math.max(...layerIndices)}`;
        } else if (layerIndices === "ALL" || (typeof layerIndices === 'string' && layerIndices.toUpperCase() === "ALL")) {
          rangeStr = "ALL";
        } else {
          rangeStr = "ALL";
        }
        if (!activeFilters.layers.has(rangeStr)) return false;
      }

      // Check prompt
      if (activeFilters.prompts.size > 0) {
        const promptId = `p${video.prompt_variation?.index || 0}`;
        if (!activeFilters.prompts.has(promptId)) return false;
      }

      // Check seed
      if (activeFilters.seeds.size > 0 && !activeFilters.seeds.has(video.seed)) {
        return false;
      }

      return true;
    });

    // Get unique prompt×seed combinations from filtered videos
    const combos = new Set();
    filtered.forEach(video => {
      const promptIdx = video.prompt_variation?.index || 0;
      const seed = video.seed;
      combos.add(`p${promptIdx}_s${seed}`);
    });

    return { 
      filteredVideos: filtered, 
      promptSeedCombos: Array.from(combos).sort()
    };
  }, [bendingVideos, activeFilters]);

  // Format operation name for display - defined before it's used
  const formatOperationName = (opType, params) => {
    const formatNumber = (num) => {
      if (typeof num !== 'number') return num;
      // Round to 2 decimal places and remove trailing zeros
      return parseFloat(num.toFixed(2));
    };

    const opTypeLower = opType.toLowerCase();
    
    // Handle SCALE operations
    if (opTypeLower === 'scale') {
      const scaleX = params.scale_x ?? params.scale;
      const scaleY = params.scale_y ?? params.scale;
      
      if (scaleX === scaleY || scaleY == null) {
        return `Scale: ${formatNumber(scaleX)}×`;
      }
      return `Scale: ${formatNumber(scaleX)}×${formatNumber(scaleY)}×`;
    }
    
    // Handle ROTATE operations
    if (opTypeLower === 'rotate') {
      const angle = params.angle ?? params.rotation;
      return `Rotate: ${formatNumber(angle)}°`;
    }
    
    // Handle AMPLIFY operations
    if (opTypeLower === 'amplify') {
      const factor = params.amplify_factor ?? params.factor;
      return `Amplify: ${formatNumber(factor)}×`;
    }
    
    // Handle SHARPEN operations
    if (opTypeLower === 'sharpen') {
      const amount = params.sharpen_amount ?? params.amount;
      return `Sharpen: ${formatNumber(amount)}×`;
    }
    
    // Handle BLUR operations
    if (opTypeLower === 'blur') {
      const sigma = params.sigma;
      return `Blur: ${formatNumber(sigma)}×`;
    }
    
    // Handle FLIP_HORIZONTAL operations
    if (opTypeLower === 'flip_horizontal') {
      return 'Flip Horizontal';
    }
    
    // Handle FLIP_VERTICAL operations
    if (opTypeLower === 'flip_vertical') {
      return 'Flip Vertical';
    }
    
    // Handle TRANSLATE operations
    if (opTypeLower === 'translate') {
      if (params.translate_x !== undefined && params.translate_x !== null) {
        return `Translate X: ${formatNumber(params.translate_x)}`;
      }
      if (params.translate_y !== undefined && params.translate_y !== null) {
        return `Translate Y: ${formatNumber(params.translate_y)}`;
      }
    }
    
    // Generic formatting for other operations
    const paramStr = Object.entries(params)
      .map(([k, v]) => `${k}:${formatNumber(v)}`)
      .join(' ');
    return `${opTypeLower}: ${paramStr}`;
  };

  // Group videos by operation type and parameters
  const groupedVideos = useMemo(() => {
    const groups = {};
    
    filteredVideos.forEach(video => {
      const bending = video.bending_metadata;
      if (!bending) return;

      const opType = bending.transformation_type;
      if (!groups[opType]) {
        groups[opType] = {};
      }

      // Create a key from ALL dimensions: params + token + timestep + layer
      const params = bending.transformation_params || {};
      const paramKey = Object.entries(params)
        .sort()
        .map(([k, v]) => `${k}=${v}`)
        .join(',');

      // Add token/timestep/layer to the key
      const targetToken = bending.target_token || 'ALL';
      const resolvedTokens = bending.resolved_tokens?.[targetToken] || [targetToken];
      const tokenKey = resolvedTokens.sort().join(',');

      const timestepRange = bending.timestep_range;
      const timestepKey = (timestepRange && Array.isArray(timestepRange) && timestepRange.length === 2)
        ? `${timestepRange[0]}-${timestepRange[1]}`
        : 'ALL';

      const layerIndices = bending.layer_indices || bending.apply_to_layers;
      let layerKey;
      if (layerIndices && Array.isArray(layerIndices) && layerIndices.length >= 2) {
        layerKey = `${Math.min(...layerIndices)}-${Math.max(...layerIndices)}`;
      } else if (layerIndices === "ALL" || (typeof layerIndices === 'string' && layerIndices.toUpperCase() === "ALL")) {
        layerKey = 'ALL';
      } else {
        layerKey = 'ALL';
      }

      const fullKey = `${paramKey}|token:${tokenKey}|t:${timestepKey}|l:${layerKey}`;

      if (!groups[opType][fullKey]) {
        groups[opType][fullKey] = {
          params: params,
          displayName: formatOperationName(opType, params),
          metadata: { 
            tokenDisplay: resolvedTokens.join(', '),
            timestepDisplay: timestepKey,
            layerDisplay: layerKey
          },
          videos: {}
        };
      }

      // Index by prompt×seed
      const promptIdx = video.prompt_variation?.index || 0;
      const seed = video.seed;
      const comboKey = `p${promptIdx}_s${seed}`;
      groups[opType][fullKey].videos[comboKey] = video;
    });

    // Sort operations within each type by parameter values
    Object.keys(groups).forEach(opType => {
      const sortedEntries = Object.entries(groups[opType]).sort((a, b) => {
        const aParams = a[1].params;
        const bParams = b[1].params;
        
        // Special handling for translate operations - group by axis first
        if (opType.toLowerCase() === 'translate') {
          const aHasX = aParams.translate_x !== undefined && aParams.translate_x !== null;
          const bHasX = bParams.translate_x !== undefined && bParams.translate_x !== null;
          
          // Group all X translations before Y translations
          if (aHasX && !bHasX) return -1;
          if (!aHasX && bHasX) return 1;
          
          // Within the same axis, sort by value
          const aVal = aHasX ? aParams.translate_x : aParams.translate_y;
          const bVal = bHasX ? bParams.translate_x : bParams.translate_y;
          
          if (typeof aVal === 'number' && typeof bVal === 'number') {
            return aVal - bVal;
          }
          return String(aVal).localeCompare(String(bVal));
        }
        
        // Default sorting for other operations by the first parameter value numerically
        const aVal = Object.values(aParams)[0];
        const bVal = Object.values(bParams)[0];
        
        if (typeof aVal === 'number' && typeof bVal === 'number') {
          return aVal - bVal;
        }
        return String(aVal).localeCompare(String(bVal));
      });

      groups[opType] = Object.fromEntries(sortedEntries);
    });

    return groups;
  }, [filteredVideos, formatOperationName]);

  // Get baseline video for a prompt×seed combo
  const getBaselineVideo = (comboKey) => {
    const [promptPart, seedPart] = comboKey.split('_');
    const promptIdx = parseInt(promptPart.substring(1));
    const seed = parseInt(seedPart.substring(1));

    return baselineVideos.find(v => 
      (v.prompt_variation?.index || 0) === promptIdx && v.seed === seed
    );
  };

  // Video cell component with hover-to-play functionality
  const VideoCell = ({ video }) => {
    const [useVideoElement, setUseVideoElement] = useState(false);
    const hoverTimeoutRef = useRef(null);

    const handleMouseEnter = useCallback(() => {
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
      }
      setUseVideoElement(true);
    }, []);

    const handleMouseLeave = useCallback(() => {
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
      }
      hoverTimeoutRef.current = setTimeout(() => {
        setUseVideoElement(false);
      }, 300);
    }, []);

    const getThumbnailPath = (videoPath) => {
      if (!videoPath) return null;
      const thumbnailPath = videoPath.replace(/\.mp4$/, '.jpg');
      return getThumbnailUrl(thumbnailPath);
    };

    if (!video) {
      return <div className="video-cell empty"></div>;
    }

    return (
      <div 
        className="video-cell"
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        style={{
          width: `${videoSize}px`,
          height: `${videoHeight}px`
        }}
      >
        {useVideoElement ? (
          <video 
            src={getVideoUrl(video.video_path)}
            loop 
            muted 
            autoPlay
            playsInline
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover'
            }}
          />
        ) : (
          <img
            src={getThumbnailPath(video.video_path)}
            alt="Video thumbnail"
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover'
            }}
          />
        )}
      </div>
    );
  };

  const renderVideoCell = (video) => {
    return <VideoCell video={video} />;
  };

  if (promptSeedCombos.length === 0) {
    return (
      <div className="grid-empty">
        <p>No videos match the selected filters.</p>
        <p className="hint">Try adjusting your filter selections.</p>
      </div>
    );
  }

  const MAX_COLUMNS = 5000;
  const showingAll = promptSeedCombos.length <= MAX_COLUMNS;
  const displayedCombos = showingAll ? promptSeedCombos : promptSeedCombos.slice(0, MAX_COLUMNS);

  return (
    <div className="attention-bending-grid">
      {!showingAll && (
        <div className="cutoff-warning">
          ⚠️ Showing {MAX_COLUMNS} of {promptSeedCombos.length} videos. Adjust filters to narrow results.
        </div>
      )}

      <div className="grid-container">
        {/* Header Row with Column Labels */}
        <div className="header-row">
          <div className="grid-row-header corner-header"></div>
          <div className="video-row">
            {displayedCombos.map(combo => {
              const [promptPart, seedPart] = combo.split('_');
              const promptIdx = parseInt(promptPart.substring(1));
              const seed = parseInt(seedPart.substring(1));
              
              // Find prompt text
              const promptText = baselineVideos.find(v => 
                (v.prompt_variation?.index || 0) === promptIdx
              )?.prompt_variation?.text || `Prompt ${promptIdx}`;
              
              return (
                <div 
                  key={combo} 
                  className="column-header-cell"
                  style={{
                    width: `${videoSize}px`,
                    minWidth: `${videoSize}px`,
                    maxWidth: `${videoSize}px`
                  }}
                  title={`Prompt: ${promptText}\nSeed: ${seed}`}
                >
                  <span className="combo-label">{combo.replace('_', ' ')}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Baseline Row (Sticky when pinned) */}
        <div className={`baseline-section ${pinBaseline ? 'pinned' : ''}`}>
          <div className="operation-row">
            <div className="grid-row-header baseline-header">
              📌 Baseline
            </div>
            <div className="video-row">
              {displayedCombos.map(combo => (
                <div key={combo} className="video-column">
                  {renderVideoCell(getBaselineVideo(combo))}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bending Sections by Operation Type */}
        {Object.entries(groupedVideos).map(([opType, operations]) => (
          <div key={opType} className="operation-section">
            <div className="operation-type-header">
              <span className="sticky-label">{opType}</span>
            </div>
            {Object.entries(operations).map(([paramKey, opData]) => (
              <div key={paramKey} className="operation-row">
                <div className="grid-row-header">
                  <div className="operation-name">{opData.displayName}</div>
                  <div className="operation-metadata">
                    <div>token: {opData.metadata.tokenDisplay}</div>
                    <div>timestep: {opData.metadata.timestepDisplay}</div>
                    <div>layers: {opData.metadata.layerDisplay}</div>
                  </div>
                </div>
                <div className="video-row">
                  {displayedCombos.map(combo => (
                    <div key={combo} className="video-column">
                      {renderVideoCell(opData.videos[combo])}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
};

export default AttentionBendingGrid;
