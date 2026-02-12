import React, { useEffect, useState, useMemo, useRef } from 'react';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import LatentVideoCell from './LatentVideoCell';
import LatentVideoLightbox from './LatentVideoLightbox';
import { getVariationTextFromPromptKey } from '../../utils/variationText';
import './LatentVideosView.css';

const LatentVideosView = ({ experimentPath }) => {
  const { state, actions } = useApp();
  const { currentLatentVideos, latentVideosLoading, latentVideosError, currentExperiment } = state;

  // View mode state
  const [viewMode, setViewMode] = useState('across-prompts'); // 'across-prompts' or 'by-prompt'
  const [selectedPrompt, setSelectedPrompt] = useState('prompt_000');
  const [selectedSeed, setSelectedSeed] = useState('vid_001');
  const [cellSize, setCellSize] = useState(120); // Local cell size state
  const [selectedAttentionTokens, setSelectedAttentionTokens] = useState([]); // Array of selected tokens
  const [tokensExpanded, setTokensExpanded] = useState(false); // Token selector expansion state
  const tokensDropdownRef = useRef(null); // Ref for click-outside detection
  const [tooltip, setTooltip] = useState({ show: false, text: '', x: 0, y: 0 });
  const [lightbox, setLightbox] = useState({ isOpen: false, position: null });
  const [controlsExpanded, setControlsExpanded] = useState(true); // Collapsible controls

  // Attention video tier filtering state
  const [attentionViewMode, setAttentionViewMode] = useState('default'); // 'default', 'by-layer', 'by-head'
  const [selectedLayer, setSelectedLayer] = useState('averaged'); // 'averaged' or layer number
  const [selectedHead, setSelectedHead] = useState('averaged'); // 'averaged' or head number

  // Click outside handler for tokens dropdown
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (tokensExpanded && tokensDropdownRef.current && !tokensDropdownRef.current.contains(event.target)) {
        setTokensExpanded(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [tokensExpanded]);

  // Load latent videos data when component mounts or experiment changes
  useEffect(() => {
    const loadLatentVideos = async () => {
      if (!experimentPath) return;

      try {
        actions.setLatentVideosLoading(true);
        actions.clearLatentVideosError();

        const latentVideosData = await api.getExperimentLatentVideos(experimentPath);
        actions.setCurrentLatentVideos(latentVideosData);

        // Auto-select first available prompt and seed
        if (latentVideosData?.latent_videos) {
          const firstPrompt = Object.keys(latentVideosData.latent_videos)[0];
          if (firstPrompt) {
            setSelectedPrompt(firstPrompt);
            const firstSeed = Object.keys(latentVideosData.latent_videos[firstPrompt])[0];
            if (firstSeed) {
              setSelectedSeed(firstSeed);
            }
          }
        } else if (latentVideosData?.attention_videos) {
          // If no latent videos but we have attention videos, use those for structure
          const firstPrompt = Object.keys(latentVideosData.attention_videos)[0];
          if (firstPrompt) {
            setSelectedPrompt(firstPrompt);
            const firstSeed = Object.keys(latentVideosData.attention_videos[firstPrompt])[0];
            if (firstSeed) {
              setSelectedSeed(firstSeed);
            }
          }
        }

        // Auto-select first attention token if no latent videos available
        if (!latentVideosData?.has_latent_videos && latentVideosData?.attention_videos) {
          const firstPrompt = Object.keys(latentVideosData.attention_videos)[0];
          if (firstPrompt) {
            const firstSeed = Object.keys(latentVideosData.attention_videos[firstPrompt])[0];
            if (firstSeed) {
              const tokens = Object.keys(latentVideosData.attention_videos[firstPrompt][firstSeed]);
              if (tokens.length > 0) {
                setSelectedAttentionTokens([tokens[0]]);
              }
            }
          }
        } else {
          // Reset attention token selection when loading new data with latent videos
          setSelectedAttentionTokens([]);
        }
      } catch (error) {
        console.error('Error loading latent videos:', error);
        actions.setLatentVideosError(error.message);
      } finally {
        actions.setLatentVideosLoading(false);
      }
    };

    loadLatentVideos();
  }, [experimentPath]); // Removed actions from dependency array to prevent infinite loop

  // Helper function to get display text for prompts with truncation
  // Helper function to generate summary text for collapsed view
  const getControlsSummary = () => {
    const parts = [];

    // View mode
    parts.push(`View: ${viewMode === 'across-prompts' ? 'Across Prompts' : 'By Prompt'}`);

    // Seed or Prompt
    if (viewMode === 'across-prompts') {
      parts.push(`Seed: ${selectedSeed.replace('vid_', '').replace(/^0+/, '') || '?'}`);
    } else {
      const promptText = getPromptDisplayText(selectedPrompt, 20).display;
      parts.push(`Prompt: ${promptText}`);
    }

    // Attention tokens
    if (selectedAttentionTokens.length > 0) {
      if (selectedAttentionTokens.length === 1) {
        const tokenDisplay = selectedAttentionTokens[0].length > 10 ? `${selectedAttentionTokens[0].substring(0, 10)}...` : selectedAttentionTokens[0];
        parts.push(`Token: ${tokenDisplay}`);
      } else {
        parts.push(`Tokens: ${selectedAttentionTokens.length} selected`);
      }

      // Attention view
      if (attentionViewMode === 'by-layer') {
        parts.push('View: By Layer');
      } else if (attentionViewMode === 'by-head') {
        parts.push('View: By Head');
      }
    } else {
      parts.push('Tokens: None');
    }

    return parts.join(' | ');
  };

  const getPromptDisplayText = (promptId, maxLength = 40) => {
    const fullText = getVariationTextFromPromptKey(promptId, currentExperiment);

    if (fullText.length <= maxLength) {
      return { display: fullText, full: fullText };
    }

    // Truncate and add ellipsis
    const truncated = fullText.substring(0, maxLength - 3) + '...';
    return { display: truncated, full: fullText };
  };

  // Helper function to format video_id labels (e.g., p000_b001_s000 -> readable label)
  const getVideoIdLabel = (videoId, metadata) => {
    // Check if it's old format (vid001)
    if (videoId.startsWith('vid')) {
      return videoId.replace('vid', 'Seed ').replace(/^Seed 0+/, 'Seed ');
    }

    // New format: p000_b001_s000
    if (videoId.startsWith('p') && videoId.includes('_b') && videoId.includes('_s')) {
      // Extract seed number from video_id (e.g., p000_b001_s000 -> s000)
      const seedMatch = videoId.match(/_s(\d+)/);
      const seedNum = seedMatch ? parseInt(seedMatch[1]) : null;
      const seedLabel = seedNum !== null ? ` (Seed ${seedNum})` : '';

      // Try to get metadata for this video
      const videoMetadata = metadata?.[videoId];

      // Try both field names (attention_bending_settings and bending_metadata)
      const bending = videoMetadata?.attention_bending_settings || videoMetadata?.bending_metadata;

      if (bending && bending !== null) {
        const parts = [];

        // Handle spatial/frequency transformations (new attention bending)
        if (bending.transformation_type) {
          const type = bending.transformation_type;
          const params = bending.transformation_params || {};

          // Helper to format numeric values to 2-3 decimal places
          const formatNum = (num) => {
            if (num === undefined || num === null) return num;
            // Remove trailing zeros after decimal point
            return parseFloat(num.toFixed(3));
          };

          if (type === 'scale') {
            parts.push(`Scale ${formatNum(params.scale_x || params.scale)}×`);
          } else if (type === 'rotate') {
            parts.push(`Rotate ${formatNum(params.angle)}°`);
          } else if (type === 'translate') {
            // Check params to determine if it's translate_x or translate_y
            if (params.translate_x !== undefined) {
              const val = formatNum(params.translate_x);
              parts.push(`Translate X ${val > 0 ? '+' : ''}${val}`);
            } else if (params.translate_y !== undefined) {
              const val = formatNum(params.translate_y);
              parts.push(`Translate Y ${val > 0 ? '+' : ''}${val}`);
            } else if (params.shift_x !== undefined) {
              const val = formatNum(params.shift_x);
              parts.push(`Translate X ${val > 0 ? '+' : ''}${val}`);
            } else if (params.shift_y !== undefined) {
              const val = formatNum(params.shift_y);
              parts.push(`Translate Y ${val > 0 ? '+' : ''}${val}`);
            } else {
              parts.push('Translate');
            }
          } else if (type === 'flip_horizontal') {
            parts.push('Flip Horizontal');
          } else if (type === 'flip_vertical') {
            parts.push('Flip Vertical');
          } else if (type === 'flip') {
            // Check params to determine flip direction
            if (params.flip_horizontal) {
              parts.push('Flip Horizontal');
            } else if (params.flip_vertical) {
              parts.push('Flip Vertical');
            } else {
              parts.push('Flip');
            }
          } else if (type === 'gaussian_blur' || type === 'blur') {
            parts.push(`Blur σ=${formatNum(params.sigma)}`);
          } else if (type === 'edge_enhance') {
            parts.push(`Edge α=${formatNum(params.alpha)}`);
          } else if (type === 'frequency_filter') {
            const mode = params.mode || 'lowpass';
            parts.push(`${mode.charAt(0).toUpperCase() + mode.slice(1)} f=${formatNum(params.cutoff_freq)}`);
          } else {
            parts.push(type);
          }

          // Add phase if specified
          if (bending.phase) {
            parts.push(`Phase ${bending.phase}`);
          }

          // Add timesteps (only if NOT null/ALL)
          if (bending.timestep_range && Array.isArray(bending.timestep_range)) {
            parts.push(`T:${bending.timestep_range[0]}-${bending.timestep_range[1]}`);
          }

          // Add layers (only if NOT null/ALL)
          if (bending.layer_indices && bending.layer_indices.length > 0) {
            if (bending.layer_indices.length === 1) {
              parts.push(`L:${bending.layer_indices[0]}`);
            } else {
              parts.push(`L:${bending.layer_indices[0]}-${bending.layer_indices[bending.layer_indices.length - 1]}`);
            }
          }

          // Add token if specified and not ALL
          if (bending.target_token && bending.target_token !== 'ALL') {
            const tokenText = bending.target_token.length > 15
              ? `"${bending.target_token.substring(0, 15)}..."`
              : `"${bending.target_token}"`;
            parts.push(tokenText);
          }

          return parts.join(' | ') + seedLabel;
        }

        // Legacy amplitude-based operations (old attention bending)
        if (bending.operation) {
          if (bending.operation === 'scale') {
            parts.push(`Scale ${bending.value}×`);
          } else if (bending.operation === 'add') {
            parts.push(`Add ${bending.value}`);
          } else if (bending.operation === 'set') {
            parts.push(`Set ${bending.value}`);
          }

          // Timesteps
          if (bending.timestep_spec) {
            parts.push(`T:${bending.timestep_spec}`);
          }

          // Layers
          if (bending.layer_spec && bending.layer_spec !== 'ALL') {
            parts.push(`L:${bending.layer_spec}`);
          } else if (bending.layer_spec === 'ALL') {
            parts.push('L:ALL');
          }

          // Token
          if (bending.target_token && bending.target_token !== 'ALL') {
            parts.push(`"${bending.target_token}"`);
          }

          return parts.join(' | ') + seedLabel;
        }
      } else if (videoId.includes('_b000_')) {
        // Baseline - bending is null or not present
        // Prefer showing the actual prompt if available
        if (metadata && metadata.prompt_variation) {
          return metadata.prompt_variation + seedLabel;
        }
        return 'Baseline (No Bending)' + seedLabel;
      } else {
        // Fallback: parse the ID
        try {
          const bendingPart = videoId.split('_b')[1].split('_')[0];
          return `Bending ${bendingPart}`;
        } catch {
          return videoId;
        }
      }
    }

    return videoId;
  };

  // Organize data based on view mode
  const gridData = useMemo(() => {
    // Check if we have ANY data to display (latent videos or attention videos)
    const hasLatentVideos = currentLatentVideos?.latent_videos;
    const hasAttentionVideos = currentLatentVideos?.attention_videos;

    if (!hasLatentVideos && !hasAttentionVideos) return null;

    // Get video metadata map for labels
    const videoMetadataMap = currentLatentVideos?.video_metadata_map || {};

    // If we only have attention videos, use them to determine structure
    const dataSource = hasLatentVideos ? currentLatentVideos.latent_videos : currentLatentVideos.attention_videos;
    const attentionVideos = currentLatentVideos.attention_videos;

    let allPrompts, allSeeds, allSteps;

    if (hasLatentVideos) {
      // Use latent videos structure
      allPrompts = Object.keys(dataSource).sort();
      // Collect all unique seeds across all prompts
      const seedsSet = new Set();
      allPrompts.forEach(prompt => {
        Object.keys(dataSource[prompt]).forEach(seed => seedsSet.add(seed));
      });
      allSeeds = Array.from(seedsSet).sort();

      if (allPrompts.length > 0 && allSeeds.length > 0) {
        const firstVideo = dataSource[allPrompts[0]][allSeeds[0]];
        allSteps = Object.keys(firstVideo).sort();
      } else {
        allSteps = [];
      }
    } else {
      // Use attention videos structure (first token)
      allPrompts = Object.keys(dataSource).sort();
      // Collect all unique seeds across all prompts
      const seedsSet = new Set();
      allPrompts.forEach(prompt => {
        Object.keys(dataSource[prompt]).forEach(seed => seedsSet.add(seed));
      });
      allSeeds = Array.from(seedsSet).sort();

      if (allPrompts.length > 0 && allSeeds.length > 0) {
        // Get first token's data to determine steps
        const firstTokenData = dataSource[allPrompts[0]][allSeeds[0]];
        const firstToken = Object.keys(firstTokenData)[0];
        const allFilenames = Object.keys(firstTokenData[firstToken]).sort();

        // Filter to only base step names (tier 1: no _layer_ or _head_ suffix)
        // This gives us step_000, step_001, etc. without the layer/head variants
        allSteps = allFilenames.filter(filename => {
          return !filename.includes('_layer_') && !filename.includes('_head_');
        });
      } else {
        allSteps = [];
      }
    }

    // Helper function to get video/image paths with attention video fallback
    const getVideoData = (promptId, seedId, stepId, tokenId = null, layerOverride = null, headOverride = null) => {
      let videoPath = null;
      let imagePath = null;

      // Try attention videos first if a token is provided
      if (tokenId && attentionVideos) {
        const tokenData = attentionVideos[promptId]?.[seedId]?.[tokenId];
        if (tokenData) {
          // Determine which video to use based on layer/head overrides (for expanded rows)
          // or attention view mode and filters (for single row views)
          let targetFilename = stepId; // Default: tier 1 (e.g., "step_000")

          if (layerOverride !== null && headOverride !== null) {
            // Tier 3: specific layer and head (from row expansion)
            targetFilename = `${stepId}_layer_${String(layerOverride).padStart(2, '0')}_head_${String(headOverride).padStart(2, '0')}`;
          } else if (layerOverride !== null) {
            // Tier 2: specific layer, averaged heads (from row expansion)
            targetFilename = `${stepId}_layer_${String(layerOverride).padStart(2, '0')}`;
          } else if (attentionViewMode === 'by-head' && selectedLayer !== 'averaged' && selectedHead !== 'averaged') {
            // Tier 3: specific layer and head (from filter selection)
            targetFilename = `${stepId}_layer_${String(selectedLayer).padStart(2, '0')}_head_${String(selectedHead).padStart(2, '0')}`;
          } else if (attentionViewMode === 'by-layer' && selectedLayer !== 'averaged') {
            // Tier 2: specific layer, averaged heads (from filter selection)
            targetFilename = `${stepId}_layer_${String(selectedLayer).padStart(2, '0')}`;
          }
          // else: tier 1 (default) - use base stepId

          const attentionData = tokenData[targetFilename];
          if (attentionData) {
            videoPath = attentionData.video_path;
            imagePath = attentionData.image_path;
          } else if (attentionViewMode !== 'default' || layerOverride !== null) {
            // Fallback to tier 1 if specific tier not found
            const fallbackData = tokenData[stepId];
            if (fallbackData) {
              videoPath = fallbackData.video_path;
              imagePath = fallbackData.image_path;
            }
          }
        }
      }

      // Fallback to latent videos if no attention video found
      if (!videoPath && hasLatentVideos) {
        const latentData = dataSource[promptId]?.[seedId]?.[stepId];
        if (latentData) {
          videoPath = latentData.video_path;
          imagePath = latentData.image_path;
        }
      }

      return { videoPath, imagePath };
    };

    // Check if we need to expand rows for by-layer or by-head modes
    const hasSelectedTokens = selectedAttentionTokens.length > 0;
    const needsLayerExpansion = hasSelectedTokens && attentionViewMode === 'by-layer' && selectedLayer === 'averaged';
    const needsHeadExpansion = hasSelectedTokens && attentionViewMode === 'by-head' && (selectedLayer === 'averaged' || selectedHead === 'averaged');

    // Get available layers and heads for expansion
    const availableLayers = currentLatentVideos?.available_layers || [];
    const availableHeads = currentLatentVideos?.available_heads || [];

    if (needsHeadExpansion && availableLayers.length > 0 && availableHeads.length > 0) {
      // By-head mode with expansion: show filtered layer×head combinations in hierarchical sections
      const sections = [];
      const MAX_ROWS = 500;
      let totalRows = 0;

      // Determine which layers and heads to iterate over based on filters
      const layersToShow = selectedLayer === 'averaged' ? availableLayers : [selectedLayer];
      const headsToShow = selectedHead === 'averaged' ? availableHeads : [selectedHead];

      for (const promptId of (viewMode === 'across-prompts' ? allPrompts : [selectedPrompt])) {
        if (totalRows >= MAX_ROWS) break;

        for (const seedId of (viewMode === 'across-prompts' ? [selectedSeed] : allSeeds)) {
          if (totalRows >= MAX_ROWS) break;

          // Get token for this combination - use selected tokens
          const tokenData = attentionVideos?.[promptId]?.[seedId];
          if (!tokenData) continue;

          // Use selected tokens (already filtered by user)
          const tokens = selectedAttentionTokens.filter(token => tokenData[token]);

          for (const token of tokens) {
            if (totalRows >= MAX_ROWS) break;

            const rows = [];
            for (const layer of layersToShow) {
              if (totalRows >= MAX_ROWS) break;

              for (const head of headsToShow) {
                if (totalRows >= MAX_ROWS) break;

                rows.push({
                  id: `${promptId}_${seedId}_${token}_l${layer}_h${head}`,
                  label: selectedLayer === 'averaged' && selectedHead === 'averaged'
                    ? `Layer ${layer} Head ${head}`
                    : selectedLayer === 'averaged'
                      ? `Layer ${layer}`
                      : `Head ${head}`,
                  fullLabel: `Layer ${layer} Head ${head}`,
                  steps: allSteps.map(stepId => {
                    const { videoPath, imagePath } = getVideoData(promptId, seedId, stepId, token, layer, head);
                    return {
                      stepId,
                      stepNumber: parseInt(stepId.replace('step_', '')),
                      videoPath,
                      imagePath,
                    };
                  })
                });
                totalRows++;
              }
            }

            // Filter out rows with no videos
            const filteredRows = rows.filter(row => 
              row.steps.some(step => step.videoPath)
            );

            if (filteredRows.length > 0) {
              // Get prompt text from video metadata (supports new format with bending_metadata)
              const videoMetadata = videoMetadataMap[seedId];
              const promptText = videoMetadata?.prompt_variation || getPromptDisplayText(promptId).display;

              sections.push({
                promptId,
                promptLabel: promptText,
                tokenId: token,
                tokenLabel: `Token ${token}`,
                rows: filteredRows
              });
            }
          }
        }
      }

      return {
        sections,
        columnHeaders: allSteps.map(stepId => `Step ${stepId.replace('step_', '')}`),
        rowType: 'hierarchical',
        selectedItem: selectedLayer !== 'averaged' && selectedHead === 'averaged'
          ? `Showing ${totalRows} heads for Layer ${selectedLayer} (max 500)`
          : selectedLayer === 'averaged' && selectedHead !== 'averaged'
            ? `Showing ${totalRows} layers for Head ${selectedHead} (max 500)`
            : `Showing ${totalRows} layer×head combinations (max 500)`,
        rows: sections.flatMap(s => s.rows) // Flat list for lightbox
      };
    } else if (needsLayerExpansion && availableLayers.length > 0) {
      // By-layer mode with expansion: show all layers in hierarchical sections
      const sections = [];
      const MAX_ROWS = 500;
      let totalRows = 0;

      for (const promptId of (viewMode === 'across-prompts' ? allPrompts : [selectedPrompt])) {
        if (totalRows >= MAX_ROWS) break;

        for (const seedId of (viewMode === 'across-prompts' ? [selectedSeed] : allSeeds)) {
          if (totalRows >= MAX_ROWS) break;

          // Get token for this combination - use selected tokens
          const tokenData = attentionVideos?.[promptId]?.[seedId];
          if (!tokenData) continue;

          // Use selected tokens (already filtered by user)
          const tokens = selectedAttentionTokens.filter(token => tokenData[token]);

          for (const token of tokens) {
            if (totalRows >= MAX_ROWS) break;

            const rows = [];
            for (const layer of availableLayers) {
              if (totalRows >= MAX_ROWS) break;

              rows.push({
                id: `${promptId}_${seedId}_${token}_l${layer}`,
                label: `Layer ${layer}`,
                fullLabel: `Layer ${layer}`,
                steps: allSteps.map(stepId => {
                  const { videoPath, imagePath } = getVideoData(promptId, seedId, stepId, token, layer, null);
                  return {
                    stepId,
                    stepNumber: parseInt(stepId.replace('step_', '')),
                    videoPath,
                    imagePath,
                  };
                })
              });
              totalRows++;
            }

            // Filter out rows with no videos
            const filteredRows = rows.filter(row => 
              row.steps.some(step => step.videoPath)
            );

            if (filteredRows.length > 0) {
              // Get prompt text from video metadata (supports new format with bending_metadata)
              const videoMetadata = videoMetadataMap[seedId];
              const promptText = videoMetadata?.prompt_variation || getPromptDisplayText(promptId).display;

              sections.push({
                promptId,
                promptLabel: promptText,
                tokenId: token,
                tokenLabel: `Token ${token}`,
                rows: filteredRows
              });
            }
          }
        }
      }

      return {
        sections,
        columnHeaders: allSteps.map(stepId => `Step ${stepId.replace('step_', '')}`),
        rowType: 'hierarchical',
        selectedItem: `Showing ${totalRows} layers (max 500)`,
        rows: sections.flatMap(s => s.rows) // Flat list for lightbox
      };
    } else if (hasSelectedTokens) {
      // Show sections for each selected token
      const sections = [];

      for (const token of selectedAttentionTokens) {
        if (viewMode === 'across-prompts') {
          // Each row is a different prompt, all using the same seed
          const rows = allPrompts.map(promptId => {
            const dataSource = hasLatentVideos ? currentLatentVideos.latent_videos : currentLatentVideos.attention_videos;
            const seedsForPrompt = Object.keys(dataSource[promptId] || {});
            
            // Find the corresponding seed for this prompt based on the selected seed's pattern
            // Extract the seed number from selectedSeed (e.g., "p000_b000_s001" -> "s001")
            const selectedSeedMatch = selectedSeed.match(/_s(\d+)$/);
            const selectedSeedNum = selectedSeedMatch ? selectedSeedMatch[1] : '000';
            
            // Find a seed in this prompt that has the same seed number
            const correspondingSeed = seedsForPrompt.find(seed => seed.endsWith(`_s${selectedSeedNum}`)) || seedsForPrompt[0] || selectedSeed;
            
            const seedForMetadata = seedsForPrompt[0] || correspondingSeed;

            const videoMetadata = videoMetadataMap[seedForMetadata];
            const promptText = videoMetadata?.prompt_variation || getPromptDisplayText(promptId).display;

            return {
              id: `${promptId}_${token}`,
              label: promptText,
              fullLabel: promptText,
              steps: allSteps.map(stepId => {
                const { videoPath, imagePath } = getVideoData(promptId, correspondingSeed, stepId, token);
                return {
                  stepId,
                  stepNumber: parseInt(stepId.replace('step_', '')),
                  videoPath,
                  imagePath,
                };
              })
            };
          });

          // Filter out rows with no videos (where all steps have null videoPath)
          const filteredRows = rows.filter(row => 
            row.steps.some(step => step.videoPath)
          );

          if (filteredRows.length > 0) {
            const selectedSeedLabel = getVideoIdLabel(selectedSeed, videoMetadataMap);
            sections.push({
              promptId: 'all',
              promptLabel: selectedSeedLabel,
              tokenId: token,
              tokenLabel: token,
              rows: filteredRows
            });
          }
        } else {
          // Each row is a different seed, all using the same prompt
          const rows = allSeeds.map(seedId => {
            const readableLabel = getVideoIdLabel(seedId, videoMetadataMap);
            return {
              id: `${seedId}_${token}`,
              label: readableLabel,
              fullLabel: readableLabel,
              steps: allSteps.map(stepId => {
                const { videoPath, imagePath } = getVideoData(selectedPrompt, seedId, stepId, token);
                return {
                  stepId,
                  stepNumber: parseInt(stepId.replace('step_', '')),
                  videoPath,
                  imagePath,
                };
              })
            };
          });

          // Filter out rows with no videos (where all steps have null videoPath)
          const filteredRows = rows.filter(row => 
            row.steps.some(step => step.videoPath)
          );

          if (filteredRows.length > 0) {
            const firstSeed = allSeeds[0];
            const videoMetadata = videoMetadataMap[firstSeed];
            const selectedPromptDisplay = videoMetadata?.prompt_variation || getPromptDisplayText(selectedPrompt).display;

            sections.push({
              promptId: selectedPrompt,
              promptLabel: selectedPromptDisplay,
              tokenId: token,
              tokenLabel: token,
              rows: filteredRows
            });
          }
        }
      }

      return {
        sections,
        columnHeaders: allSteps.map(stepId => `Step ${stepId.replace('step_', '')}`),
        rowType: 'hierarchical',
        selectedItem: viewMode === 'across-prompts' ? getVideoIdLabel(selectedSeed, videoMetadataMap) : (videoMetadataMap[allSeeds[0]]?.prompt_variation || getPromptDisplayText(selectedPrompt).display),
        rows: sections.flatMap(s => s.rows)
      };
    } else if (viewMode === 'across-prompts') {
      // Latent videos only - each row is a different prompt
      const rows = allPrompts.map(promptId => {
        // Get the seeds available for this specific prompt
        const dataSource = hasLatentVideos ? currentLatentVideos.latent_videos : currentLatentVideos.attention_videos;
        const seedsForPrompt = Object.keys(dataSource[promptId] || {});
        
        // Find the corresponding seed for this prompt based on the selected seed's pattern
        // Extract the seed number from selectedSeed (e.g., "p000_b000_s001" -> "s001")
        const selectedSeedMatch = selectedSeed.match(/_s(\d+)$/);
        const selectedSeedNum = selectedSeedMatch ? selectedSeedMatch[1] : '000';
        
        // Find a seed in this prompt that has the same seed number
        const correspondingSeed = seedsForPrompt.find(seed => seed.endsWith(`_s${selectedSeedNum}`)) || seedsForPrompt[0] || selectedSeed;
        
        // Use the first seed for metadata lookup
        const seedForMetadata = seedsForPrompt[0] || correspondingSeed;

        // Get prompt text from video metadata
        const videoMetadata = videoMetadataMap[seedForMetadata];
        const promptText = videoMetadata?.prompt_variation || getPromptDisplayText(promptId).display;

        return {
          id: promptId,
          label: promptText,
          fullLabel: promptText,
          steps: allSteps.map(stepId => {
            const { videoPath, imagePath } = getVideoData(promptId, correspondingSeed, stepId, null);
            return {
              stepId,
              stepNumber: parseInt(stepId.replace('step_', '')),
              videoPath,
              imagePath,
            };
          })
        };
      });

      // Filter out rows with no videos (where all steps have null videoPath)
      const filteredRows = rows.filter(row => 
        row.steps.some(step => step.videoPath)
      );

      // Get readable label for selected seed
      const selectedSeedLabel = getVideoIdLabel(selectedSeed, videoMetadataMap);

      return {
        rows: filteredRows,
        columnHeaders: allSteps.map(stepId => `Step ${stepId.replace('step_', '')}`),
        rowType: 'prompt',
        selectedItem: selectedSeedLabel
      };
    } else {
      // Latent videos only - each row is a different seed
      const rows = allSeeds.map(seedId => {
        const readableLabel = getVideoIdLabel(seedId, videoMetadataMap);
        return {
          id: seedId,
          label: readableLabel,
          fullLabel: readableLabel, // Already formatted
          steps: allSteps.map(stepId => {
            const { videoPath, imagePath } = getVideoData(selectedPrompt, seedId, stepId, null);
            return {
              stepId,
              stepNumber: parseInt(stepId.replace('step_', '')),
              videoPath,
              imagePath,
            };
          })
        };
      });

      // Filter out rows with no videos (where all steps have null videoPath)
      const filteredRows = rows.filter(row => 
        row.steps.some(step => step.videoPath)
      );

      // Get prompt text from video metadata of first seed
      const firstSeed = allSeeds[0];
      const videoMetadata = videoMetadataMap[firstSeed];
      const selectedPromptDisplay = videoMetadata?.prompt_variation || getPromptDisplayText(selectedPrompt).display;

      return {
        rows: filteredRows,
        columnHeaders: allSteps.map(stepId => `Step ${stepId.replace('step_', '')}`),
        rowType: 'seed',
        selectedItem: selectedPromptDisplay
      };
    }
  }, [currentLatentVideos, viewMode, selectedPrompt, selectedSeed, selectedAttentionTokens, attentionViewMode, selectedLayer, selectedHead]);

  // Get available options for dropdowns
  const availableOptions = useMemo(() => {
    // Check if we have ANY data to display
    const hasLatentVideos = currentLatentVideos?.latent_videos;
    const hasAttentionVideos = currentLatentVideos?.attention_videos;

    if (!hasLatentVideos && !hasAttentionVideos) {
      return { prompts: [], seeds: [], tokens: [], layers: [], heads: [], tiersAvailable: [] };
    }

    // Use whichever data source is available
    const dataSource = hasLatentVideos ? currentLatentVideos.latent_videos : currentLatentVideos.attention_videos;

    let prompts, seeds;

    if (hasLatentVideos) {
      prompts = Object.keys(dataSource).sort();
      seeds = prompts.length > 0 ? Object.keys(dataSource[prompts[0]]).sort() : [];
    } else {
      // For attention videos, need to navigate through token structure
      prompts = Object.keys(dataSource).sort();
      if (prompts.length > 0) {
        seeds = Object.keys(dataSource[prompts[0]]).sort();
      } else {
        seeds = [];
      }
    }

    // Get available attention tokens
    const attentionVideos = currentLatentVideos.attention_videos;
    const tokens = [];

    if (attentionVideos) {
      const tokenSet = new Set();
      Object.values(attentionVideos).forEach(promptData => {
        Object.values(promptData).forEach(videoData => {
          Object.keys(videoData).forEach(token => {
            tokenSet.add(token);
          });
        });
      });
      tokens.push(...Array.from(tokenSet).sort());
    }

    // Get tier information for attention videos
    const layers = currentLatentVideos?.available_layers || [];
    const heads = currentLatentVideos?.available_heads || [];
    const tiersAvailable = currentLatentVideos?.tiers_available || [];

    return { prompts, seeds, tokens, layers, heads, tiersAvailable };
  }, [currentLatentVideos]);

  const handleCellClick = (cellData, rowIndex, stepIndex) => {
    // Open lightbox at the clicked position
    setLightbox({
      isOpen: true,
      position: { rowIndex, stepIndex }
    });
  };

  const handleTooltipShow = (event, text) => {
    const rect = event.target.getBoundingClientRect();
    setTooltip({
      show: true,
      text: text,
      x: rect.right + 10,
      y: rect.top + rect.height / 2
    });
  };

  const handleTooltipHide = () => {
    setTooltip({ show: false, text: '', x: 0, y: 0 });
  };

  const handleLightboxClose = () => {
    setLightbox({ isOpen: false, position: null });
  };

  const handleLightboxNavigate = (newPosition) => {
    setLightbox({ isOpen: true, position: newPosition });
  };

  // Render loading state
  if (latentVideosLoading) {
    return (
      <div className="latent-videos-view">
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Loading latent videos...</p>
        </div>
      </div>
    );
  }

  // Render error state
  if (latentVideosError) {
    return (
      <div className="latent-videos-view">
        <div className="error-message">
          <h3>Error</h3>
          <p>{latentVideosError}</p>
        </div>
      </div>
    );
  }

  // Render no data state
  if (!currentLatentVideos?.has_latent_videos && !currentLatentVideos?.has_attention_videos) {
    return (
      <div className="latent-videos-view">
        <div className="empty-state">
          <h3>No latent or attention videos available</h3>
          <p>This experiment doesn't contain latent video or attention video data.</p>
        </div>
      </div>
    );
  }

  if (!gridData) {
    return (
      <div className="latent-videos-view">
        <div className="empty-state">
          <h3>No video data available</h3>
          <p>Unable to organize the latent video data.</p>
        </div>
      </div>
    );
  }

  return (
    <div
      className="latent-videos-view"
      style={{
        '--cell-size': `${cellSize}px`
      }}
    >
      {/* Fixed header with controls */}
      <div className="latent-videos-header-fixed">
        <div className="controls-container">
          {/* Collapsible controls toggle */}
          <button
            className="controls-toggle-btn"
            onClick={() => setControlsExpanded(!controlsExpanded)}
            aria-expanded={controlsExpanded}
          >
            <span className="toggle-icon">{controlsExpanded ? '▼' : '▶'}</span>
            <span className="toggle-label">Controls</span>
            {!controlsExpanded && (
              <span className="controls-summary">{getControlsSummary()}</span>
            )}
          </button>

          {/* Collapsible section */}
          {controlsExpanded && (
            <div className="controls-collapsible-content">
              {/* Info banner if latent videos missing but attention videos available */}
              {/* {!currentLatentVideos?.has_latent_videos && currentLatentVideos?.has_attention_videos && (
                <div className="info-banner">
                  ℹ️ Latent videos not decoded. Showing attention videos only. Select a token from the "Attention" dropdown to view.
                </div>
              )} */}

              {/* Main controls row */}
              <div className="latent-videos-controls">
                {/* View mode toggle */}
                <div className="control-group">
                  <label>View Mode:</label>
                  <div className="toggle-group">
                    <button
                      className={`toggle-btn ${viewMode === 'across-prompts' ? 'active' : ''}`}
                      onClick={() => setViewMode('across-prompts')}
                    >
                      Across Prompts
                    </button>
                    <button
                      className={`toggle-btn ${viewMode === 'by-prompt' ? 'active' : ''}`}
                      onClick={() => setViewMode('by-prompt')}
                    >
                      By Prompt
                    </button>
                  </div>
                </div>

                {/* Selection dropdown */}
                <div className="control-group">
                  {viewMode === 'across-prompts' ? (
                    <>
                      <label htmlFor="seed-select">Seed:</label>
                      <select
                        id="seed-select"
                        value={selectedSeed}
                        onChange={(e) => setSelectedSeed(e.target.value)}
                      >
                        {availableOptions.seeds.map(seedId => {
                          // Extract seed number from seedId (e.g., "p000_b000_s001" -> "001")
                          const seedMatch = seedId.match(/_s(\d+)$/) || seedId.match(/vid_(\d+)/);
                          const seedNum = seedMatch ? seedMatch[1] : seedId;
                          const displayText = `Seed ${seedNum}`;
                          return (
                            <option key={seedId} value={seedId} title={seedId}>
                              {displayText}
                            </option>
                          );
                        })}
                      </select>
                    </>
                  ) : (
                    <>
                      <label htmlFor="prompt-select">Prompt:</label>
                      <select
                        id="prompt-select"
                        value={selectedPrompt}
                        onChange={(e) => setSelectedPrompt(e.target.value)}
                      >
                        {availableOptions.prompts.map(promptId => {
                          // Get prompt text from metadata if available
                          const metadataMap = currentLatentVideos?.video_metadata_map || {};
                          const dataSource = currentLatentVideos?.latent_videos || currentLatentVideos?.attention_videos || {};
                          const firstSeed = Object.keys(dataSource[promptId] || {})[0];
                          const seedMetadata = firstSeed ? metadataMap[firstSeed] : null;
                          const promptText = seedMetadata?.prompt_variation || getPromptDisplayText(promptId).display;
                          const fullPrompt = seedMetadata?.prompt_variation || getPromptDisplayText(promptId).full;
                          return (
                            <option key={promptId} value={promptId} title={fullPrompt}>
                              {promptText}
                            </option>
                          );
                        })}
                      </select>
                    </>
                  )}
                </div>

                {/* Attention token multi-select dropdown */}
                {availableOptions.tokens.length > 0 && (
                  <div className="control-group attention-tokens-group">
                    <label>
                      Attention Tokens:
                      {!currentLatentVideos?.has_latent_videos && (
                        <span style={{ color: '#ffc107', marginLeft: '4px' }}>*</span>
                      )}
                    </label>
                    <div className="tokens-dropdown-container" ref={tokensDropdownRef}>
                      <button
                        className="tokens-dropdown-toggle"
                        onClick={() => setTokensExpanded(!tokensExpanded)}
                        type="button"
                      >
                        <span className="tokens-summary">
                          {selectedAttentionTokens.length === 0
                            ? 'None selected'
                            : selectedAttentionTokens.join(', ')}
                        </span>
                        <span className="dropdown-icon">{tokensExpanded ? '▲' : '▼'}</span>
                      </button>
                      {tokensExpanded && (
                        <div className="tokens-multiselect">
                          {availableOptions.tokens.map(token => {
                            const isChecked = selectedAttentionTokens.includes(token);
                            return (
                              <label key={token} className="token-checkbox" title={token}>
                                <input
                                  type="checkbox"
                                  checked={isChecked}
                                  onChange={(e) => {
                                    if (e.target.checked) {
                                      setSelectedAttentionTokens([...selectedAttentionTokens, token]);
                                    } else {
                                      setSelectedAttentionTokens(selectedAttentionTokens.filter(t => t !== token));
                                    }
                                  }}
                                />
                                <span className="token-label">{token.length > 15 ? `${token.substring(0, 15)}...` : token}</span>
                              </label>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Attention view mode - only show if attention tokens selected and higher tiers available */}
                {selectedAttentionTokens.length > 0 && (availableOptions.tiersAvailable.includes(2) || availableOptions.tiersAvailable.includes(3)) && (
                  <div className="control-group">
                    <label>Attention View:</label>
                    <div className="toggle-group">
                      <button
                        className={`toggle-btn ${attentionViewMode === 'default' ? 'active' : ''}`}
                        onClick={() => setAttentionViewMode('default')}
                        title="Show averaged attention (all layers & heads)"
                      >
                        Default
                      </button>
                      {availableOptions.tiersAvailable.includes(2) && (
                        <button
                          className={`toggle-btn ${attentionViewMode === 'by-layer' ? 'active' : ''}`}
                          onClick={() => setAttentionViewMode('by-layer')}
                          title="Show attention by layer"
                        >
                          By Layer
                        </button>
                      )}
                      {availableOptions.tiersAvailable.includes(3) && (
                        <button
                          className={`toggle-btn ${attentionViewMode === 'by-head' ? 'active' : ''}`}
                          onClick={() => setAttentionViewMode('by-head')}
                          title="Show attention by layer and head"
                        >
                          By Head
                        </button>
                      )}
                    </div>
                  </div>
                )}

                {/* Cell size slider */}
                <div className="control-group">
                  <label htmlFor="cell-size-slider">Size:</label>
                  <input
                    id="cell-size-slider"
                    type="range"
                    min="5"
                    max="300"
                    value={cellSize}
                    onChange={(e) => setCellSize(parseInt(e.target.value))}
                  />
                  <span className="slider-value">{cellSize}px</span>
                </div>
              </div>

              {/* Layer/Head filter row - separate row below main controls */}
              {selectedAttentionTokens.length > 0 && (attentionViewMode === 'by-layer' || attentionViewMode === 'by-head') && (
                <div className="latent-videos-controls layer-head-filters">
                  {/* Layer selection - show if by-layer or by-head mode and layers available */}
                  {availableOptions.layers.length > 0 && (
                    <div className="control-group">
                      <label htmlFor="layer-select">Layer:</label>
                      <select
                        id="layer-select"
                        value={selectedLayer}
                        onChange={(e) => setSelectedLayer(e.target.value === 'averaged' ? 'averaged' : parseInt(e.target.value))}
                      >
                        <option value="averaged">Averaged (All Layers)</option>
                        {availableOptions.layers.map(layer => (
                          <option key={layer} value={layer}>
                            Layer {layer}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}

                  {/* Head selection - show if by-head mode and heads available */}
                  {attentionViewMode === 'by-head' && availableOptions.heads.length > 0 && (
                    <div className="control-group">
                      <label htmlFor="head-select">Head:</label>
                      <select
                        id="head-select"
                        value={selectedHead}
                        onChange={(e) => setSelectedHead(e.target.value === 'averaged' ? 'averaged' : parseInt(e.target.value))}
                      >
                        <option value="averaged">Averaged (All Heads)</option>
                        {availableOptions.heads.map(head => (
                          <option key={head} value={head}>
                            Head {head}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}

                  {/* View info - only show in hierarchical view (by-layer or by-head with expansion) */}
                  {gridData.rowType === 'hierarchical' && (
                    <div className="view-info">
                      {gridData.selectedItem}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Scrollable grid container */}
      <div className="latent-videos-grid-container">
        {/* Column headers */}
        <div className="grid-headers">
          <div className="row-label-header">
            {gridData.rowType === 'hierarchical'
              ? 'Layers'
              : (viewMode === 'across-prompts' ? 'Prompts' : 'Seeds')}
          </div>
          {gridData.columnHeaders.map(header => (
            <div key={header} className="column-header">
              {header}
            </div>
          ))}
        </div>

        {/* Grid rows */}
        <div className="grid-rows">
          {gridData.rowType === 'hierarchical' ? (
            // Hierarchical view with sections
            gridData.sections.map((section, sectionIndex) => (
              <div key={`${section.promptId}_${section.tokenId}`} className="grid-section">
                {/* Prompt header */}
                <div className="section-header prompt-header">
                  <div className="section-header-label">{section.promptLabel}</div>
                  <div className="section-header-spacer"></div>
                </div>

                {/* Token header */}
                <div className="section-header token-header">
                  <div className="section-header-label">{section.tokenLabel}</div>
                  <div className="section-header-spacer"></div>
                </div>

                {/* Layer rows */}
                {section.rows.map((row, rowIndex) => {
                  const globalRowIndex = gridData.sections.slice(0, sectionIndex).reduce((sum, s) => sum + s.rows.length, 0) + rowIndex;
                  return (
                    <div key={row.id} className="grid-row">
                      <div
                        className="row-label"
                        onMouseEnter={(e) => handleTooltipShow(e, row.fullLabel)}
                        onMouseLeave={handleTooltipHide}
                      >
                        {row.label}
                      </div>
                      {row.steps.map((step, stepIndex) => (
                        <div key={step.stepId} className="grid-cell">
                          <LatentVideoCell
                            videoPath={step.videoPath}
                            imagePath={step.imagePath}
                            videoSize={cellSize}
                            stepNumber={step.stepNumber}
                            onClick={() => handleCellClick(step, globalRowIndex, stepIndex)}
                          />
                        </div>
                      ))}
                    </div>
                  );
                })}
              </div>
            ))
          ) : (
            // Flat view (legacy)
            gridData.rows.map((row, rowIndex) => (
              <div key={row.id} className="grid-row">
                <div
                  className="row-label"
                  onMouseEnter={(e) => handleTooltipShow(e, row.fullLabel)}
                  onMouseLeave={handleTooltipHide}
                >
                  {row.label}
                </div>
                {row.steps.map((step, stepIndex) => (
                  <div key={step.stepId} className="grid-cell">
                    <LatentVideoCell
                      videoPath={step.videoPath}
                      imagePath={step.imagePath}
                      videoSize={cellSize}
                      stepNumber={step.stepNumber}
                      onClick={() => handleCellClick(step, rowIndex, stepIndex)}
                    />
                  </div>
                ))}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Custom tooltip */}
      {tooltip.show && (
        <div
          className="custom-tooltip"
          style={{
            position: 'fixed',
            left: `${tooltip.x}px`,
            top: `${tooltip.y}px`,
            transform: 'translateY(-50%)',
            zIndex: 10000,
            pointerEvents: 'none'
          }}
        >
          {tooltip.text}
        </div>
      )}

      {/* Lightbox */}
      <LatentVideoLightbox
        isOpen={lightbox.isOpen}
        onClose={handleLightboxClose}
        onNavigate={handleLightboxNavigate}
        currentPosition={lightbox.position}
        gridData={gridData}
        currentLatentVideos={currentLatentVideos}
        currentExperiment={currentExperiment}
        selectedAttentionTokens={selectedAttentionTokens}
        setSelectedAttentionTokens={setSelectedAttentionTokens}
        availableTokens={availableOptions.tokens}
        viewMode={viewMode}
      />
    </div>
  );
};

export default LatentVideosView;
