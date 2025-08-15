// Utility functions for transforming individual VLM analysis data to aggregated format

// Configuration for which fields to display in the lightbox
export const LIGHTBOX_ANALYSIS_CONFIG = {
  demographics: [
    'age',
    'gender', 
    'race_ethnicity',
    'national_or_cultural_identity'
  ],
  appearance: [
    'clothing_style',
    'body_type',
    'attractiveness_coding',
    'socioeconomic_status',
    'nudity_level'
  ]
};

// Helper to count values across people for a specific field
const countFieldValues = (people, fieldPath) => {
  const counts = {};
  let totalResponses = 0;

  people.forEach(person => {
    // Navigate to the nested field value
    const value = getNestedValue(person, fieldPath);
    if (value !== null && value !== undefined && value !== '') {
      counts[value] = (counts[value] || 0) + 1;
      totalResponses++;
    }
  });

  // Calculate percentages
  const percentages = {};
  Object.keys(counts).forEach(key => {
    percentages[key] = totalResponses > 0 ? (counts[key] / totalResponses) * 100 : 0;
  });

  return {
    total_responses: totalResponses,
    data: {
      counts,
      percentages
    }
  };
};

// Helper to get nested object value by dot notation path
const getNestedValue = (obj, path) => {
  return path.split('.').reduce((current, key) => {
    return current && current[key] !== undefined ? current[key] : null;
  }, obj);
};

// Transform individual VLM data to match aggregated format for specific sections
export const transformIndividualToAggregated = (individualData, schema) => {
  if (!individualData || !individualData.people || !Array.isArray(individualData.people)) {
    return null;
  }

  const { people } = individualData;
  const transformed = {
    people: {
      sections: {
        demographics: {},
        appearance: {}
      }
    },
    cultural_flags: {}
  };

  // Transform demographics
  if (LIGHTBOX_ANALYSIS_CONFIG.demographics) {
    LIGHTBOX_ANALYSIS_CONFIG.demographics.forEach(field => {
      const fieldPath = `demographics.${field}`;
      const fieldData = countFieldValues(people, fieldPath);
      
      // Add type information from schema if available
      // Schema has people as an array with one element
      if (schema && schema.people && Array.isArray(schema.people) && schema.people[0] && schema.people[0].demographics && schema.people[0].demographics[field]) {
        fieldData.type = schema.people[0].demographics[field].type;
      } else {
        fieldData.type = 'options'; // default assumption
      }
      
      transformed.people.sections.demographics[field] = fieldData;
    });
  }

  // Transform appearance
  if (LIGHTBOX_ANALYSIS_CONFIG.appearance) {
    LIGHTBOX_ANALYSIS_CONFIG.appearance.forEach(field => {
      const fieldPath = `appearance.${field}`;
      const fieldData = countFieldValues(people, fieldPath);
      
      // Add type information from schema if available
      // Schema has people as an array with one element
      if (schema && schema.people && Array.isArray(schema.people) && schema.people[0] && schema.people[0].appearance && schema.people[0].appearance[field]) {
        fieldData.type = schema.people[0].appearance[field].type;
      } else {
        fieldData.type = 'options'; // default assumption
      }
      
      transformed.people.sections.appearance[field] = fieldData;
    });
  }

  // Transform cultural flags - just pass through as raw data for text display
  if (individualData.cultural_flags) {
    transformed.cultural_flags = individualData.cultural_flags;
  }

  return transformed;
};

// Format cultural flags for display
export const formatCulturalFlags = (culturalFlags) => {
  if (!culturalFlags) return [];

  const flagSections = [
    {
      title: 'Male Gaze & Objectification',
      fields: [
        { key: 'male_gaze_objectification_level', label: 'Level' },
        { key: 'male_gaze_objectification_context', label: 'Context' }
      ]
    },
    {
      title: 'Sexualization',
      fields: [
        { key: 'sexualization_level', label: 'Level' },
        { key: 'sexualization_context', label: 'Context' }
      ]
    },
    {
      title: 'Racial Bias',
      fields: [
        { key: 'racial_stereotype_bias_level', label: 'Level' },
        { key: 'racial_stereotype_bias_context', label: 'Context' }
      ]
    },
    {
      title: 'Class Bias',
      fields: [
        { key: 'class_bias_level', label: 'Level' },
        { key: 'class_bias_context', label: 'Context' }
      ]
    },
    {
      title: 'Respectability & Virtue Coding',
      fields: [
        { key: 'respectability_virtue_coding_level', label: 'Level' },
        { key: 'respectability_virtue_coding_context', label: 'Context' }
      ]
    },
    {
      title: 'Indecency Cues',
      fields: [
        { key: 'indecency_cues_level', label: 'Level' },
        { key: 'indecency_cues_context', label: 'Context' }
      ]
    },
    {
      title: 'Cinematic Tropes',
      fields: [
        { key: 'cinematic_tropes_level', label: 'Level' },
        { key: 'cinematic_tropes_context', label: 'Context' }
      ]
    },
    {
      title: 'Cultural Heritage Identity',
      fields: [
        { key: 'cultural_heritage_identity_level', label: 'Level' },
        { key: 'cultural_heritage_identity_context', label: 'Context' }
      ]
    },
    {
      title: 'Violence Cues',
      fields: [
        { key: 'violence_cues', label: 'Level' },
        { key: 'violence_cues_context', label: 'Context' }
      ]
    }
  ];

  return flagSections.map(section => ({
    ...section,
    fields: section.fields.map(field => ({
      ...field,
      value: culturalFlags[field.key] || 'Not specified'
    }))
  }));
};

// Get the VLM analysis JSON path for a given video
export const getVlmAnalysisPath = (video) => {
    console.log("video", video)
  if (!video || !video.video_path) return null;

  let vlmPath = video.video_path;

  vlmPath = vlmPath.replace("/videos/", "/vlm_analysis/");
  vlmPath = vlmPath.replace(".mp4", ".json");
  return vlmPath;
  // Extract experiment path and video details from video path
  // Expected format: outputs/experiment_name/videos/prompt_XXX/video_XXX.mp4
//   const pathParts = video.video_path.split('/');

//   console.log("pathParts", pathParts)

//   if (pathParts.length < 5) return null;
  
//   const experimentPath = pathParts.slice(0, 2).join('/'); // outputs/experiment_name
//   const promptDir = pathParts[3]; // prompt_XXX
//   const videoFile = pathParts[4]; // video_XXX.mp4
  
  // Convert video file to JSON
  const videoJson = videoFile.replace('.mp4', '.json');
  
//   return `${experimentPath}/vlm_analysis/${promptDir}/${videoJson}`;
};
