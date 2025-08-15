// API service for WAN Video Matrix Viewer

// Environment-aware API base URL
const getApiBase = () => {
  // Check if we're in development (Vite dev server)
  if (import.meta.env.DEV) {
    return ''; // Use proxy in development
  }

  // return "http://127.0.0.1:8888"
  
  // Production - use environment variable or default to your university server
  return import.meta.env.VITE_API_BASE_URL || 'https://acole9.pythonanywhere.com';
};

const API_BASE = getApiBase();

// Helper function to add timeout to fetch requests
const fetchWithTimeout = (url, options = {}, timeout = 10000) => {
  return Promise.race([
    fetch(url, options),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Request timeout')), timeout)
    )
  ]);
};

export const getVideoUrl = (videoPath) => {
    return `${getApiBase()}/media/${videoPath}`;
};

export const getThumbnailUrl = (imgPath) => {
  // Convert .mp4 to .jpg for thumbnail
  const thumbnailPath = imgPath.endsWith('.mp4') ? imgPath.replace('.mp4', '.jpg') : imgPath;
  return `${getApiBase()}/media/${thumbnailPath}`;
}

export const api = {
  async getExperiments() {
    console.log('API: Fetching experiments tree...');
    try {
      const response = await fetchWithTimeout(`${API_BASE}/api/experiments`);
      console.log('API: Experiments response status:', response.status);
      if (!response.ok) {
        throw new Error(`Failed to fetch experiments: ${response.status}`);
      }
      const data = await response.json();
      console.log('API: Experiments tree data:', data);
      return data;
    } catch (error) {
      console.error('API: Error fetching experiments:', error);
      throw error;
    }
  },

  async getExperiment(experimentPath) {
    console.log('API: Fetching experiment:', experimentPath);
    const response = await fetch(`${API_BASE}/api/experiment/${experimentPath}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch experiment: ${response.status}`);
    }
    const experiment = await response.json();
    if (experiment.error) {
      throw new Error(experiment.error);
    }
    return experiment;
  },

  async getExperimentAnalysis(experimentPath) {
    console.log('API: Fetching experiment analysis:', experimentPath);
    const response = await fetch(`${API_BASE}/api/experiment/${experimentPath}/analysis`);
    if (!response.ok) {
      throw new Error(`Failed to fetch experiment analysis: ${response.status}`);
    }
    const analysisData = await response.json();
    if (analysisData.error) {
      throw new Error(analysisData.error);
    }
    return analysisData;
  },

  // Helper function to flatten tree into list for search/filtering
  flattenExperimentTree(tree) {
    const experiments = [];
    
    const traverse = (node, path = '') => {
      if (node.type === 'experiment') {
        experiments.push({
          ...node.experiment_data,
          path: node.path,
          hierarchical_name: path ? `${path}/${node.name}` : node.name
        });
      } else if (node.children) {
        const currentPath = path ? `${path}/${node.name}` : node.name;
        node.children.forEach(child => {
          traverse(child, node.name === 'outputs' ? '' : currentPath);
        });
      }
    };
    
    traverse(tree);
    return experiments;
  },

  async scanExperiments() {
    const response = await fetch(`${API_BASE}/api/scan`);
    if (!response.ok) {
      throw new Error(`Failed to scan experiments: ${response.status}`);
    }
    return response.json();
  },

  async fetchVideoBlob(videoPath, signal = null) {
    const options = signal ? { signal } : {};
    const response = await fetch(getVideoUrl(videoPath), options);
    if (!response.ok) {
      throw new Error(`Failed to fetch video: ${response.status}`);
    }
    return response.blob();
  }
};
