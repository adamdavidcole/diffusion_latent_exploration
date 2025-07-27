// API service for WAN Video Matrix Viewer

// Environment-aware API base URL
const getApiBase = () => {
  // Check if we're in development (Vite dev server)
  if (import.meta.env.DEV) {
    return ''; // Use proxy in development
  }
  
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

export const api = {
  async getExperiments() {
    console.log('API: Fetching experiments...');
    try {
      const response = await fetchWithTimeout(`${API_BASE}/api/experiments`);
      console.log('API: Experiments response status:', response.status);
      if (!response.ok) {
        throw new Error(`Failed to fetch experiments: ${response.status}`);
      }
      const data = await response.json();
      console.log('API: Experiments data:', data);
      return data;
    } catch (error) {
      console.error('API: Error fetching experiments:', error);
      throw error;
    }
  },

  async getExperiment(experimentName) {
    const response = await fetch(`${API_BASE}/api/experiment/${experimentName}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch experiment: ${response.status}`);
    }
    const experiment = await response.json();
    if (experiment.error) {
      throw new Error(experiment.error);
    }
    return experiment;
  },

  async scanExperiments() {
    const response = await fetch(`${API_BASE}/api/scan`);
    if (!response.ok) {
      throw new Error(`Failed to scan experiments: ${response.status}`);
    }
    return response.json();
  },

  getVideoUrl(videoPath) {
    return `${API_BASE}/api/video/${videoPath}`;
  },

  async fetchVideoBlob(videoPath, signal = null) {
    const options = signal ? { signal } : {};
    const response = await fetch(this.getVideoUrl(videoPath), options);
    if (!response.ok) {
      throw new Error(`Failed to fetch video: ${response.status}`);
    }
    return response.blob();
  }
};
