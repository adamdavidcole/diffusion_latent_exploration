// API service for WAN Video Matrix Viewer

const API_BASE = '';

export const api = {
  async getExperiments() {
    console.log('API: Fetching experiments...');
    try {
      const response = await fetch(`${API_BASE}/api/experiments`);
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

  async fetchVideoBlob(videoPath) {
    const response = await fetch(this.getVideoUrl(videoPath));
    if (!response.ok) {
      throw new Error(`Failed to fetch video: ${response.status}`);
    }
    return response.blob();
  }
};
