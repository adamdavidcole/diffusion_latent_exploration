import React, { useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { api } from '../../services/api';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { Bar, Pie } from 'react-chartjs-2';
import { WordCloudChart } from 'chartjs-chart-wordcloud';
import './AnalysisDashboard.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  WordCloudChart
);

const AnalysisDashboard = ({ experimentPath }) => {
  const { state, actions } = useApp();
  const { 
    currentAnalysis, 
    analysisLoading, 
    analysisError, 
    currentExperiment, 
    analysisSchema,
    analysisViewBy 
  } = state;

    // Load analysis data when component mounts or experiment changes
  useEffect(() => {
    const loadAnalysisData = async () => {
      if (!experimentPath || !currentExperiment?.has_vlm_analysis) {
        return;
      }

      try {
        actions.setAnalysisLoading(true);
        actions.clearAnalysisError();
        
        const analysisData = await api.getExperimentAnalysis(experimentPath);
        actions.setCurrentAnalysis(analysisData);
      } catch (error) {
        console.error('Error loading analysis data:', error);
        actions.setAnalysisError(`Failed to load analysis data: ${error.message}`);
      } finally {
        actions.setAnalysisLoading(false);
      }
    };

    loadAnalysisData();
  }, [experimentPath, currentExperiment?.has_vlm_analysis]); // Removed actions from dependencies

  // Helper function to get metrics in schema order
  const getMetricsInSchemaOrder = () => {
    if (!analysisSchema || !currentAnalysis?.vlm_analysis?.prompt_groups) {
      return [];
    }

    const metrics = new Set();
    
    // Collect all metrics from analysis data
    Object.values(currentAnalysis.vlm_analysis.prompt_groups).forEach(group => {
      Object.keys(group).forEach(metric => {
        if (metric !== 'prompt_group_id') {
          metrics.add(metric);
        }
      });
    });

    // Order metrics based on schema structure
    const orderedMetrics = [];
    
    // Demographics metrics
    if (analysisSchema.people?.[0]?.demographics) {
      Object.keys(analysisSchema.people[0].demographics).forEach(metric => {
        if (metrics.has(metric)) {
          orderedMetrics.push(metric);
        }
      });
    }
    
    // Appearance metrics
    if (analysisSchema.people?.[0]?.appearance) {
      Object.keys(analysisSchema.people[0].appearance).forEach(metric => {
        if (metrics.has(metric)) {
          orderedMetrics.push(metric);
        }
      });
    }
    
    // Role and agency metrics
    if (analysisSchema.people?.[0]?.role_and_agency) {
      Object.keys(analysisSchema.people[0].role_and_agency).forEach(metric => {
        if (metrics.has(metric)) {
          orderedMetrics.push(metric);
        }
      });
    }
    
    // Composition metrics
    if (analysisSchema.composition) {
      Object.keys(analysisSchema.composition).forEach(metric => {
        if (metrics.has(metric)) {
          orderedMetrics.push(metric);
        }
      });
    }
    
    // Add any remaining metrics not in schema
    Array.from(metrics).forEach(metric => {
      if (!orderedMetrics.includes(metric)) {
        orderedMetrics.push(metric);
      }
    });
    
    return orderedMetrics;
  };

  // Helper function to generate charts based on schema type
  const generateChart = (metricName, data, schemaInfo, promptGroupId) => {
    if (!data || !schemaInfo) return null;

    const chartId = `chart-${promptGroupId}-${metricName}`;
    
    if (schemaInfo.type === 'options') {
      // Pie chart for options
      const chartData = {
        labels: Object.keys(data.counts || {}),
        datasets: [{
          data: Object.values(data.counts || {}),
          backgroundColor: [
            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', 
            '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'
          ],
          borderWidth: 1,
        }]
      };

      const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              boxWidth: 12,
              font: { size: 10 },
              color: '#ffffff'
            }
          },
          tooltip: {
            callbacks: {
              label: (context) => {
                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                const percentage = ((context.parsed / total) * 100).toFixed(1);
                return `${context.label}: ${context.parsed} (${percentage}%)`;
              }
            }
          }
        }
      };

      return (
        <div className="chart-container" key={chartId}>
          <h6 className="chart-title">{metricName}</h6>
          <div className="chart-wrapper">
            <Pie data={chartData} options={options} />
          </div>
        </div>
      );
    }
    
    if (schemaInfo.type === 'range') {
      // Bar chart for ranges
      const chartData = {
        labels: Object.keys(data.counts || {}),
        datasets: [{
          label: metricName,
          data: Object.values(data.counts || {}),
          backgroundColor: '#36A2EB',
          borderColor: '#36A2EB',
          borderWidth: 1,
        }]
      };

      const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => `${context.label}: ${context.parsed.y}`
            }
          }
        },
        scales: {
          x: {
            ticks: { color: '#ffffff', font: { size: 10 } }
          },
          y: {
            ticks: { color: '#ffffff', font: { size: 10 } }
          }
        }
      };

      return (
        <div className="chart-container" key={chartId}>
          <h6 className="chart-title">{metricName}</h6>
          <div className="chart-wrapper">
            <Bar data={chartData} options={options} />
          </div>
        </div>
      );
    }
    
    if (schemaInfo.type === 'open') {
      // Word cloud for open text (simplified as text list for now)
      const words = Object.entries(data.counts || {}).map(([word, count]) => 
        `${word} (${count})`
      ).join(', ');

      return (
        <div className="chart-container" key={chartId}>
          <h6 className="chart-title">{metricName}</h6>
          <div className="word-cloud-container">
            <p className="word-cloud-text">{words || 'No data available'}</p>
          </div>
        </div>
      );
    }

    return null;
  };

  // Helper function to get metric data from prompt group data
  const getMetricData = (promptData, metric) => {
    try {
      // Navigate through the nested data structure to find the metric
      if (promptData.aggregated_data?.people?.sections?.demographics?.[metric]?.data) {
        return promptData.aggregated_data.people.sections.demographics[metric].data;
      }
      if (promptData.aggregated_data?.people?.sections?.appearance?.[metric]?.data) {
        return promptData.aggregated_data.people.sections.appearance[metric].data;
      }
      if (promptData.aggregated_data?.people?.sections?.role_and_agency?.[metric]?.data) {
        return promptData.aggregated_data.people.sections.role_and_agency[metric].data;
      }
      if (promptData.aggregated_data?.composition?.[metric]?.data) {
        return promptData.aggregated_data.composition[metric].data;
      }
      return null;
    } catch (error) {
      console.error(`Error accessing metric ${metric}:`, error);
      return null;
    }
  };

  // Helper function to get schema info for a metric
  const getSchemaInfo = (metric) => {
    if (!analysisSchema) return null;
    
    // Check demographics
    if (analysisSchema.people?.[0]?.demographics?.[metric]) {
      return analysisSchema.people[0].demographics[metric];
    }
    // Check appearance
    if (analysisSchema.people?.[0]?.appearance?.[metric]) {
      return analysisSchema.people[0].appearance[metric];
    }
    // Check role_and_agency
    if (analysisSchema.people?.[0]?.role_and_agency?.[metric]) {
      return analysisSchema.people[0].role_and_agency[metric];
    }
    // Check composition
    if (analysisSchema.composition?.[metric]) {
      return analysisSchema.composition[metric];
    }
    
    return null;
  };

  // Helper function to safely get gender counts
  const getGenderCounts = (promptGroupData) => {
    try {
      return promptGroupData?.aggregated_data?.people?.sections?.demographics?.gender?.data?.counts || {};
    } catch (error) {
      console.error('Error accessing gender data:', error);
      return {};
    }
  };

  // Helper function to get variation name from prompt group
  const getVariationName = (promptGroupKey, promptGroupData) => {
    // Try to get variation name from the experiment's video grid
    if (currentExperiment?.video_grid) {
      const matchingVariation = currentExperiment.video_grid.find(v => 
        v.variation_id === promptGroupKey || 
        v.variation_id === `variation_${promptGroupKey.split('_')[1]}`
      );
      if (matchingVariation) {
        return matchingVariation.variation;
      }
    }
    
    // Fallback to prompt group key
    return promptGroupKey.replace('prompt_', 'Prompt ');
  };

  if (!currentExperiment?.has_vlm_analysis) {
    return (
      <div className="analysis-dashboard">
        <div className="analysis-unavailable">
          <h3>VLM Analysis Not Available</h3>
          <p>This experiment does not have VLM analysis data.</p>
          <p>To generate analysis data, run the VLM analysis script on this experiment.</p>
        </div>
      </div>
    );
  }

  if (analysisLoading) {
    return (
      <div className="analysis-dashboard">
        <div className="analysis-loading">
          <div className="loading-spinner"></div>
          <p>Loading VLM analysis data...</p>
        </div>
      </div>
    );
  }

  if (analysisError) {
    return (
      <div className="analysis-dashboard">
        <div className="analysis-error">
          <h3>Error Loading Analysis</h3>
          <p>{analysisError}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="retry-button"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!currentAnalysis?.vlm_analysis) {
    return (
      <div className="analysis-dashboard">
        <div className="analysis-unavailable">
          <h3>No Analysis Data</h3>
          <p>VLM analysis data is not available for this experiment.</p>
        </div>
      </div>
    );
  }

  const { prompt_groups, overall } = currentAnalysis.vlm_analysis;
  // Helper function to get prompt variation text
  const getPromptVariationText = (promptKey) => {
    // Extract variation part from prompt group name
    // e.g., "p0_black-woman-professional" -> "black woman professional"
    if (typeof promptKey === 'string' && promptKey.includes('_')) {
      const parts = promptKey.split('_');
      if (parts.length > 1) {
        return parts.slice(1).join(' ').replace(/-/g, ' ');
      }
    }
    return promptKey || 'Unknown';
  };
  
  // Get ordered metrics and prompt groups
  const orderedMetrics = getMetricsInSchemaOrder();
  const promptGroupKeys = Object.keys(prompt_groups);

  return (
    <div className="analysis-dashboard">
      <div className="analysis-content">
        <h3>VLM Analysis Results</h3>
        
        {/* Overall Statistics */}
        {overall && (
          <div className="analysis-section">
            <h4>Overall Experiment Statistics</h4>
            <div className="overall-stats">
              <div className="stat-card">
                <span className="stat-label">Total Videos</span>
                <span className="stat-value">{overall.metadata?.successfully_loaded || 0}</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Total People</span>
                <span className="stat-value">{overall.aggregated_data?.people?.total_people || 0}</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Female Count</span>
                <span className="stat-value">
                  {getGenderCounts(overall)?.Female || 0}
                </span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Male Count</span>
                <span className="stat-value">
                  {getGenderCounts(overall)?.Male || 0}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Chart Visualization Grid */}
        <div className="analysis-section">
          <h4>Analysis Visualizations - {analysisViewBy === 'metric' ? 'By Metric' : 'By Prompt Group'}</h4>
          
          {promptGroupKeys.length === 0 || orderedMetrics.length === 0 ? (
            <p>No analysis data available.</p>
          ) : (
            <div className="charts-grid">
              {analysisViewBy === 'metric' ? (
                // View by Metric: rows = metrics, columns = prompt groups
                <>
                  {/* Header row with prompt groups */}
                  <div className="charts-header">
                    <div className="chart-header-cell metric-header">Metric</div>
                    {promptGroupKeys.map(promptKey => (
                      <div key={promptKey} className="chart-header-cell prompt-header">
                        <span className="prompt-header-text">
                          {getPromptVariationText(promptKey)}
                        </span>
                      </div>
                    ))}
                  </div>
                  
                  {/* Chart rows grouped by section */}
                  {['demographics', 'appearance', 'role_and_agency', 'composition'].map(section => {
                    const sectionMetrics = orderedMetrics.filter(metric => {
                      const schemaInfo = getSchemaInfo(metric);
                      if (section === 'demographics') return analysisSchema?.people?.[0]?.demographics?.[metric];
                      if (section === 'appearance') return analysisSchema?.people?.[0]?.appearance?.[metric];
                      if (section === 'role_and_agency') return analysisSchema?.people?.[0]?.role_and_agency?.[metric];
                      if (section === 'composition') return analysisSchema?.composition?.[metric];
                      return false;
                    });
                    
                    if (sectionMetrics.length === 0) return null;
                    
                    return (
                      <div key={section} className="charts-section">
                        <h5 className="section-title">{section.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</h5>
                        {sectionMetrics.map(metric => (
                          <div key={metric} className="charts-row">
                            <div className="chart-row-label">{metric}</div>
                            {promptGroupKeys.map(promptKey => {
                              const metricData = getMetricData(prompt_groups[promptKey], metric);
                              const schemaInfo = getSchemaInfo(metric);
                              return (
                                <div key={`${metric}-${promptKey}`} className="chart-cell">
                                  {generateChart(metric, metricData, schemaInfo, promptKey)}
                                </div>
                              );
                            })}
                          </div>
                        ))}
                      </div>
                    );
                  })}
                </>
              ) : (
                // View by Prompt: rows = prompt groups, columns = metrics
                <>
                  {/* Header row with metrics grouped by section */}
                  <div className="charts-header">
                    <div className="chart-header-cell prompt-header">Prompt Group</div>
                    {['demographics', 'appearance', 'role_and_agency', 'composition'].map(section => {
                      const sectionMetrics = orderedMetrics.filter(metric => {
                        if (section === 'demographics') return analysisSchema?.people?.[0]?.demographics?.[metric];
                        if (section === 'appearance') return analysisSchema?.people?.[0]?.appearance?.[metric];
                        if (section === 'role_and_agency') return analysisSchema?.people?.[0]?.role_and_agency?.[metric];
                        if (section === 'composition') return analysisSchema?.composition?.[metric];
                        return false;
                      });
                      
                      return sectionMetrics.map(metric => (
                        <div key={metric} className="chart-header-cell metric-header">
                          <span className="section-label">{section}</span>
                          <span className="metric-label">{metric}</span>
                        </div>
                      ));
                    })}
                  </div>
                  
                  {/* Chart rows for each prompt group */}
                  {promptGroupKeys.map(promptKey => (
                    <div key={promptKey} className="charts-row">
                      <div className="chart-row-label">
                        {getPromptVariationText(promptKey)}
                      </div>
                      {['demographics', 'appearance', 'role_and_agency', 'composition'].map(section => {
                        const sectionMetrics = orderedMetrics.filter(metric => {
                          if (section === 'demographics') return analysisSchema?.people?.[0]?.demographics?.[metric];
                          if (section === 'appearance') return analysisSchema?.people?.[0]?.appearance?.[metric];
                          if (section === 'role_and_agency') return analysisSchema?.people?.[0]?.role_and_agency?.[metric];
                          if (section === 'composition') return analysisSchema?.composition?.[metric];
                          return false;
                        });
                        
                        return sectionMetrics.map(metric => {
                          const metricData = getMetricData(prompt_groups[promptKey], metric);
                          const schemaInfo = getSchemaInfo(metric);
                          return (
                            <div key={`${promptKey}-${metric}`} className="chart-cell">
                              {generateChart(metric, metricData, schemaInfo, promptKey)}
                            </div>
                          );
                        });
                      })}
                    </div>
                  ))}
                </>
              )}
            </div>
          )}
        </div>

        {/* Prompt Group Cards - Original Visualization */}
        <div className="analysis-section">
          <h4>Prompt Group Analysis Cards</h4>
          
          {Object.keys(prompt_groups).length === 0 ? (
            <p>No prompt group data available.</p>
          ) : (
            <div className="prompt-groups">
              {Object.entries(prompt_groups).map(([promptKey, promptData]) => {
                const genderCounts = getGenderCounts(promptData);
                const variationText = getPromptVariationText(promptKey);
                
                return (
                  <div key={promptKey} className="prompt-group-card">
                    <h5 className="prompt-group-title">{variationText}</h5>
                    
                    <div className="prompt-stats">
                      <div className="stat-item">
                        <span className="stat-label">Videos:</span>
                        <span className="stat-value">{promptData.metadata?.successfully_loaded || 0}</span>
                      </div>
                      
                      <div className="stat-item">
                        <span className="stat-label">People:</span>
                        <span className="stat-value">{promptData.aggregated_data?.people?.total_people || 0}</span>
                      </div>
                      
                      <div className="stat-item">
                        <span className="stat-label">Female:</span>
                        <span className="stat-value">{genderCounts.Female || 0}</span>
                      </div>
                      
                      <div className="stat-item">
                        <span className="stat-label">Male:</span>
                        <span className="stat-value">{genderCounts.Male || 0}</span>
                      </div>
                      
                      {genderCounts.Ambiguous && (
                        <div className="stat-item">
                          <span className="stat-label">Ambiguous:</span>
                          <span className="stat-value">{genderCounts.Ambiguous}</span>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AnalysisDashboard;
