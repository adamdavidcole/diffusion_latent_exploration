import React, { useState } from 'react';
import { useApp } from '../../context/AppContext';
import ChartRenderer from '../Charts/ChartRenderer';
import './AnalysisGrid.css';

const AnalysisGrid = () => {
  const { state } = useApp();
  const { currentAnalysis, analysisSchema } = state;
  
  // Dashboard controls state
  const [visibleSections, setVisibleSections] = useState({
    people: true,
    composition: true,
    setting: true,
    cultural_flags: true,
  });
  
  const [chartSize, setChartSize] = useState(250);

  if (!currentAnalysis?.vlm_analysis || !analysisSchema) {
    return (
      <div className="analysis-grid">
        <div className="no-analysis">
          <p>Analysis data or schema not available</p>
        </div>
      </div>
    );
  }

  const { prompt_groups } = currentAnalysis.vlm_analysis;
  const promptGroupKeys = Object.keys(prompt_groups);

  if (promptGroupKeys.length === 0) {
    return (
      <div className="analysis-grid">
        <div className="no-analysis">
          <p>No prompt groups found</p>
        </div>
      </div>
    );
  }

  // Helper function to get field schema recursively
  const getFieldSchema = (path, schema) => {
    const parts = path.split('.');
    let current = schema;
    for (const part of parts) {
      if (current[part]) {
        current = current[part];
      } else {
        return null;
      }
    }
    return current;
  };

  // Helper function to get data from prompt group recursively
  const getFieldData = (path, promptData) => {
    const parts = path.split('.');
    let current = promptData?.aggregated_data;
    for (const part of parts) {
      if (current && current[part]) {
        current = current[part];
      } else {
        return null;
      }
    }
    return current?.data;
  };

  // Get all metrics to display based on schema and visible sections
  const getMetricsToDisplay = () => {
    const metrics = [];
    
    Object.entries(visibleSections).forEach(([sectionKey, isVisible]) => {
      if (!isVisible || !analysisSchema[sectionKey]) return;
      
      const section = analysisSchema[sectionKey];
      
      if (Array.isArray(section)) {
        // Handle array sections like 'people'
        const sampleItem = section[0];
        Object.entries(sampleItem).forEach(([subsectionKey, subsection]) => {
          if (typeof subsection === 'object' && subsection !== null) {
            Object.entries(subsection).forEach(([fieldKey, fieldSchema]) => {
              if (fieldSchema.type) {
                metrics.push({
                  section: sectionKey,
                  subsection: subsectionKey,
                  field: fieldKey,
                  schema: fieldSchema,
                  path: `${sectionKey}.sections.${subsectionKey}.${fieldKey}`,
                  title: `${subsectionKey} - ${fieldKey}`.replace(/_/g, ' '),
                });
              } else if (fieldKey === 'hair' && typeof fieldSchema === 'object') {
                // Handle nested objects like hair
                Object.entries(fieldSchema).forEach(([nestedKey, nestedSchema]) => {
                  if (nestedSchema.type) {
                    metrics.push({
                      section: sectionKey,
                      subsection: subsectionKey,
                      field: `${fieldKey}.${nestedKey}`,
                      schema: nestedSchema,
                      path: `${sectionKey}.sections.${subsectionKey}.${fieldKey}.${nestedKey}`,
                      title: `${subsectionKey} - ${fieldKey} ${nestedKey}`.replace(/_/g, ' '),
                    });
                  }
                });
              }
            });
          }
        });
      } else {
        // Handle object sections like 'composition', 'setting', 'cultural_flags'
        Object.entries(section).forEach(([fieldKey, fieldSchema]) => {
          if (fieldSchema.type) {
            metrics.push({
              section: sectionKey,
              subsection: null,
              field: fieldKey,
              schema: fieldSchema,
              path: `${sectionKey}.${fieldKey}`,
              title: `${sectionKey} - ${fieldKey}`.replace(/_/g, ' '),
            });
          }
        });
      }
    });
    
    return metrics;
  };

  const metrics = getMetricsToDisplay();

  // Group metrics by section and subsection for headers
  const getGroupedMetrics = () => {
    const grouped = {};
    metrics.forEach(metric => {
      const sectionKey = metric.section;
      const subsectionKey = metric.subsection || 'main';
      
      if (!grouped[sectionKey]) {
        grouped[sectionKey] = {};
      }
      if (!grouped[sectionKey][subsectionKey]) {
        grouped[sectionKey][subsectionKey] = [];
      }
      grouped[sectionKey][subsectionKey].push(metric);
    });
    return grouped;
  };

  const groupedMetrics = getGroupedMetrics();

  const toggleSection = (sectionKey) => {
    setVisibleSections(prev => ({
      ...prev,
      [sectionKey]: !prev[sectionKey]
    }));
  };

  return (
    <div className="analysis-grid">
      {/* Dashboard Controls */}
      <div className="analysis-controls">
        <div className="section-toggles">
          <h4>Visible Sections:</h4>
          {Object.entries(visibleSections).map(([sectionKey, isVisible]) => (
            <label key={sectionKey} className="section-toggle">
              <input
                type="checkbox"
                checked={isVisible}
                onChange={() => toggleSection(sectionKey)}
              />
              <span>{sectionKey.replace(/_/g, ' ')}</span>
            </label>
          ))}
        </div>
        
        <div className="size-control">
          <label>
            Chart Size: {chartSize}px
            <input
              type="range"
              min="150"
              max="400"
              value={chartSize}
              onChange={(e) => setChartSize(Number(e.target.value))}
              className="size-slider"
            />
          </label>
        </div>
      </div>

      {/* Analysis Grid */}
      <div className="analysis-content">
        {Object.entries(groupedMetrics).map(([sectionKey, subsections]) => (
          <div key={sectionKey} className="analysis-section">
            <h3 className="section-header">{sectionKey.replace(/_/g, ' ')}</h3>
            
            {Object.entries(subsections).map(([subsectionKey, sectionMetrics]) => (
              <div key={`${sectionKey}-${subsectionKey}`} className="analysis-subsection">
                {subsectionKey !== 'main' && (
                  <h4 className="subsection-header">{subsectionKey.replace(/_/g, ' ')}</h4>
                )}
                
                <div className="metrics-grid">
                  {/* Header row with prompt group names */}
                  <div className="grid-header">
                    <div className="metric-label-header">Metric</div>
                    {promptGroupKeys.map(promptKey => (
                      <div key={promptKey} className="prompt-group-header">
                        {promptKey.replace('prompt_', 'Prompt ')}
                      </div>
                    ))}
                  </div>
                  
                  {/* Data rows */}
                  {sectionMetrics.map(metric => (
                    <div key={metric.path} className="metric-row">
                      <div className="metric-label">
                        {metric.field.replace(/_/g, ' ')}
                      </div>
                      {promptGroupKeys.map(promptKey => {
                        const promptData = prompt_groups[promptKey];
                        const fieldData = getFieldData(metric.path, promptData);
                        
                        return (
                          <div key={`${metric.path}-${promptKey}`} className="chart-cell">
                            <ChartRenderer
                              schemaField={metric.schema}
                              data={fieldData}
                              title={metric.field.replace(/_/g, ' ')}
                              size={chartSize}
                            />
                          </div>
                        );
                      })}
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

export default AnalysisGrid;
