import React, { useState } from 'react';
import { useApp } from '../../context/AppContext';
import ChartRenderer from '../Charts/ChartRenderer';
import { getVariationTextFromPromptKey } from '../../utils/variationText';
import './AnalysisGrid.css';

const AnalysisGrid = () => {
  const { state } = useApp();
  const {
    currentAnalysis,
    analysisSchema,
    analysisViewBy,
    analysisChartSize,
    analysisVisibleSections,
    analysisChartConfig,
    currentExperiment
  } = state;

  // Section tab state
  const [activeSection, setActiveSection] = useState('people');

  // Dashboard controls state (fallback to defaults if not in global state)
  const [localVisibleSections, setLocalVisibleSections] = useState({
    people: true,
    composition: true,
    setting: true,
    cultural_flags: true,
  });

  // Use global state if available, otherwise use local state
  const visibleSections = analysisVisibleSections || localVisibleSections;
  const chartSize = analysisChartSize || 250;

  const sortedSectionNames = ['people', 'composition', 'setting', 'cultural_flags']

  // Available sections based on schema
  // const availableSections = Object.keys(analysisSchema || {}).filter(key =>
  //   sortedSectionNames.includes(key)
  // );

  const availableSections = sortedSectionNames

  if (!currentAnalysis?.vlm_analysis || !analysisSchema) {
    return (
      <div className="analysis-grid">
        <div className="no-analysis">
          <p>Analysis data or schema not available</p>
        </div>
      </div>
    );
  }

  const { prompt_groups, overall } = currentAnalysis.vlm_analysis;

  // Create combined prompt groups including overall as "Combined"
  const allPromptGroups = { ...prompt_groups };
  if (overall) {
    allPromptGroups['combined'] = overall;
  }

  // Get prompt group keys with 'combined' first if it exists
  const promptGroupKeys = Object.keys(allPromptGroups);
  const orderedPromptGroupKeys = overall
    ? ['combined', ...promptGroupKeys.filter(key => key !== 'combined')]
    : promptGroupKeys;

  if (orderedPromptGroupKeys.length === 0) {
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
    console.log(`Getting field data for path: ${path}`, promptData);
    const parts = path.split('.');
    let current = promptData?.aggregated_data;
    for (const part of parts) {
      if (current && current[part]) {
        current = current[part];
      } else {
        console.warn(`Path ${path} failed at part: ${part}`, current);
        return null;
      }
    }
    return current?.data;
  };

  // Helper function to get config for a section from the array structure
  const getSectionConfig = (sectionName) => {
    if (!analysisChartConfig || !Array.isArray(analysisChartConfig)) {
      console.warn('analysisChartConfig is not an array:', analysisChartConfig);
      return null;
    }

    // Find the config object that contains this section
    for (const configObj of analysisChartConfig) {
      if (configObj[sectionName]) {
        console.log(`Found config for section ${sectionName}:`, configObj[sectionName]);
        return configObj[sectionName];
      }
    }
    console.warn(`No config found for section: ${sectionName}`);
    return null;
  };

  // Get all metrics to display based on active section and configuration
  const getMetricsToDisplay = () => {
    const metrics = [];

    // Only process the active section
    if (!analysisSchema[activeSection]) return metrics;

    const section = analysisSchema[activeSection];
    const sectionConfig = getSectionConfig(activeSection);

    if (!sectionConfig) {
      console.warn(`No config found for section: ${activeSection}`);
      return metrics;
    }

    if (Array.isArray(section)) {
      // Handle array sections like 'people'
      // Process in the order defined in the config
      if (Array.isArray(sectionConfig)) {
        sectionConfig.forEach(subsectionConfigObj => {
          Object.entries(subsectionConfigObj).forEach(([subsectionKey, subsectionConfigArray]) => {
            // Find the corresponding subsection in the schema
            const sampleItem = section[0];
            const subsection = sampleItem[subsectionKey];

            if (typeof subsection === 'object' && subsection !== null && Array.isArray(subsectionConfigArray)) {
              // Process fields in the order they appear in the config
              subsectionConfigArray.forEach(fieldConfigObj => {
                Object.entries(fieldConfigObj).forEach(([fieldKey, fieldEnabled]) => {
                  console.log(`Field ${fieldKey} in ${subsectionKey}: ${fieldEnabled}`);

                  const fieldSchema = subsection[fieldKey];
                  if (!fieldSchema) return; // Skip if field doesn't exist in schema

                  if (fieldSchema.type && fieldEnabled) {
                    console.log(`Adding metric: ${subsectionKey}.${fieldKey} (enabled: ${fieldEnabled})`);
                    metrics.push({
                      section: activeSection,
                      subsection: subsectionKey,
                      field: fieldKey,
                      schema: fieldSchema,
                      path: `${activeSection}.sections.${subsectionKey}.${fieldKey}`,
                      title: `${subsectionKey} - ${fieldKey}`.replace(/_/g, ' '),
                    });
                  } else if (fieldKey === 'hair' && typeof fieldSchema === 'object' && fieldEnabled) {
                    // Handle nested objects like hair
                    Object.entries(fieldSchema).forEach(([nestedKey, nestedSchema]) => {
                      // For nested objects, check if there's a nested config
                      let nestedEnabled = false;

                      if (Array.isArray(fieldEnabled)) {
                        for (const nestedConfigObj of fieldEnabled) {
                          if (nestedConfigObj.hasOwnProperty(nestedKey)) {
                            nestedEnabled = nestedConfigObj[nestedKey];
                            break;
                          }
                        }
                      }

                      if (nestedSchema.type && nestedEnabled) {
                        metrics.push({
                          section: activeSection,
                          subsection: subsectionKey,
                          field: `${fieldKey}.${nestedKey}`,
                          schema: nestedSchema,
                          path: `${activeSection}.sections.${subsectionKey}.${fieldKey}.${nestedKey}`,
                          title: `${subsectionKey} - ${fieldKey} ${nestedKey}`.replace(/_/g, ' '),
                        });
                      }
                    });
                  } else {
                    console.log(`Skipping metric: ${subsectionKey}.${fieldKey} (enabled: ${fieldEnabled}, hasType: ${!!fieldSchema.type})`);
                  }
                });
              });
            }
          });
        });
      }
    } else {
      // Handle object sections like 'composition', 'setting', 'cultural_flags'
      if (Array.isArray(sectionConfig)) {
        // Process fields in the order they appear in the config
        sectionConfig.forEach(fieldConfigObj => {
          Object.entries(fieldConfigObj).forEach(([fieldKey, fieldEnabled]) => {
            const fieldSchema = section[fieldKey];
            if (!fieldSchema) return; // Skip if field doesn't exist in schema

            if (fieldSchema.type && fieldEnabled) {
              metrics.push({
                section: activeSection,
                subsection: null,
                field: fieldKey,
                schema: fieldSchema,
                path: `${activeSection}.${fieldKey}`,
                title: `${activeSection} - ${fieldKey}`.replace(/_/g, ' '),
              });
            }
          });
        });
      }
    }

    return metrics;
  };

  const metrics = getMetricsToDisplay();
  console.log(`Found ${metrics.length} metrics for section ${activeSection}:`, metrics);

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
    if (analysisVisibleSections) {
      // Use global state - this would need a global action
      // For now, use local state as fallback
      setLocalVisibleSections(prev => ({
        ...prev,
        [sectionKey]: !prev[sectionKey]
      }));
    } else {
      setLocalVisibleSections(prev => ({
        ...prev,
        [sectionKey]: !prev[sectionKey]
      }));
    }
  };

  return (
    <div className="analysis-grid">
      {/* Dashboard Controls */}
      <div className="analysis-controls">
        <div className="section-tabs">
          {availableSections.map(sectionKey => (
            <button
              key={sectionKey}
              className={`section-tab ${activeSection === sectionKey ? 'active' : ''}`}
              onClick={() => setActiveSection(sectionKey)}
            >
              {sectionKey.replace(/_/g, ' ')}
            </button>
          ))}
        </div>
      </div>

      {/* Analysis Grid */}
      <div className="analysis-content">
        <h3 className="section-header">{activeSection.replace(/_/g, ' ')} Analysis</h3>

        {analysisViewBy === 'prompt' ? (
          // View by Prompt: rows = prompt groups, columns = metrics
          <div className="prompt-view">
            <div className="metrics-grid">
              <div className="grid-header">
                <div className="prompt-group-header">Prompt Group</div>
                {metrics.map(metric => (
                  <div key={metric.path} className="metric-label-header" style={{ width: `${chartSize}px` }}>
                    <span className="section-label">{metric.subsection || activeSection}</span>
                    <span className="metric-label">{metric.field.replace(/_/g, ' ')}</span>
                  </div>
                ))}
              </div>

              {/* Data rows for each prompt group */}
              {orderedPromptGroupKeys.map(promptKey => (
                <div key={promptKey} className="metric-row">
                  <div className="prompt-group-label">
                    {getVariationTextFromPromptKey(promptKey, currentExperiment)}
                  </div>
                  {metrics.map(metric => {
                    const promptData = allPromptGroups[promptKey];
                    const fieldData = getFieldData(metric.path, promptData);

                    return (
                      <div key={`${promptKey}-${metric.path}`} className="chart-cell" style={{ width: `${chartSize}px` }}>
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
        ) : (
          // View by Metric: rows = metrics, columns = prompt groups
          <div className="metric-view">
            {Object.entries(groupedMetrics).map(([sectionKey, subsections]) => (
              <div key={sectionKey}>
                {Object.entries(subsections).map(([subsectionKey, sectionMetrics]) => (
                  <div key={`${sectionKey}-${subsectionKey}`} className="analysis-subsection">
                    {subsectionKey !== 'main' && (
                      <h4 className="subsection-header">{subsectionKey.replace(/_/g, ' ')}</h4>
                    )}

                    <div className="metrics-grid">
                      {/* Header row with prompt group names */}
                      <div className="grid-header">
                        <div className="metric-label-header">Metric</div>
                        {orderedPromptGroupKeys.map(promptKey => (
                          <div key={promptKey} className="prompt-group-header" style={{ minWidth: `${chartSize}px` }}>
                            {getVariationTextFromPromptKey(promptKey, currentExperiment)}
                          </div>
                        ))}
                      </div>

                      {/* Data rows */}
                      {sectionMetrics.map(metric => (
                        <div key={metric.path} className="metric-row">
                          <div className="metric-label">
                            {metric.field.replace(/_/g, ' ')}
                          </div>
                          {orderedPromptGroupKeys.map(promptKey => {
                            const promptData = allPromptGroups[promptKey];
                            const fieldData = getFieldData(metric.path, promptData);

                            return (
                              <div key={`${metric.path}-${promptKey}`} className="chart-cell" style={{ width: `${chartSize}px` }}>
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
        )}
      </div>
    </div>
  );
};

export default AnalysisGrid;
