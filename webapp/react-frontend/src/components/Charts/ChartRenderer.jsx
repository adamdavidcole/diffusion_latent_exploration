import React from 'react';
import PieChart from './PieChart';
import BarChart from './BarChart';
import WordCloud from './WordCloud';
import './ChartRenderer.css';

const ChartRenderer = ({
  schemaField,
  data,
  title,
  size = 250
}) => {
  if (!data || !schemaField) {
    return (
      <div className="chart-container" style={{ width: size, height: size }}>
        <div className="chart-title">{title}</div>
        <div className="no-data">No data available</div>
      </div>
    );
  }

  // Helper function to extend options data with missing options set to 0
  const extendOptionsData = (data, schemaField) => {
    if (schemaField.type !== 'options' || !schemaField.options || !data.counts) {
      return data;
    }

    // Create new objects in the order defined by the schema
    const orderedCounts = {};
    const orderedPercentages = {};

    // Process options in schema order, adding existing values or 0 for missing ones
    schemaField.options.forEach(option => {
      orderedCounts[option] = data.counts[option] || 0;
      orderedPercentages[option] = data.percentages?.[option] || 0;
    });

    console.log(`Extended options data for ${title}:`, {
      original: data.counts,
      ordered: orderedCounts,
      schemaOptions: schemaField.options
    });

    return {
      ...data,
      counts: orderedCounts,
      percentages: orderedPercentages
    };
  };

  // Determine chart type based on schema field type
  const getChartComponent = () => {
    switch (schemaField.type) {
      case 'options':
        // Extend data with missing options set to 0 for consistency
        const extendedData = extendOptionsData(data, schemaField);
        return <PieChart data={extendedData} title={title} size={size} />;

      case 'range':
        // Use bar chart for numerical ranges
        return <BarChart data={data} title={title} size={size} />;

      case 'open':
        // Use word cloud for open text responses
        return <WordCloud data={data} title={title} size={size} />;

      default:
        return (
          <div className="chart-container" style={{ width: size, height: size }}>
            <div className="chart-title">{title}</div>
            <div className="no-data">Unknown data type: {schemaField.type}</div>
          </div>
        );
    }
  };

  return getChartComponent();
};

export default ChartRenderer;
