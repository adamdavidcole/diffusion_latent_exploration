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

  // Determine chart type based on schema field type
  const getChartComponent = () => {
    switch (schemaField.type) {
      case 'options':
        // Use pie chart for categorical data
        return <PieChart data={data} title={title} size={size} />;
      
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
