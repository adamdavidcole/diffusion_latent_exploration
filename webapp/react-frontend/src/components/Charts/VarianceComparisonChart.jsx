import React from 'react';
import { Bar } from 'react-chartjs-2';
import '../../utils/chartSetup';
import { getPromptGroupColors } from '../../utils/chartColors';
import { getVariationTextFromPromptKey } from '../../utils/variationText';

const VarianceComparisonChart = ({ 
  overallVarianceData,
  varianceAcrossVideosData,
  varianceAcrossStepsData,
  title, 
  size = 250, 
  yLabel = 'Variance',
  currentExperiment,
  showFullVariationText = false,
  beginAtZero = true
}) => {
  if (!overallVarianceData || !varianceAcrossVideosData || !varianceAcrossStepsData) {
    return (
      <div style={{ width: size, height: size, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <span style={{ color: '#b0b0b0' }}>No data available</span>
      </div>
    );
  }

  const promptGroups = Object.keys(overallVarianceData);
  const colors = getPromptGroupColors(promptGroups);

  // Prepare labels
  const labels = promptGroups.map(promptGroup => {
    return currentExperiment ?
      getVariationTextFromPromptKey(promptGroup, currentExperiment, showFullVariationText) :
      promptGroup.replace('prompt_', 'P');
  });

  // Extract data for each variance type
  const overallValues = promptGroups.map(group => overallVarianceData[group] || 0);
  const acrossVideosValues = promptGroups.map(group => varianceAcrossVideosData[group] || 0);
  const acrossStepsValues = promptGroups.map(group => varianceAcrossStepsData[group] || 0);

  const chartData = {
    labels,
    datasets: [
      {
        label: 'Overall Variance',
        data: overallValues,
        backgroundColor: 'rgba(75, 192, 192, 0.6)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1,
      },
      {
        label: 'Across Videos',
        data: acrossVideosValues,
        backgroundColor: 'rgba(255, 206, 86, 0.6)',
        borderColor: 'rgba(255, 206, 86, 1)',
        borderWidth: 1,
      },
      {
        label: 'Across Steps',
        data: acrossStepsValues,
        backgroundColor: 'rgba(153, 102, 255, 0.6)',
        borderColor: 'rgba(153, 102, 255, 1)',
        borderWidth: 1,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          color: '#b0b0b0',
          font: {
            size: 10
          }
        }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderWidth: 1,
      },
    },
    scales: {
      y: {
        beginAtZero,
        title: {
          display: !!yLabel,
          text: yLabel,
          color: '#b0b0b0',
        },
        ticks: {
          color: '#b0b0b0',
        },
        grid: {
          color: 'rgba(176, 176, 176, 0.1)',
        },
      },
      x: {
        title: {
          display: true,
          text: 'Prompt Groups',
          color: '#b0b0b0',
        },
        ticks: {
          color: '#b0b0b0',
          maxTicksLimit: 50, // Ensure all labels are shown
          maxRotation: 45,
          minRotation: 45,
          display: true, // Force all ticks to display
        },
        grid: {
          color: 'rgba(176, 176, 176, 0.1)',
        },
      },
    },
  };

  return (
    <div style={{ width: size, height: size }}>
      <Bar data={chartData} options={options} />
    </div>
  );
};

export default VarianceComparisonChart;
