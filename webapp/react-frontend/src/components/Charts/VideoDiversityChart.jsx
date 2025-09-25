import React from 'react';
import { Bar } from 'react-chartjs-2';
import '../../utils/chartSetup';
import { getPromptGroupColors } from '../../utils/chartColors';
import { getVariationTextFromPromptKey } from '../../utils/variationText';

const VideoDiversityChart = ({ 
  diversityMeanData,
  diversityStdData,
  title, 
  size = 250, 
  yLabel = 'Diversity',
  currentExperiment,
  showFullVariationText = false,
  beginAtZero = true
}) => {
  if (!diversityMeanData || !diversityStdData) {
    return (
      <div style={{ width: size, height: size, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <span style={{ color: '#b0b0b0' }}>No data available</span>
      </div>
    );
  }

  const promptGroups = Object.keys(diversityMeanData);
  const colors = getPromptGroupColors(promptGroups);

  // Prepare labels
  const labels = promptGroups.map(promptGroup => {
    return currentExperiment ?
      getVariationTextFromPromptKey(promptGroup, currentExperiment, showFullVariationText) :
      promptGroup.replace('prompt_', 'P');
  });

  // Extract data for each diversity metric
  const meanValues = promptGroups.map(group => diversityMeanData[group] || 0);
  const stdValues = promptGroups.map(group => diversityStdData[group] || 0);

  const chartData = {
    labels,
    datasets: [
      {
        label: 'Mean Diversity',
        data: meanValues,
        backgroundColor: 'rgba(54, 162, 235, 0.6)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
      },
      {
        label: 'Std Diversity',
        data: stdValues,
        backgroundColor: 'rgba(255, 99, 132, 0.6)',
        borderColor: 'rgba(255, 99, 132, 1)',
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
        callbacks: {
          afterLabel: (context) => {
            if (context.datasetIndex === 0) {
              return 'How much spatial detail varies within each video';
            } else {
              return 'Consistency of spatial variation across videos';
            }
          }
        }
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

export default VideoDiversityChart;