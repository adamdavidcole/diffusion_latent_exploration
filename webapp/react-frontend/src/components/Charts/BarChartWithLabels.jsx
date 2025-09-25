import React from 'react';
import { Bar } from 'react-chartjs-2';
import '../../utils/chartSetup';
import { getPromptGroupColors } from '../../utils/chartColors';
import { getVariationTextFromPromptKey } from '../../utils/variationText';

const BarChartWithLabels = ({ 
  data, 
  labelData,
  title, 
  size = 250, 
  yLabel = '',
  currentExperiment,
  showFullVariationText = false,
  beginAtZero = true
}) => {
  if (!data || typeof data !== 'object' || Object.keys(data).length === 0) {
    return (
      <div style={{ width: size, height: size, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <span style={{ color: '#b0b0b0' }}>No data available</span>
      </div>
    );
  }

  const promptGroups = Object.keys(data);
  const colors = getPromptGroupColors(promptGroups);

  // Prepare labels
  const labels = promptGroups.map(promptGroup => {
    return currentExperiment ?
      getVariationTextFromPromptKey(promptGroup, currentExperiment, showFullVariationText) :
      promptGroup.replace('prompt_', 'P');
  });

  // Extract data values
  const values = promptGroups.map(group => data[group] || 0);
  const stepValues = labelData ? promptGroups.map(group => labelData[group]) : null;

  const chartData = {
    labels,
    datasets: [
      {
        label: title,
        data: values,
        backgroundColor: colors.map(color => `${color}80`), // Add transparency
        borderColor: colors,
        borderWidth: 1,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderWidth: 1,
        callbacks: stepValues ? {
          afterBody: function(context) {
            const index = context[0].dataIndex;
            const stepValue = stepValues[index];
            return stepValue !== undefined ? [`Step: ${stepValue}`] : [];
          }
        } : undefined
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
    animation: {
      onComplete: function() {
        // Add text labels on top of bars if we have step data
        // Check if chart and context still exist (prevents errors during fast tab switches)
        try {
          if (stepValues && this.chart && this.chart.ctx && !this.chart.destroyed) {
            const ctx = this.chart.ctx;
            ctx.font = '12px Arial';
            ctx.fillStyle = '#b0b0b0';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';

            this.data.datasets.forEach((dataset, i) => {
              const meta = this.chart.getDatasetMeta(i);
              if (meta && meta.data) {
                meta.data.forEach((bar, index) => {
                  const stepValue = stepValues[index];
                  if (stepValue !== undefined && bar && bar.x !== undefined && bar.y !== undefined) {
                    const data = dataset.data[index];
                    ctx.fillText(`@${stepValue}`, bar.x, bar.y - 5);
                  }
                });
              }
            });
          }
        } catch (error) {
          // Silently ignore animation errors during component unmounting
          console.warn('Chart animation error (likely due to component unmounting):', error.message);
        }
      }
    }
  };

  return (
    <div style={{ width: size, height: size }}>
      <Bar data={chartData} options={options} />
    </div>
  );
};

export default BarChartWithLabels;
