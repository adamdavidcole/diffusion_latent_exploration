import React from 'react';
import { Line } from 'react-chartjs-2';
import '../../utils/chartSetup';
import { getPromptGroupColors } from '../../utils/chartColors';
import { getVariationTextFromPromptKey } from '../../utils/variationText';

const LineChart = ({ 
  data, 
  title, 
  size = 250, 
  yLabel = '', 
  xLabel = 'Step',
  color = '#4A90E2',
  currentExperiment,
  showFullVariationText = false,
  beginAtZero = true
}) => {
  // Handle object data with multiple prompt groups (like { prompt_1: [array], prompt_2: [array] })
  if (data && typeof data === 'object' && !Array.isArray(data)) {
    const promptGroups = Object.keys(data);
    const colors = getPromptGroupColors(promptGroups);
    
    // Find the maximum array length to create consistent x-axis labels
    const maxLength = Math.max(...promptGroups.map(group => 
      Array.isArray(data[group]) ? data[group].length : 0
    ));
    
    const labels = Array.from({ length: maxLength }, (_, i) => `${i + 1}`);
    
    const datasets = promptGroups.map((promptGroup, index) => {
      const groupData = data[promptGroup];
      if (!Array.isArray(groupData)) return null;
      
      const label = showFullVariationText && currentExperiment ?
        getVariationTextFromPromptKey(promptGroup, currentExperiment) :
        promptGroup.replace('prompt_', 'P');
      
      return {
        label,
        data: groupData,
        borderColor: colors[index],
        backgroundColor: `${colors[index]}20`,
        borderWidth: 2,
        fill: false,
        tension: 0.4,
        pointRadius: 3,
        pointHoverRadius: 5,
        pointBackgroundColor: colors[index],
        pointBorderColor: '#ffffff',
        pointBorderWidth: 2,
      };
    }).filter(Boolean);

    const chartData = {
      labels,
      datasets
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
            display: !!xLabel,
            text: xLabel,
            color: '#b0b0b0',
          },
          ticks: {
            color: '#b0b0b0',
          },
          grid: {
            color: 'rgba(176, 176, 176, 0.1)',
          },
        },
      },
    };

    return (
      <div style={{ width: size, height: size }}>
        <Line data={chartData} options={options} />
      </div>
    );
  }
  // Handle array data (time series)
  if (Array.isArray(data)) {
    const chartData = {
      labels: data.map((_, index) => `${index + 1}`),
      datasets: [
        {
          label: title,
          data: data,
          borderColor: color,
          backgroundColor: `${color}20`,
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointRadius: 3,
          pointHoverRadius: 5,
          pointBackgroundColor: color,
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
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
          borderColor: color,
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
            display: !!xLabel,
            text: xLabel,
            color: '#b0b0b0',
          },
          ticks: {
            color: '#b0b0b0',
          },
          grid: {
            color: 'rgba(176, 176, 176, 0.1)',
          },
        },
      },
    };

    return (
      <div style={{ width: size, height: size }}>
        <Line data={chartData} options={options} />
      </div>
    );
  }

  // Handle single value data
  if (typeof data === 'number') {
    const chartData = {
      labels: [title],
      datasets: [
        {
          label: title,
          data: [data],
          backgroundColor: color,
          borderColor: color,
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
          borderColor: color,
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
          ticks: {
            color: '#b0b0b0',
          },
          grid: {
            color: 'rgba(176, 176, 176, 0.1)',
          },
        },
      },
    };

    return (
      <div style={{ width: size, height: size }}>
        <Line data={chartData} options={options} />
      </div>
    );
  }

  // Handle object data with mean/min/max
  if (data && typeof data === 'object' && data.mean !== undefined) {
    const chartData = {
      labels: ['Min', 'Mean', 'Max'],
      datasets: [
        {
          label: title,
          data: [data.min || 0, data.mean || 0, data.max || 0],
          borderColor: color,
          backgroundColor: `${color}50`,
          borderWidth: 2,
          fill: false,
          tension: 0.4,
          pointRadius: 5,
          pointHoverRadius: 7,
          pointBackgroundColor: color,
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
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
          borderColor: color,
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
          ticks: {
            color: '#b0b0b0',
          },
          grid: {
            color: 'rgba(176, 176, 176, 0.1)',
          },
        },
      },
    };

    return (
      <div style={{ width: size, height: size }}>
        <Line data={chartData} options={options} />
      </div>
    );
  }

  return (
    <div style={{ width: size, height: size, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <span style={{ color: '#b0b0b0' }}>No data available</span>
    </div>
  );
};

export default LineChart;
