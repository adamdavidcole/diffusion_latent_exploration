import React from 'react';
import { Bar } from 'react-chartjs-2';
import '../../utils/chartSetup';

const BarChart = ({ data, title, size = 250 }) => {
  // Handle range data (mean, min, max, etc.)
  if (data.mean !== undefined) {
    const chartData = {
      labels: ['Min', 'Mean', 'Max'],
      datasets: [
        {
          label: title,
          data: [data.min || 0, data.mean || 0, data.max || 0],
          backgroundColor: ['#D0021B', '#4A90E2', '#7ED321'],
          borderColor: ['#D0021B', '#4A90E2', '#7ED321'],
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
          borderColor: '#4A90E2',
          borderWidth: 1,
        },
      },
      scales: {
        y: {
          beginAtZero: true,
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
      <div className="chart-container" style={{ width: size, height: size }}>
        <div className="chart-title">{title}</div>
        <Bar data={chartData} options={options} />
      </div>
    );
  }

  // Handle count data (if it's formatted as counts)
  if (data.counts) {
    const labels = Object.keys(data.counts);
    const values = Object.values(data.counts);

    const chartData = {
      labels,
      datasets: [
        {
          label: title,
          data: values,
          backgroundColor: '#4A90E2',
          borderColor: '#357abd',
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
          borderColor: '#4A90E2',
          borderWidth: 1,
        },
      },
      scales: {
        y: {
          beginAtZero: true,
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
            maxRotation: 45,
          },
          grid: {
            color: 'rgba(176, 176, 176, 0.1)',
          },
        },
      },
    };

    return (
      <div className="chart-container" style={{ width: size, height: size }}>
        <div className="chart-title">{title}</div>
        <Bar data={chartData} options={options} />
      </div>
    );
  }

  return (
    <div className="chart-container" style={{ width: size, height: size }}>
      <div className="chart-title">{title}</div>
      <div className="no-data">No data available</div>
    </div>
  );
};

export default BarChart;
