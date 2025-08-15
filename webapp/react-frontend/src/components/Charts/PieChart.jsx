import React from 'react';
import { Pie } from 'react-chartjs-2';
import '../../utils/chartSetup';

const PieChart = ({ data, title, size = 250 }) => {
  // Convert data counts to chart format
  const labels = Object.keys(data.counts || {});
  const values = Object.values(data.counts || {});

  const chartData = {
    labels,
    datasets: [
      {
        data: values,
        backgroundColor: [
          '#4A90E2',
          '#7ED321',
          '#F5A623',
          '#D0021B',
          '#9013FE',
          '#50E3C2',
          '#BD10E0',
          '#B8E986',
          '#4A4A4A',
          '#9B9B9B',
        ],
        borderWidth: 1,
        borderColor: '#2d2d2d',
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    plugins: {
      legend: {
        position: 'bottom',
        labels: {
          color: '#b0b0b0',
          font: {
            size: 11,
          },
          padding: 10,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#ffffff',
        bodyColor: '#ffffff',
        borderColor: '#4A90E2',
        borderWidth: 1,
      },
    },
  };

  if (labels.length === 0) {
    return (
      <div className="chart-container" style={{ width: size, height: size }}>
        <div className="chart-title">{title}</div>
        <div className="no-data">No data available</div>
      </div>
    );
  }

  return (
    <div className="chart-container" style={{ width: size, height: size }}>
      <div className="chart-title">{title}</div>
      <Pie data={chartData} options={options} />
    </div>
  );
};

export default PieChart;
