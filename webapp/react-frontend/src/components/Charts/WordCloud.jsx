import React, { useRef, useEffect } from 'react';
import { Chart } from 'react-chartjs-2';
import { WordCloudController, WordElement } from 'chartjs-chart-wordcloud';
import { ChartJS } from '../../utils/chartSetup';

const WordCloud = ({ data, title, size = 250 }) => {
  const chartRef = useRef(null);

  useEffect(() => {
    // Register word cloud components if not already registered
    try {
      if (!ChartJS.registry.getController('wordCloud')) {
        ChartJS.register(WordCloudController, WordElement);
      }
    } catch (error) {
      // If registry doesn't exist or controller check fails, just register
      ChartJS.register(WordCloudController, WordElement);
    }
  }, []);

  // Convert word frequency data to word cloud format
  const getWordCloudData = () => {
    if (!data.top_words && !data.word_frequency) {
      return [];
    }

    const words = data.top_words || data.word_frequency || {};

    return Object.entries(words).map(([text, value]) => ({
      text,
      value: typeof value === 'number' ? value : 1,
    }));
  };

  const wordData = getWordCloudData();

  if (wordData.length === 0) {
    return (
      <div className="chart-container" style={{ width: size, height: size }}>
        <div className="chart-title">{title}</div>
        <div className="no-data">No word data available</div>
      </div>
    );
  }

  const chartData = {
    datasets: [
      {
        label: title,
        data: wordData,
        color: '#4A90E2',
        family: 'Inter, sans-serif',
        size: [10, 40],
        padding: 2,
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
        callbacks: {
          title: (context) => context[0].raw.text,
          label: (context) => `Count: ${context.raw.value}`,
        },
      },
    },
  };

  return (
    <div className="chart-container" style={{ width: size, height: size }}>
      <div className="chart-title">{title}</div>
      <Chart
        ref={chartRef}
        type="wordCloud"
        data={chartData}
        options={options}
        width={size}
        height={size - 40} // Account for title
      />
    </div>
  );
};

export default WordCloud;
