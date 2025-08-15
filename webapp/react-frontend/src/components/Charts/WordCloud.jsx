import React, { useRef, useEffect } from 'react';
import { Chart } from 'react-chartjs-2';
import { WordCloudController, WordElement } from 'chartjs-chart-wordcloud';
import { ChartJS } from '../../utils/chartSetup';

const WordCloud = ({ data, title, size = 250 }) => {
  const chartRef = useRef(null);

  console.log('WordCloud received data:', data, 'title:', title);

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
    // Handle nested data structure from VLM analysis
    let wordData = null;

    if (data?.data?.top_words) {
      wordData = data.data.top_words;
    } else if (data?.data?.word_frequency) {
      wordData = data.data.word_frequency;
    } else if (data?.top_words) {
      wordData = data.top_words;
    } else if (data?.word_frequency) {
      wordData = data.word_frequency;
    }

    if (!wordData || typeof wordData !== 'object') {
      console.warn('No valid word data found:', data);
      return { labels: [], data: [] };
    }

    // Convert to the format expected by chartjs-chart-wordcloud
    const entries = Object.entries(wordData);
    const labels = entries.map(([text, value]) => text);
    const values = entries.map(([text, value]) => typeof value === 'number' ? value : 1);

    return { labels, data: values };
  };

  const wordCloudData = getWordCloudData();
  console.log('WordCloud processed wordData:', wordCloudData);

  if (wordCloudData.labels.length === 0) {
    return (
      <div className="chart-container" style={{ width: size, height: size }}>
        <div className="chart-title">{title}</div>
        <div className="no-data">No word data available</div>
      </div>
    );
  }

  const chartData = {
    labels: wordCloudData.labels,
    datasets: [
      {
        label: title,
        data: wordCloudData.data,
        color: '#4A90E2',
        family: 'Inter, sans-serif',
        size: [10, 40],
        padding: 2,
      },
    ],
  };

  console.log('WordCloud chartData:', chartData);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
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
          title: (context) => {
            console.log('WordCloud tooltip title context:', context);
            const item = context[0];
            // For the labels/data format, the label should be the word
            return item?.label || 'Unknown';
          },
          label: (context) => {
            console.log('WordCloud tooltip label context:', context);
            const item = context[0];
            // The actual data value should be in the dataset at the dataIndex
            const value = item?.dataset?.data?.[item?.dataIndex] || item?.parsed?.y || item?.formattedValue || 'Unknown';
            return `Count: ${value}`;
          },
        },
      },
    },
    layout: {
      padding: 10,
    },
  };

  return (
    <div className="chart-container" style={{ width: size, height: size, padding: '5px', boxSizing: 'border-box' }}>
      <div className="chart-title">{title}</div>
      <div style={{ width: '100%', height: size - 50, overflow: 'hidden' }}>
        <Chart
          ref={chartRef}
          type="wordCloud"
          data={chartData}
          options={options}
          width={size - 20}
          height={size - 60} // Account for title and padding
        />
      </div>
    </div>
  );
};

export default WordCloud;
