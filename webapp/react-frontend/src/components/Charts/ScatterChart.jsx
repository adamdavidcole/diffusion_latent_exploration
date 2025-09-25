import React from 'react';
import { Scatter } from 'react-chartjs-2';
import '../../utils/chartSetup';
import { getVariationTextFromPromptKey } from '../../utils/variationText';

const ScatterChart = ({
    data,
    title,
    size = 500,
    xLabel = '',
    yLabel = '',
    colors = null,
    currentExperiment = null,
    showFullVariationText = false,
    beginAtZero = false,
    xMin = null,
    xMax = null,
    yMin = null,
    yMax = null
}) => {

    if (!data || Object.keys(data).length === 0) {
        return (
            <div style={{ width: size, height: size, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span style={{ color: '#b0b0b0' }}>No data available</span>
            </div>
        );
    }

    // Default color palette
    const defaultColors = [
        '#4A90E2', '#7ED321', '#F5A623', '#D0021B', '#9013FE', '#50E3C2',
        '#B8E986', '#4BD5EE', '#9AA0A6', '#F8E71C'
    ];

    const promptGroups = Object.keys(data).sort();
    const chartColors = colors || defaultColors.slice(0, promptGroups.length);

    // Create datasets for each prompt group
    const datasets = promptGroups.map((promptGroup, index) => {
        const groupData = data[promptGroup] || [];

        // Get label for the group
        const abbreviatedLabel = promptGroup.replace('prompt_', 'P');
        const variationText = currentExperiment ?
            getVariationTextFromPromptKey(promptGroup, currentExperiment, showFullVariationText) :
            abbreviatedLabel;

        const label = variationText;

        return {
            label: label,
            data: groupData,
            backgroundColor: chartColors[index] + '80', // Add transparency
            borderColor: chartColors[index],
            borderWidth: 1,
            pointRadius: 4,
            pointHoverRadius: 6,
        };
    });

    const chartData = {
        datasets: datasets
    };

    // Calculate height dynamically based on width, maintaining aspect ratio
    const calculateHeight = (width) => {
        const baseHeight = 400; // Base height for responsiveness
        const aspectRatio = 1.2; // Width to height ratio
        return Math.max(baseHeight, width / aspectRatio);
    };

    const chartHeight = calculateHeight(size);

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            title: {
                display: true,
                text: title,
                font: {
                    size: 14,
                    weight: 'bold'
                },
                padding: {
                    bottom: 20
                }
            },
            legend: {
                display: true,
                position: 'top',
                labels: {
                    padding: 20,
                    font: {
                        size: 11,
                        color: 'rgba(255, 255, 255, 1)'
                    },
                    generateLabels: (chart) => {
                        return chart.data.datasets.map((dataset, i) => ({
                            text: dataset.label,
                            fillStyle: dataset.backgroundColor,
                            strokeStyle: dataset.borderColor,
                            lineWidth: dataset.borderWidth,
                            hidden: !chart.isDatasetVisible(i),
                            index: i
                        }));
                    }
                }
            },
            tooltip: {
                callbacks: {
                    title: function (context) {
                        const datasetIndex = context[0].datasetIndex;
                        const promptGroup = promptGroups[datasetIndex];

                        // Always show full variation text in tooltip, regardless of showFullVariationText setting
                        const fullVariationText = currentExperiment ?
                            getVariationTextFromPromptKey(promptGroup, currentExperiment, true) :
                            promptGroup.replace('prompt_', 'P');

                        return fullVariationText;
                    },
                    label: function (context) {
                        return `${xLabel}: ${context.parsed.x?.toFixed(3) || 'N/A'}, ${yLabel}: ${context.parsed.y?.toFixed(3) || 'N/A'}`;
                    }
                }
            }
        },
        scales: {
            x: {
                type: 'linear',
                position: 'bottom',
                title: {
                    display: !!xLabel,
                    text: xLabel,
                    font: {
                        size: 12,
                        weight: 'bold'
                    }
                },
                grid: {
                    display: true,
                    color: 'rgba(0,0,0,0.1)'
                },
                beginAtZero: beginAtZero,
                min: xMin,
                max: xMax
            },
            y: {
                title: {
                    display: !!yLabel,
                    text: yLabel,
                    font: {
                        size: 12,
                        weight: 'bold'
                    }
                },
                grid: {
                    display: true,
                    color: 'rgba(0,0,0,0.1)'
                },
                beginAtZero: beginAtZero,
                min: yMin,
                max: yMax
            }
        },
        interaction: {
            intersect: false,
            mode: 'point'
        }
    };

    return (
        <div style={{
            width: '100%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center'
        }}>
            <div style={{
                width: `${size}px`,
                height: `${chartHeight}px`,
                position: 'relative'
            }}>
                <Scatter data={chartData} options={options} />
            </div>
        </div>
    );
};

export default ScatterChart;
