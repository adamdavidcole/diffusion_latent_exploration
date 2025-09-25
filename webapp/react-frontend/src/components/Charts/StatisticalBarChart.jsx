import React from 'react';
import { Bar } from 'react-chartjs-2';
import '../../utils/chartSetup';
import { getVariationTextFromPromptKey } from '../../utils/variationText';

const StatisticalBarChart = ({
    data,
    statisticalData,
    title,
    size = 500,
    yLabel = '',
    colors = null,
    currentExperiment = null,
    showFullVariationText = false,
    beginAtZero = false
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
    const values = promptGroups.map(key => data[key]);
    const chartColors = colors || defaultColors.slice(0, promptGroups.length);

    // Create mappings for labels and variation text
    const chartLabels = promptGroups.map(key => {
        const abbreviatedLabel = key.replace('prompt_', 'P');
        const variationText = currentExperiment ?
            getVariationTextFromPromptKey(key, currentExperiment, showFullVariationText) :
            abbreviatedLabel;

        return variationText;
    });

    // Prepare statistical ranges if available
    const hasStatisticalData = statisticalData && Object.keys(statisticalData).length > 0;
    
    const datasets = [];

    if (hasStatisticalData) {
        // Create range bars (from min to max)
        const rangeData = promptGroups.map(key => {
            const stats = statisticalData[key];
            if (stats && stats.min != null && stats.max != null) {
                return {
                    min: stats.min,
                    max: stats.max,
                    range: stats.max - stats.min
                };
            }
            return { min: 0, max: 0, range: 0 };
        });

        // Background range bars (showing full min-max range)
        datasets.push({
            label: 'Range (Min-Max)',
            data: rangeData.map(r => r.range),
            backgroundColor: chartColors.map(color => color + '20'), // Very transparent
            borderColor: chartColors.map(color => color + '40'),
            borderWidth: 1,
            yAxisID: 'y',
            order: 2, // Render behind main bars
        });

        // Standard deviation range bars
        const stdDevData = promptGroups.map(key => {
            const stats = statisticalData[key];
            if (stats && stats.mean != null && stats.std != null) {
                const stdRange = stats.std * 2; // ±1 std dev = 2 std devs total
                return stdRange;
            }
            return 0;
        });

        datasets.push({
            label: '±1 Std Dev',
            data: stdDevData,
            backgroundColor: chartColors.map(color => color + '40'), // Semi-transparent
            borderColor: chartColors.map(color => color + '60'),
            borderWidth: 1,
            yAxisID: 'y',
            order: 1, // Render behind main bars but in front of range
        });
    }

    // Main data bars (means)
    datasets.push({
        label: title,
        data: values,
        backgroundColor: chartColors,
        borderColor: chartColors.map(color => color.replace('1)', '0.8)')),
        borderWidth: 2,
        yAxisID: 'y',
        order: 0, // Render on top
    });

    const chartData = {
        labels: chartLabels,
        datasets: datasets
    };

    // Calculate height dynamically based on width, maintaining aspect ratio
    const calculateHeight = (width) => {
        const baseHeight = 400;
        const aspectRatio = 1.2;
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
                display: hasStatisticalData,
                position: 'top',
                labels: {
                    padding: 20,
                    font: {
                        size: 11
                    }
                }
            },
            tooltip: {
                callbacks: {
                    title: function(context) {
                        const index = context[0].dataIndex;
                        const promptGroup = promptGroups[index];
                        
                        // Always show full variation text in tooltip
                        const fullVariationText = currentExperiment ?
                            getVariationTextFromPromptKey(promptGroup, currentExperiment, true) :
                            promptGroup.replace('prompt_', 'P');
                        
                        return fullVariationText;
                    },
                    label: function(context) {
                        const index = context.dataIndex;
                        const promptGroup = promptGroups[index];
                        const stats = statisticalData?.[promptGroup];
                        
                        if (context.datasetIndex === 0 && hasStatisticalData) {
                            // Main data bar - show comprehensive stats
                            const lines = [
                                `Mean: ${context.parsed.y?.toFixed(3) || 'N/A'}`
                            ];
                            
                            if (stats) {
                                if (stats.std != null) lines.push(`Std Dev: ±${stats.std.toFixed(3)}`);
                                if (stats.min != null) lines.push(`Min: ${stats.min.toFixed(3)}`);
                                if (stats.max != null) lines.push(`Max: ${stats.max.toFixed(3)}`);
                                if (stats.median != null) lines.push(`Median: ${stats.median.toFixed(3)}`);
                            }
                            
                            return lines;
                        } else {
                            return `${context.dataset.label}: ${context.parsed.y?.toFixed(3) || 'N/A'}`;
                        }
                    }
                }
            }
        },
        scales: {
            x: {
                title: {
                    display: false
                },
                grid: {
                    display: false
                }
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
                beginAtZero: beginAtZero
            }
        },
        interaction: {
            intersect: false,
            mode: 'index'
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
                <Bar data={chartData} options={options} />
            </div>
        </div>
    );
};

export default StatisticalBarChart;
