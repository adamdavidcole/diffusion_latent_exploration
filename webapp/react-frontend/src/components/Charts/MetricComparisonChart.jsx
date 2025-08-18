import React, { useState } from 'react';
import { Bar } from 'react-chartjs-2';
import '../../utils/chartSetup';
import { getVariationTextFromPromptKey } from '../../utils/variationText';

const MetricComparisonChart = ({
    data,
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
            getVariationTextFromPromptKey(key, currentExperiment) :
            abbreviatedLabel;

        return showFullVariationText ? variationText : abbreviatedLabel;
    });

    const chartData = {
        labels: chartLabels,
        datasets: [
            {
                label: title,
                data: values,
                backgroundColor: chartColors.map(color => `${color}80`),
                borderColor: chartColors,
                borderWidth: 2,
                borderRadius: 4,
                borderSkipped: false,
            },
        ],
    };

    const options = {
        responsive: true,
        width: size,
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
                    title: function (context) {
                        const index = context[0].dataIndex;
                        const promptKey = promptGroups[index];
                        const variationText = currentExperiment ?
                            getVariationTextFromPromptKey(promptKey, currentExperiment) :
                            promptKey.replace('prompt_', 'Prompt ');
                        return variationText;
                    },
                    label: function (context) {
                        const value = typeof context.parsed.y === 'number' ?
                            context.parsed.y.toFixed(4) : context.parsed.y;
                        return `${title}: ${value}`;
                    }
                }
            },
        },
        scales: {
            y: {
                beginAtZero: beginAtZero,
                title: {
                    display: !!yLabel,
                    text: yLabel,
                    color: '#b0b0b0',
                },
                ticks: {
                    color: '#b0b0b0',
                    callback: function (value) {
                        return typeof value === 'number' ? value.toFixed(2) : value;
                    }
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
                    maxRotation: showFullVariationText ? 45 : 0,
                    minRotation: showFullVariationText ? 45 : 0,
                },
                grid: {
                    color: 'rgba(176, 176, 176, 0.1)',
                },
            },
        },
    };

    return (
        <div style={{ position: 'relative' }}>
            {/* Main Chart */}
            <Bar data={chartData} options={options} />
        </div>
    );
};

export default MetricComparisonChart;
