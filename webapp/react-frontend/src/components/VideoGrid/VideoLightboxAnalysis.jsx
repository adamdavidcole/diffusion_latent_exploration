import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { getVideoUrl } from '../../services/api';
import ChartRenderer from '../Charts/ChartRenderer';
import {
    transformIndividualToAggregated,
    formatCulturalFlags,
    getVlmAnalysisPath,
    LIGHTBOX_ANALYSIS_CONFIG
} from '../../utils/vlmDataTransform';
import './VideoLightboxAnalysis.css';

const VideoLightboxAnalysis = ({ video }) => {
    const { state } = useApp();
    const [analysisData, setAnalysisData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchAnalysisData = async () => {
            if (!video) return;

            setLoading(true);
            setError(null);

            try {
                const vlmPath = getVlmAnalysisPath(video);
                if (!vlmPath) {
                    throw new Error('Could not determine VLM analysis path for video');
                }

                // Use the same URL pattern as video URLs but for VLM JSON files
                const vlmUrl = getVideoUrl(vlmPath);
                const response = await fetch(vlmUrl);
                if (!response.ok) {
                    throw new Error(`Failed to fetch analysis: ${response.status}`);
                }

                const individualData = await response.json();
                console.log('Fetched individual VLM data:', individualData);

                // Transform to aggregated format
                const transformedData = transformIndividualToAggregated(individualData, state.analysisSchema);
                console.log('Transformed data:', transformedData);

                setAnalysisData({
                    transformed: transformedData,
                    culturalFlags: formatCulturalFlags(individualData.cultural_flags)
                });

                // Debug logging
                console.log('Final analysis data:', {
                    transformed: transformedData,
                    schema: state.analysisSchema,
                    demographicsExists: !!transformedData?.people?.sections?.demographics,
                    appearanceExists: !!transformedData?.people?.sections?.appearance,
                    demographicsFields: Object.keys(transformedData?.people?.sections?.demographics || {}),
                    appearanceFields: Object.keys(transformedData?.people?.sections?.appearance || {})
                });
            } catch (err) {
                console.error('Error fetching VLM analysis:', err);
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchAnalysisData();
    }, [video, state.analysisSchema]);

    if (loading) {
        return (
            <div className="lightbox-analysis">
                <div className="analysis-loading">
                    Loading analysis...
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="lightbox-analysis">
                <div className="analysis-error">
                    Failed to load analysis: {error}
                </div>
            </div>
        );
    }

    if (!analysisData || !analysisData.transformed) {
        return (
            <div className="lightbox-analysis">
                <div className="analysis-error">
                    No analysis data available for this video
                </div>
            </div>
        );
    }

    const { transformed, culturalFlags } = analysisData;

    // Debug logging for render
    console.log('Render debug:', {
        transformedExists: !!transformed,
        demographicsSection: transformed?.people?.sections?.demographics,
        appearanceSection: transformed?.people?.sections?.appearance,
        schema: state.analysisSchema,
        schemaDemographics: state.analysisSchema?.people?.[0]?.demographics,
        schemaAppearance: state.analysisSchema?.people?.[0]?.appearance
    });

    return (
        <div className="lightbox-analysis">
            <h3>Video Analysis</h3>

            {/* Demographics Charts */}
            {transformed.people?.sections?.demographics && (
                <div className="analysis-section">
                    <h4>Demographics</h4>
                    <div className="analysis-charts-grid">
                        {LIGHTBOX_ANALYSIS_CONFIG.demographics.map(field => {
                            const fieldData = transformed.people.sections.demographics[field];
                            // Fix schema access - people is an array with one element
                            const schemaField = state.analysisSchema?.people?.[0]?.demographics?.[field];

                            console.log(`Demographics field ${field}:`, {
                                fieldData,
                                schemaField,
                                hasData: !!fieldData?.data,
                                hasSchema: !!schemaField
                            });

                            if (!fieldData || !fieldData.data || !schemaField) return null;

                            return (
                                <ChartRenderer
                                    key={field}
                                    schemaField={schemaField}
                                    data={fieldData.data}
                                    title={field.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                    size={250}
                                />
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Appearance Charts */}
            {transformed.people?.sections?.appearance && (
                <div className="analysis-section">
                    <h4>Appearance</h4>
                    <div className="analysis-charts-grid">
                        {LIGHTBOX_ANALYSIS_CONFIG.appearance.map(field => {
                            const fieldData = transformed.people.sections.appearance[field];
                            // Fix schema access - people is an array with one element
                            const schemaField = state.analysisSchema?.people?.[0]?.appearance?.[field];

                            console.log(`Appearance field ${field}:`, {
                                fieldData,
                                schemaField,
                                hasData: !!fieldData?.data,
                                hasSchema: !!schemaField
                            });

                            if (!fieldData || !fieldData.data || !schemaField) return null;

                            return (
                                <ChartRenderer
                                    key={field}
                                    schemaField={schemaField}
                                    data={fieldData.data}
                                    title={field.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                    size={250}
                                />
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Cultural Flags */}
            {culturalFlags && culturalFlags.length > 0 && (
                <div className="analysis-section">
                    <h4>Cultural Flags</h4>
                    <div className="cultural-flags">
                        {culturalFlags.map((section, idx) => (
                            <div key={idx} className="cultural-flag-section">
                                <h5>{section.title}</h5>
                                {section.fields.map((field, fieldIdx) => (
                                    <div key={fieldIdx} className="cultural-flag-field">
                                        <div className="cultural-flag-label">{field.label}:</div>
                                        <div className="cultural-flag-value">{field.value}</div>
                                    </div>
                                ))}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default VideoLightboxAnalysis;
