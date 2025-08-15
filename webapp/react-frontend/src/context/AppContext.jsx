import React, { createContext, useContext, useReducer, useCallback, useEffect, useRef } from 'react';
import { api } from '../services/api';

// Helper function to extract available tokens from attention videos
const getAvailableTokensFromAttentionVideos = (attentionVideos) => {
    if (!attentionVideos || !attentionVideos.available || !attentionVideos.prompts) {
        return [];
    }

    const tokenSet = new Set();
    
    // Iterate through all prompts and videos to collect unique tokens
    Object.values(attentionVideos.prompts).forEach(promptData => {
        Object.values(promptData.videos).forEach(videoData => {
            Object.keys(videoData.tokens).forEach(token => {
                tokenSet.add(token);
            });
        });
    });
    
    return Array.from(tokenSet).sort();
};

// Initial state
const initialState = {
    currentExperiment: null,
    experiments: [], // Flat list for backward compatibility
    experimentsTree: null, // Hierarchical tree structure
    videoSize: 200,
    showLabels: true,
    sidebarCollapsed: true,
    loading: false,
    error: null,
    videoDuration: 4.0,
    scrubbingActive: false,
    // Attention video state
    attentionMode: false,
    selectedToken: null,
    availableTokens: [],
    // Analysis state
    currentAnalysis: null,
    analysisLoading: false,
    analysisError: null,
    // Analysis schema
    analysisSchema: null,
    schemaLoading: false,
    schemaError: null,
    // Analysis view preferences
    analysisViewBy: 'metric', // 'metric' or 'prompt'
    analysisChartSize: 250
};

// Action types
const ActionTypes = {
    SET_EXPERIMENTS: 'SET_EXPERIMENTS',
    SET_EXPERIMENTS_TREE: 'SET_EXPERIMENTS_TREE',
    SET_CURRENT_EXPERIMENT: 'SET_CURRENT_EXPERIMENT',
    SET_VIDEO_SIZE: 'SET_VIDEO_SIZE',
    TOGGLE_LABELS: 'TOGGLE_LABELS',
    TOGGLE_SIDEBAR: 'TOGGLE_SIDEBAR',
    SET_VIDEO_DURATION: 'SET_VIDEO_DURATION',
    SET_SCRUBBING_ACTIVE: 'SET_SCRUBBING_ACTIVE',
    SET_LOADING: 'SET_LOADING',
    SET_ERROR: 'SET_ERROR',
    CLEAR_ERROR: 'CLEAR_ERROR',
    // Attention video actions
    TOGGLE_ATTENTION_MODE: 'TOGGLE_ATTENTION_MODE',
    SET_SELECTED_TOKEN: 'SET_SELECTED_TOKEN',
    SET_AVAILABLE_TOKENS: 'SET_AVAILABLE_TOKENS',
    // Analysis actions
    SET_CURRENT_ANALYSIS: 'SET_CURRENT_ANALYSIS',
    SET_ANALYSIS_LOADING: 'SET_ANALYSIS_LOADING',
    SET_ANALYSIS_ERROR: 'SET_ANALYSIS_ERROR',
    CLEAR_ANALYSIS_ERROR: 'CLEAR_ANALYSIS_ERROR',
    // Schema actions
    SET_ANALYSIS_SCHEMA: 'SET_ANALYSIS_SCHEMA',
    SET_SCHEMA_LOADING: 'SET_SCHEMA_LOADING',
    SET_SCHEMA_ERROR: 'SET_SCHEMA_ERROR',
    // Analysis view actions
    SET_ANALYSIS_VIEW_BY: 'SET_ANALYSIS_VIEW_BY',
    SET_ANALYSIS_CHART_SIZE: 'SET_ANALYSIS_CHART_SIZE'
};

// Reducer
const appReducer = (state, action) => {
    switch (action.type) {
        case ActionTypes.SET_EXPERIMENTS:
            return { ...state, experiments: action.payload, error: null };

        case ActionTypes.SET_EXPERIMENTS_TREE:
            return { ...state, experimentsTree: action.payload, error: null };

        case ActionTypes.SET_CURRENT_EXPERIMENT:
            return {
                ...state,
                currentExperiment: action.payload,
                videoDuration: 0, // Reset duration for new experiment
                error: null,
                // Reset attention mode state when switching experiments
                selectedToken: null,
                availableTokens: action.payload?.attention_videos?.available ? 
                    getAvailableTokensFromAttentionVideos(action.payload.attention_videos) : []
            };

        case ActionTypes.SET_VIDEO_SIZE:
            return { ...state, videoSize: action.payload };

        case ActionTypes.TOGGLE_LABELS:
            return { ...state, showLabels: !state.showLabels };

        case ActionTypes.TOGGLE_SIDEBAR:
            return { ...state, sidebarCollapsed: !state.sidebarCollapsed };

        case ActionTypes.SET_VIDEO_DURATION:
            return { ...state, videoDuration: action.payload };

        case ActionTypes.SET_SCRUBBING_ACTIVE:
            return { ...state, isScrubbingActive: action.payload };

        case ActionTypes.SET_LOADING:
            return { ...state, isLoading: action.payload };

        case ActionTypes.SET_ERROR:
            return { ...state, error: action.payload, isLoading: false };

        case ActionTypes.CLEAR_ERROR:
            return { ...state, error: null };

        case ActionTypes.TOGGLE_ATTENTION_MODE:
            return { ...state, attentionMode: !state.attentionMode };

        case ActionTypes.SET_SELECTED_TOKEN:
            return { ...state, selectedToken: action.payload };

        case ActionTypes.SET_AVAILABLE_TOKENS:
            return { ...state, availableTokens: action.payload };

        case ActionTypes.SET_CURRENT_ANALYSIS:
            return { ...state, currentAnalysis: action.payload, analysisError: null };

        case ActionTypes.SET_ANALYSIS_LOADING:
            return { ...state, analysisLoading: action.payload };

        case ActionTypes.SET_ANALYSIS_ERROR:
            return { ...state, analysisError: action.payload, analysisLoading: false };

        case ActionTypes.CLEAR_ANALYSIS_ERROR:
            return { ...state, analysisError: null };

        case ActionTypes.SET_ANALYSIS_SCHEMA:
            return { ...state, analysisSchema: action.payload, schemaError: null };

        case ActionTypes.SET_SCHEMA_LOADING:
            return { ...state, schemaLoading: action.payload };

        case ActionTypes.SET_SCHEMA_ERROR:
            return { ...state, schemaError: action.payload, schemaLoading: false };

        case ActionTypes.SET_ANALYSIS_VIEW_BY:
            return { ...state, analysisViewBy: action.payload };

        case ActionTypes.SET_ANALYSIS_CHART_SIZE:
            return { ...state, analysisChartSize: action.payload };

        default:
            return state;
    }
};

// Context
const AppContext = createContext();

// Provider component
export const AppProvider = ({ children }) => {
    const [state, dispatch] = useReducer(appReducer, initialState);
    const schemaLoadedRef = useRef(false);

    // Action creators
    const actions = {
        setExperiments: useCallback((experiments) =>
            dispatch({ type: ActionTypes.SET_EXPERIMENTS, payload: experiments }), []),

        setExperimentsTree: useCallback((experimentsTree) =>
            dispatch({ type: ActionTypes.SET_EXPERIMENTS_TREE, payload: experimentsTree }), []),

        setCurrentExperiment: useCallback((experiment) =>
            dispatch({ type: ActionTypes.SET_CURRENT_EXPERIMENT, payload: experiment }), []),

        setVideoSize: useCallback((size) =>
            dispatch({ type: ActionTypes.SET_VIDEO_SIZE, payload: size }), []),

        toggleLabels: useCallback(() =>
            dispatch({ type: ActionTypes.TOGGLE_LABELS }), []),

        toggleSidebar: useCallback(() =>
            dispatch({ type: ActionTypes.TOGGLE_SIDEBAR }), []),

        setVideoDuration: useCallback((duration) =>
            dispatch({ type: ActionTypes.SET_VIDEO_DURATION, payload: duration }), []),

        setScrubbingActive: useCallback((active) =>
            dispatch({ type: ActionTypes.SET_SCRUBBING_ACTIVE, payload: active }), []),

        setLoading: useCallback((loading) =>
            dispatch({ type: ActionTypes.SET_LOADING, payload: loading }), []),

        setError: useCallback((error) =>
            dispatch({ type: ActionTypes.SET_ERROR, payload: error }), []),

        clearError: useCallback(() =>
            dispatch({ type: ActionTypes.CLEAR_ERROR }), []),

        // Attention video actions
        toggleAttentionMode: useCallback(() =>
            dispatch({ type: ActionTypes.TOGGLE_ATTENTION_MODE }), []),

        setSelectedToken: useCallback((token) =>
            dispatch({ type: ActionTypes.SET_SELECTED_TOKEN, payload: token }), []),

        setAvailableTokens: useCallback((tokens) =>
            dispatch({ type: ActionTypes.SET_AVAILABLE_TOKENS, payload: tokens }), []),

        // Analysis actions
        setCurrentAnalysis: useCallback((analysis) =>
            dispatch({ type: ActionTypes.SET_CURRENT_ANALYSIS, payload: analysis }), []),

        setAnalysisLoading: useCallback((loading) =>
            dispatch({ type: ActionTypes.SET_ANALYSIS_LOADING, payload: loading }), []),

        setAnalysisError: useCallback((error) =>
            dispatch({ type: ActionTypes.SET_ANALYSIS_ERROR, payload: error }), []),

        clearAnalysisError: useCallback(() =>
            dispatch({ type: ActionTypes.CLEAR_ANALYSIS_ERROR }), []),

        // Schema actions
        setAnalysisSchema: useCallback((schema) =>
            dispatch({ type: ActionTypes.SET_ANALYSIS_SCHEMA, payload: schema }), []),

        setSchemaLoading: useCallback((loading) =>
            dispatch({ type: ActionTypes.SET_SCHEMA_LOADING, payload: loading }), []),

        setSchemaError: useCallback((error) =>
            dispatch({ type: ActionTypes.SET_SCHEMA_ERROR, payload: error }), []),

        // Analysis view actions
        setAnalysisViewBy: useCallback((viewBy) =>
            dispatch({ type: ActionTypes.SET_ANALYSIS_VIEW_BY, payload: viewBy }), []),

        setAnalysisChartSize: useCallback((size) =>
            dispatch({ type: ActionTypes.SET_ANALYSIS_CHART_SIZE, payload: size }), [])
    };

        // Load analysis schema on app initialization
    useEffect(() => {
        const loadAnalysisSchema = async () => {
            if (schemaLoadedRef.current) return;
            
            try {
                schemaLoadedRef.current = true;
                dispatch({ type: ActionTypes.SET_SCHEMA_LOADING, payload: true });
                const schemaData = await api.getAnalysisSchema();
                dispatch({ type: ActionTypes.SET_ANALYSIS_SCHEMA, payload: schemaData.vlm_analysis_schema });
            } catch (error) {
                console.error('Failed to load analysis schema:', error);
                dispatch({ type: ActionTypes.SET_SCHEMA_ERROR, payload: error.message });
                schemaLoadedRef.current = false; // Reset on error so it can retry
            } finally {
                dispatch({ type: ActionTypes.SET_SCHEMA_LOADING, payload: false });
            }
        };

        loadAnalysisSchema();
    }, []); // Empty dependency array

    return (
        <AppContext.Provider value={{ state, actions }}>
            {children}
        </AppContext.Provider>
    );
};

// Custom hook to use the context
export const useApp = () => {
    const context = useContext(AppContext);
    if (!context) {
        throw new Error('useApp must be used within an AppProvider');
    }
    return context;
};

export default AppContext;
