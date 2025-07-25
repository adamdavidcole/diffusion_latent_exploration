import React, { createContext, useContext, useReducer, useCallback } from 'react';

// Initial state
const initialState = {
    currentExperiment: null,
    experiments: [],
    videoSize: 200,
    showLabels: true,
    sidebarCollapsed: true,
    videoDuration: 0,
    isScrubbingActive: false,
    isLoading: true, // Start with loading true for initial skeleton
    error: null
};

// Action types
const ActionTypes = {
    SET_EXPERIMENTS: 'SET_EXPERIMENTS',
    SET_CURRENT_EXPERIMENT: 'SET_CURRENT_EXPERIMENT',
    SET_VIDEO_SIZE: 'SET_VIDEO_SIZE',
    TOGGLE_LABELS: 'TOGGLE_LABELS',
    TOGGLE_SIDEBAR: 'TOGGLE_SIDEBAR',
    SET_VIDEO_DURATION: 'SET_VIDEO_DURATION',
    SET_SCRUBBING_ACTIVE: 'SET_SCRUBBING_ACTIVE',
    SET_LOADING: 'SET_LOADING',
    SET_ERROR: 'SET_ERROR',
    CLEAR_ERROR: 'CLEAR_ERROR'
};

// Reducer
const appReducer = (state, action) => {
    switch (action.type) {
        case ActionTypes.SET_EXPERIMENTS:
            return { ...state, experiments: action.payload, error: null };

        case ActionTypes.SET_CURRENT_EXPERIMENT:
            return {
                ...state,
                currentExperiment: action.payload,
                videoDuration: 0, // Reset duration for new experiment
                error: null
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

        default:
            return state;
    }
};

// Context
const AppContext = createContext();

// Provider component
export const AppProvider = ({ children }) => {
    const [state, dispatch] = useReducer(appReducer, initialState);

    // Action creators
    const actions = {
        setExperiments: useCallback((experiments) =>
            dispatch({ type: ActionTypes.SET_EXPERIMENTS, payload: experiments }), []),

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
            dispatch({ type: ActionTypes.CLEAR_ERROR }), [])
    };

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
