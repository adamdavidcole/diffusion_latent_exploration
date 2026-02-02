import React, { useState, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import TreeExperimentList from './TreeExperimentList';

const Sidebar = () => {
    const { state, actions } = useApp();
    const { sidebarCollapsed, sidebarHidden, isLoading } = state;
    const [rescanFn, setRescanFn] = useState(null);

    const handleRescanRef = useCallback((rescanFunction) => {
        setRescanFn(() => rescanFunction);
    }, []);

    // Don't render sidebar at all when hidden
    if (sidebarHidden) {
        return null;
    }

    return (
        <div className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
            <div className="sidebar-header">
                {!sidebarCollapsed && <h1>Diffusion Explorer</h1>}
                <div className="sidebar-header-buttons">
                    <button
                        className="hide-sidebar-btn"
                        onClick={actions.toggleSidebarHidden}
                        aria-label="Hide sidebar"
                        title="Hide Sidebar"
                    >
                        <span className="hide-icon">
                            <span style={{ position: 'relative' }}>
                                <span style={{ position: 'absolute', left: '-5px', top: '0px', fontSize: "0.66rem" }}>←</span>
                                <span>☰</span>
                            </span>
                        </span>
                    </button>
                    <button
                        className="collapse-btn"
                        onClick={actions.toggleSidebar}
                        aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                        title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                    >
                        <span className="collapse-icon">
                            {sidebarCollapsed ? '→' : '←'}
                        </span>
                    </button>
                </div>
            </div>

            <div className="sidebar-content">
                <TreeExperimentList onRescan={handleRescanRef} />
            </div>

            <div className="rescan-section">
                <button
                    className="rescan-button control-btn"
                    onClick={rescanFn}
                    disabled={isLoading || !rescanFn}
                    title="Rescan for new experiments"
                >
                    {isLoading ? '...' : '🔄'}
                </button>
            </div>
        </div>
    );
};

export default Sidebar;
