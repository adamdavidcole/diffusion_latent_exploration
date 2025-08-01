import React, { useState, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import TreeExperimentList from './TreeExperimentList';

const Sidebar = () => {
    const { state, actions } = useApp();
    const { sidebarCollapsed, isLoading } = state;
    const [rescanFn, setRescanFn] = useState(null);

    const handleRescanRef = useCallback((rescanFunction) => {
        setRescanFn(() => rescanFunction);
    }, []);

    return (
        <div className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
            <div className="sidebar-header">
                {!sidebarCollapsed && <h1>WAN Video Matrix</h1>}
                <button
                    className="collapse-btn"
                    onClick={actions.toggleSidebar}
                    aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                >
                    <span className="collapse-icon">
                        {sidebarCollapsed ? '→' : '←'}
                    </span>
                </button>
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
