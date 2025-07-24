import React from 'react';
import { useApp } from '../../context/AppContext';
import ExperimentList from './ExperimentList';

const Sidebar = () => {
    const { state, actions } = useApp();
    const { sidebarCollapsed } = state;

    return (
        <div className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
            <div className="sidebar-header">
                <h1>WAN Video Matrix</h1>
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
                <ExperimentList />
            </div>
        </div>
    );
};

export default Sidebar;
