// WAN Video Matrix Viewer - UI Controls and Utilities

// Global functions that need to be called from HTML onclick handlers
// These bridge the gap between the class-based app and inline event handlers

function updateVideoSize(size) {
    if (window.app) {
        window.app.updateVideoSize(size);
    }
}

function toggleLabels() {
    if (window.app) {
        window.app.toggleLabels();
    }
}

function toggleSidebar() {
    if (window.app) {
        window.app.toggleSidebar();
    }
}

function playVideo(videoCell) {
    if (window.app) {
        window.app.playVideo(videoCell);
    }
}

function playAllVideos() {
    if (window.app) {
        window.app.playAllVideos();
    }
}

function pauseAllVideos() {
    if (window.app) {
        window.app.pauseAllVideos();
    }
}

function muteAllVideos() {
    if (window.app) {
        const anyUnmuted = window.app.muteAllVideos();
        
        // Update button text
        const button = event.currentTarget;
        button.textContent = anyUnmuted ? '🔊 Unmute All' : '🔇 Mute All';
    }
}

function rescanExperiments() {
    if (window.app) {
        window.app.rescanExperiments();
    }
}

// Explicitly attach functions to window to ensure they're available globally after minification
window.updateVideoSize = updateVideoSize;
window.toggleLabels = toggleLabels;
window.toggleSidebar = toggleSidebar;
window.playVideo = playVideo;
window.playAllVideos = playAllVideos;
window.pauseAllVideos = pauseAllVideos;
window.muteAllVideos = muteAllVideos;
window.rescanExperiments = rescanExperiments;

// Utility functions
class UIUtils {
    static formatFileSize(bytes) {
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        if (bytes === 0) return '0 Bytes';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }
    
    static formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    static debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    static throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        }
    }
}

// Export for potential module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UIUtils };
}
