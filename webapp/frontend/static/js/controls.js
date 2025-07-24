// WAN Video Matrix Viewer - UI Controls and Utilities

// Global functions that need to be called from HTML onclick handlers
// These bridge the gap between the class-based app and inline event handlers

function updateVideoSize(size) {
    app.updateVideoSize(size);
}

function toggleLabels() {
    app.toggleLabels();
}

function toggleSidebar() {
    app.toggleSidebar();
}

function playVideo(videoCell) {
    app.playVideo(videoCell);
}

function playAllVideos() {
    app.playAllVideos();
}

function pauseAllVideos() {
    app.pauseAllVideos();
}

function muteAllVideos() {
    const anyUnmuted = app.muteAllVideos();
    
    // Update button text
    const button = event.currentTarget;
    button.textContent = anyUnmuted ? '🔊 Unmute All' : '🔇 Mute All';
}

function rescanExperiments() {
    app.rescanExperiments();
}

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
