// WAN Video Matrix Viewer - Core Application Logic

class VideoMatrixApp {
    constructor() {
        this.currentExperiment = null;
        this.allVideos = [];
        this.videoSize = 200;
        this.showLabels = true;
        this.sidebarCollapsed = false;
        
        this.init();
    }
    
    init() {
        document.addEventListener('DOMContentLoaded', () => {
            this.loadExperiments();
            this.setupEventListeners();
        });
    }
    
    setupEventListeners() {
        // Show sync controls when videos are loaded
        const observer = new MutationObserver((mutations) => {
            const hasVideos = document.querySelectorAll('video').length > 0;
            const syncControls = document.getElementById('sync-controls');
            if (hasVideos) {
                syncControls.classList.add('visible');
            } else {
                syncControls.classList.remove('visible');
            }
        });
        
        observer.observe(document.getElementById('video-grid'), {
            childList: true,
            subtree: true
        });
        
        // Mobile responsiveness
        if (window.innerWidth <= 900) {
            document.getElementById('sidebar').classList.add('collapsed');
        }
    }
    
    async loadExperiments() {
        try {
            const response = await fetch('/api/experiments');
            const experiments = await response.json();
            
            this.renderExperimentsList(experiments);
        } catch (error) {
            console.error('Error loading experiments:', error);
            document.getElementById('experiments-list').innerHTML = `
                <div class="empty-state">
                    <h3>Error loading experiments</h3>
                    <p>Check the server connection</p>
                </div>
            `;
        }
    }
    
    renderExperimentsList(experiments) {
        const container = document.getElementById('experiments-list');
        
        if (experiments.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <h3>No experiments found</h3>
                    <p>Generate some videos first, then rescan.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = experiments.map(exp => `
            <div class="experiment-item" onclick="app.selectExperiment('${exp.name}', this)">
                <div class="experiment-header">
                    <div class="experiment-name">${exp.name}</div>
                    <div class="experiment-meta">
                        <span>${exp.videos_count} videos</span>
                        <span>${exp.variations_count} variations</span>
                        <span>${exp.seeds_count} seeds</span>
                    </div>
                    <div class="experiment-prompt">${exp.base_prompt}</div>
                </div>
            </div>
        `).join('');
    }
    
    async selectExperiment(experimentName, element) {
        // Update active state
        document.querySelectorAll('.experiment-item').forEach(item => {
            item.classList.remove('active');
        });
        element.classList.add('active');

        // Show loading
        document.getElementById('loading').style.display = 'flex';
        document.getElementById('video-grid-wrapper').style.display = 'none';
        document.getElementById('empty-state').style.display = 'none';

        try {
            const response = await fetch(`/api/experiment/${experimentName}`);
            const experiment = await response.json();
            
            if (experiment.error) {
                throw new Error(experiment.error);
            }

            this.currentExperiment = experiment;
            this.renderExperiment(experiment);
            
        } catch (error) {
            console.error('Error loading experiment:', error);
            document.getElementById('loading').style.display = 'none';
            document.getElementById('empty-state').style.display = 'block';
        }
    }
    
    renderExperiment(experiment) {
        // Update header
        document.getElementById('experiment-title').textContent = experiment.name;
        document.getElementById('base-prompt').textContent = experiment.base_prompt;

        // Render seeds header
        const seedsHeader = document.getElementById('seeds-header');
        seedsHeader.innerHTML = experiment.seeds.map(seed => 
            `<div class="seed-label" style="width: ${this.videoSize}px;">Seed ${seed}</div>`
        ).join('');

        // Render video grid
        const videoGrid = document.getElementById('video-grid');
        videoGrid.innerHTML = experiment.video_grid.map(row => `
            <div class="grid-row">
                <div class="row-label ${this.showLabels ? '' : 'hidden'}">${row.variation}</div>
                <div class="videos-row">
                    ${experiment.seeds.map(seed => {
                        const video = row.videos.find(v => v && v.seed === seed);
                        if (video) {
                            return `
                                <div class="video-cell" onclick="app.playVideo(this)">
                                    <video class="video-element" 
                                           style="width: ${this.videoSize}px; height: ${Math.round(this.videoSize * 0.56)}px;"
                                           muted loop preload="metadata">
                                        <source src="/api/video/${video.video_path}" type="video/mp4">
                                    </video>
                                    <div class="video-overlay">
                                        <div>Seed: ${video.seed}</div>
                                        <div>${video.width}x${video.height}, ${video.num_frames}f</div>
                                        <div>Steps: ${video.steps}, CFG: ${video.cfg_scale}</div>
                                    </div>
                                </div>
                            `;
                        } else {
                            return `
                                <div class="video-placeholder" 
                                     style="width: ${this.videoSize}px; height: ${Math.round(this.videoSize * 0.56)}px;">
                                    Missing
                                </div>
                            `;
                        }
                    }).join('')}
                </div>
            </div>
        `).join('');

        // Store all video elements
        this.allVideos = Array.from(document.querySelectorAll('video'));

        // Setup hover to play
        this.setupVideoHoverPlay();

        // Hide loading, show content
        document.getElementById('loading').style.display = 'none';
        if (experiment.video_grid.length > 0) {
            document.getElementById('video-grid-wrapper').style.display = 'block';
        } else {
            document.getElementById('empty-state').style.display = 'block';
        }
    }
    
    setupVideoHoverPlay() {
        this.allVideos.forEach(video => {
            video.addEventListener('mouseenter', function() {
                if (!this.playing) {
                    this.play();
                }
            });
            
            video.addEventListener('mouseleave', function() {
                this.pause();
                this.currentTime = 0;
            });
        });
    }
    
    updateVideoSize(size) {
        this.videoSize = parseInt(size);
        
        // Update all videos
        document.querySelectorAll('.video-element').forEach(video => {
            video.style.width = `${this.videoSize}px`;
            video.style.height = `${Math.round(this.videoSize * 0.56)}px`;
        });
        
        // Update placeholders
        document.querySelectorAll('.video-placeholder').forEach(placeholder => {
            placeholder.style.width = `${this.videoSize}px`;
            placeholder.style.height = `${Math.round(this.videoSize * 0.56)}px`;
        });
        
        // Update seed labels
        document.querySelectorAll('.seed-label').forEach(label => {
            label.style.width = `${this.videoSize}px`;
        });
    }
    
    toggleLabels() {
        this.showLabels = !this.showLabels;
        const toggle = document.getElementById('labels-toggle');
        const labels = document.querySelectorAll('.row-label');
        
        if (this.showLabels) {
            toggle.textContent = 'Hide Labels';
            toggle.classList.remove('active');
            labels.forEach(label => label.classList.remove('hidden'));
        } else {
            toggle.textContent = 'Show Labels';
            toggle.classList.add('active');
            labels.forEach(label => label.classList.add('hidden'));
        }
    }
    
    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        const icon = document.getElementById('collapse-icon');
        
        this.sidebarCollapsed = !this.sidebarCollapsed;
        
        if (this.sidebarCollapsed) {
            sidebar.classList.add('collapsed');
            icon.textContent = '→';
        } else {
            sidebar.classList.remove('collapsed');
            icon.textContent = '←';
        }
    }
    
    playVideo(videoCell) {
        const video = videoCell.querySelector('video');
        if (video.paused) {
            video.play();
            video.playing = true;
        } else {
            video.pause();
            video.playing = false;
        }
    }
    
    playAllVideos() {
        this.allVideos.forEach(video => {
            video.currentTime = 0;
            video.play();
            video.playing = true;
        });
    }
    
    pauseAllVideos() {
        this.allVideos.forEach(video => {
            video.pause();
            video.playing = false;
        });
    }
    
    muteAllVideos() {
        const anyUnmuted = this.allVideos.some(video => !video.muted);
        this.allVideos.forEach(video => {
            video.muted = anyUnmuted;
        });
        
        // Update button text - need to find the button from event context
        // This will be handled by the onclick handler
        return anyUnmuted;
    }
    
    async rescanExperiments() {
        try {
            await fetch('/api/scan');
            await this.loadExperiments();
        } catch (error) {
            console.error('Error rescanning:', error);
        }
    }
}

// Initialize the application
const app = new VideoMatrixApp();
