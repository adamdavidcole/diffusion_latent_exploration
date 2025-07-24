# WAN Video Matrix Viewer

A modern web application for viewing and comparing WAN AI-generated video experiments in a matrix layout.

## Features

- 🎬 **Matrix View**: Organize videos by prompt variations and seeds
- 🎮 **Interactive Controls**: Hover to play, synchronized controls
- 📱 **Responsive Design**: Works on desktop and mobile
- 🔄 **Real-time Updates**: Automatic experiment scanning
- 🎨 **Professional UI**: Dark theme with smooth animations

## Architecture

This application follows modern web development practices with a clean separation of concerns:

```
webapp/
├── backend/           # Flask API server
│   ├── app.py        # Main application
│   └── requirements.txt
├── frontend/         # Static assets
│   ├── static/
│   │   ├── css/      # Stylesheets
│   │   ├── js/       # JavaScript modules
│   │   └── dist/     # Built assets
│   └── templates/    # HTML templates
├── docker/           # Container deployment
├── scripts/          # Build and deployment
└── package.json      # Frontend build tools
```

## Quick Start

### Development Mode

```bash
cd webapp
./scripts/dev.sh
```

This will:
1. Create a Python virtual environment
2. Install all dependencies
3. Build frontend assets
4. Start the development server

Access the app at `http://localhost:5000`

### Manual Setup

1. **Backend Setup**:
```bash
cd webapp
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

2. **Frontend Setup**:
```bash
npm install
npm run build
```

3. **Start Development Server**:
```bash
cd backend
python app.py
```

## Build Commands

- `npm run build` - Build production assets
- `npm run dev` - Start development watchers
- `npm run watch:css` - Watch CSS files
- `npm run watch:js` - Watch JavaScript files

## Deployment

### Docker Deployment (Recommended)

```bash
cd webapp/docker
docker-compose up -d
```

### Manual Production Deployment

```bash
./scripts/build.sh
```

This creates a `dist/` folder with:
- Production-optimized assets
- Systemd service file
- Nginx configuration
- Deployment instructions

### Environment Variables

- `FLASK_ENV` - Set to `production` for production
- `FLASK_DEBUG` - Set to `0` for production
- `PORT` - Server port (default: 5000)

## API Endpoints

- `GET /` - Main application
- `GET /api/experiments` - List all experiments
- `GET /api/experiment/<name>` - Get experiment details
- `GET /api/video/<path>` - Serve video files
- `GET /api/scan` - Rescan experiments

## File Structure Requirements

The application expects this output structure:

```
outputs/
└── experiment_name/
    ├── configs/
    │   └── generation_config.yaml
    └── videos/
        └── prompt_000/
            ├── video_seed_1234.mp4
            ├── video_seed_5678.mp4
            └── ...
```

## Modern Features

### Frontend
- **Modular JavaScript**: ES6 classes and modules
- **CSS Custom Properties**: Modern CSS with CSS variables
- **PostCSS Processing**: Autoprefixer and optimization
- **Webpack Bundling**: Code splitting and optimization

### Backend
- **Flask Architecture**: Clean API separation
- **CORS Support**: Frontend/backend separation ready
- **File Streaming**: Efficient video delivery
- **Error Handling**: Comprehensive error responses

### Deployment
- **Docker Support**: Multi-stage builds
- **nginx Integration**: Static file serving
- **Health Checks**: Container monitoring
- **Production Scripts**: Automated deployment

## Browser Support

- Chrome/Chromium 88+
- Firefox 85+
- Safari 14+
- Edge 88+

## Development vs React

### Current Architecture Benefits:
- ✅ **Simple Setup**: No complex build pipeline
- ✅ **Fast Development**: Direct file editing
- ✅ **Small Bundle**: Minimal JavaScript
- ✅ **Server-Side Ready**: Flask template integration

### React Migration Path:
If you want to migrate to React later:

1. **API is Ready**: Backend already provides clean JSON API
2. **Styling is Modular**: CSS can be imported as-is
3. **Component Structure**: Current code maps well to React components
4. **State Management**: App class can become React context

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `npm run dev`
5. Build with `npm run build`
6. Submit a pull request

## License

MIT License - see LICENSE file for details
