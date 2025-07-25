# WAN Video Matrix Viewer

A modern React-based web application for viewing and comparing WAN AI-generated video experiments in a matrix layout.

## Features

- 🎬 **Matrix View**: Organize videos by prompt variations and seeds
- 🎮 **Interactive Controls**: Hover to play, synchronized controls
- 📱 **Responsive Design**: Works on desktop and mobile
- 🔄 **Real-time Updates**: Automatic experiment scanning
- 🎨 **Modern UI**: React with dark theme and smooth animations
- ⚡ **Fast Development**: Hot Module Replacement with Vite

## Architecture

This application follows modern full-stack development practices:

```
webapp/
├── backend/              # Flask API server
│   ├── app.py           # Main Flask application
│   └── requirements.txt # Python dependencies
├── react-frontend/       # React application
│   ├── src/             # React source code
│   │   ├── components/  # React components
│   │   ├── context/     # React context
│   │   ├── hooks/       # Custom hooks
│   │   └── services/    # API services
│   ├── dist/            # Built React app
│   └── package.json     # Frontend dependencies
├── docker/              # Container deployment
├── scripts/             # Build and deployment
└── package.json         # Root package.json with scripts
```

## Quick Start

### Development Mode

```bash
cd webapp

# Install all dependencies
npm run install:all

# Start both backend and frontend
npm run dev
```

This will:
1. Start the Flask backend on `http://localhost:5000`
2. Start the Vite dev server on `http://localhost:3001`
3. Set up API proxying from frontend to backend

Access the app at `http://localhost:3001` for development.

### Production Build

```bash
cd webapp

# Build the React frontend
npm run build

# Start the backend (serves the built React app)
npm run serve
```

Access the production app at `http://localhost:5000`.

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
