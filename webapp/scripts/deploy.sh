#!/bin/bash
# WAN Video Viewer - Production Deployment Script

set -e

echo "🚀 Building WAN Video Viewer for production..."

# Build all assets
npm run build:css
npm run build:js
npm run build:dev-css  # Also build dev versions for fallback

echo "📦 Copying production assets to dist directory..."

# Copy minified assets to production static directory
cp frontend/static/dist/*.min.* dist/static/

# Copy the single template (environment-aware)
cp frontend/templates/index.html dist/templates/index.html

echo "✅ Production build complete!"
echo ""
echo "📂 Files deployed to:"
echo "   - Static assets: dist/static/"
echo "   - Templates: dist/templates/"
echo ""
echo "🏃 To run in production mode:"
echo "   cd dist && python backend/app.py"
echo ""
echo "🛠️  To run in development mode:"
echo "   npm start"
