#!/bin/bash
# Development server script

set -e

echo "🚀 Starting WAN Video Matrix Viewer - Development Mode"
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r backend/requirements.txt

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi

# Build frontend assets
echo "Building frontend assets..."
npm run build

# Start the Flask development server
echo "Starting Flask development server..."
echo "Access the app at: http://localhost:5000"
echo "=============================================="

cd backend
python app.py
