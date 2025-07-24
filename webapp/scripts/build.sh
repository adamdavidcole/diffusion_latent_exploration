#!/bin/bash
# Production deployment script

set -e

echo "🏗️ Building WAN Video Matrix Viewer for Production"
echo "=============================================="

# Create production build directory
mkdir -p dist

# Build Python backend
echo "Setting up Python backend..."
cp -r backend dist/
cd dist/backend

# Install production dependencies
pip install -r requirements.txt --target ./vendor

# Create production requirements
pip freeze > requirements.prod.txt

cd ../..

# Build frontend
echo "Building frontend assets..."
# Check Node version and adjust build strategy
NODE_VERSION=$(node --version | sed 's/v//' | cut -d. -f1)

if [ "$NODE_VERSION" -lt 14 ]; then
    echo "Node.js version $NODE_VERSION detected. Using simple asset copying (no build pipeline)."
    # Simple approach - just copy the files as-is
    mkdir -p dist/static/css dist/static/js
    cp frontend/static/css/*.css dist/static/css/
    cp frontend/static/js/*.js dist/static/js/
else
    echo "Node.js version $NODE_VERSION detected. Using modern build pipeline."
    if [ ! -d "node_modules" ]; then
        echo "Installing Node.js dependencies..."
        npm install
    fi
    
    # Try modern build, fallback if it fails
    if timeout 30 npm run build:js; then
        echo "JavaScript build successful"
    else
        echo "JavaScript build failed, copying source files"
        mkdir -p dist/static/js
        cp frontend/static/js/*.js dist/static/js/
    fi
    
    # Try CSS build with timeout
    if timeout 30 npm run build:css; then
        echo "CSS build successful"
    else
        echo "CSS build failed or timed out, copying source files"
        mkdir -p dist/static/css
        cp frontend/static/css/*.css dist/static/css/
    fi
    
    # Copy built assets if they exist, otherwise source files
    if [ -d "frontend/static/dist" ]; then
        cp -r frontend/static/dist/* dist/static/ 2>/dev/null || true
    fi
    
    # Ensure we have CSS files
    if [ ! -f "dist/static/css/main.css" ] && [ ! -f "dist/static/main.min.css" ]; then
        mkdir -p dist/static/css
        cp frontend/static/css/*.css dist/static/css/
    fi
fi
cp -r frontend/templates dist/templates

# Create production server script
cat > dist/server.py << 'EOF'
#!/usr/bin/env python3
import sys
import os

# Add vendor directory to Python path
vendor_dir = os.path.join(os.path.dirname(__file__), 'backend', 'vendor')
sys.path.insert(0, vendor_dir)

# Add backend directory to Python path  
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_dir)

# Add parent directory for src imports
parent_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, parent_dir)

try:
    from app import create_app
    print("🎬 WAN Video Matrix Viewer - Production Server")
    print("=" * 50)
    
    app = create_app()
    
    if __name__ == '__main__':
        port = int(os.environ.get('PORT', 5000))
        print(f"Starting production server on port {port}...")
        print("=" * 50)
        app.run(host='0.0.0.0', port=port, debug=False)

except ImportError as e:
    print(f"Import error: {e}")
    print("Available modules:")
    for path in sys.path[:5]:  # Show first 5 paths
        print(f"  {path}")
        if os.path.exists(path):
            try:
                files = [f for f in os.listdir(path) if f.endswith('.py')][:5]
                for f in files:
                    print(f"    {f}")
            except:
                pass
    sys.exit(1)
EOF

# Make server script executable
chmod +x dist/server.py

# Create systemd service file
cat > dist/wan-video-viewer.service << EOF
[Unit]
Description=WAN Video Matrix Viewer
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/wan-video-viewer
Environment=PATH=/opt/wan-video-viewer/venv/bin
ExecStart=/opt/wan-video-viewer/venv/bin/python server.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Create nginx configuration
cat > dist/nginx.conf << EOF
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    location /static/ {
        alias /opt/wan-video-viewer/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
EOF

echo "✅ Production build complete in ./dist"
echo "=============================================="
echo "Files created:"
ls -la dist/
echo ""
echo "To deploy:"
echo "1. Copy dist/ to your server: scp -r dist/ user@server:/opt/wan-video-viewer"
echo "2. Install system dependencies: sudo apt install python3-pip python3-venv nginx"
echo "3. Create virtual environment: python3 -m venv /opt/wan-video-viewer/venv"
echo "4. Install dependencies: /opt/wan-video-viewer/venv/bin/pip install -r requirements.txt"
echo "5. Configure nginx: sudo cp nginx.conf /etc/nginx/sites-available/wan-video-viewer"
echo "6. Enable site: sudo ln -s /etc/nginx/sites-available/wan-video-viewer /etc/nginx/sites-enabled/"
echo "7. Install systemd service: sudo cp wan-video-viewer.service /etc/systemd/system/"
echo "8. Start services: sudo systemctl daemon-reload && sudo systemctl enable wan-video-viewer && sudo systemctl start wan-video-viewer"
echo ""
echo "Test deployment:"
echo "cd dist && python server.py"
