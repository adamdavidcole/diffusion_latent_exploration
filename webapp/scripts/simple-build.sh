#!/bin/bash
# Simple production deployment script (no npm required)

set -e

echo "🏗️ Building WAN Video Matrix Viewer for Production (Simple)"
echo "=============================================="

# Create production build directory
rm -rf dist
mkdir -p dist

# Build Python backend
echo "Setting up Python backend..."
cp -r backend dist/

# Create vendor directory for dependencies
mkdir -p dist/backend/vendor
cd dist/backend

# Install production dependencies locally
pip install -r requirements.txt --target ./vendor --quiet

# Create production requirements
pip freeze > requirements.prod.txt

cd ../..

# Copy frontend assets (no build step needed)
echo "Copying frontend assets..."
mkdir -p dist/static/css dist/static/js
cp frontend/static/css/*.css dist/static/css/
cp frontend/static/js/*.js dist/static/js/
cp -r frontend/templates dist/templates

# Update the app.py to use the correct paths
cat > dist/backend/app_prod.py << 'EOF'
#!/usr/bin/env python3
"""
Production version of the WAN Video Matrix Viewer Flask app
"""

import sys
import os

# Add vendor directory to Python path
vendor_dir = os.path.join(os.path.dirname(__file__), 'vendor')
sys.path.insert(0, vendor_dir)

# Import the original app with modified paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from pathlib import Path
from flask import Flask, jsonify, send_file, render_template, send_from_directory
import json
import yaml
from datetime import datetime

# Copy the VideoAnalyzer class and create_app function from app.py
# but modify paths for production
EOF

# Copy the main logic from app.py but modify for production structure
python3 << 'EOF'
import re

# Read the original app.py
with open('backend/app.py', 'r') as f:
    content = f.read()

# Extract the VideoAnalyzer class and functions
class_match = re.search(r'class VideoAnalyzer:.*?(?=^class|\Z)', content, re.MULTILINE | re.DOTALL)
functions_match = re.search(r'def create_app\(\):.*?(?=^if __name__|\Z)', content, re.MULTILINE | re.DOTALL)

# Modify paths for production
production_code = '''#!/usr/bin/env python3
"""
Production version of the WAN Video Matrix Viewer Flask app
"""

import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from flask import Flask, jsonify, send_file, render_template, send_from_directory

# Add vendor directory to Python path
vendor_dir = os.path.join(os.path.dirname(__file__), 'vendor')
sys.path.insert(0, vendor_dir)

try:
    from utils.file_utils import safe_filename
except ImportError:
    def safe_filename(filename):
        return filename

''' + (class_match.group(0) if class_match else '') + '''

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    
    # Configuration
    app.config['VIDEO_OUTPUTS_DIR'] = str(Path(__file__).parent.parent.parent.parent / 'outputs')
    
    # Initialize video analyzer
    analyzer = VideoAnalyzer(app.config['VIDEO_OUTPUTS_DIR'])
    
''' + (functions_match.group(0)[len('def create_app():'):] if functions_match else '') + '''

if __name__ == '__main__':
    app = create_app()
    
    print("🎬 WAN Video Matrix Viewer - Production")
    print("="*50)
    print(f"Outputs directory: {app.config['VIDEO_OUTPUTS_DIR']}")
    print("Starting production server...")
    print("="*50)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
'''

with open('dist/server.py', 'w') as f:
    f.write(production_code)
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
Environment=PYTHONPATH=/opt/wan-video-viewer/backend/vendor
ExecStart=/opt/wan-video-viewer/venv/bin/python server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Create nginx configuration
cat > dist/nginx.conf << EOF
server {
    listen 80;
    server_name your-domain.com;
    
    client_max_body_size 100M;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }
    
    location /static/ {
        alias /opt/wan-video-viewer/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    location /api/video/ {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_read_timeout 300;
        proxy_buffering off;
    }
}
EOF

# Create deployment script
cat > dist/deploy.sh << 'EOF'
#!/bin/bash
# Deployment script for WAN Video Matrix Viewer

set -e

DEPLOY_DIR="/opt/wan-video-viewer"
SERVICE_NAME="wan-video-viewer"

echo "🚀 Deploying WAN Video Matrix Viewer"
echo "=================================="

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)" 
   exit 1
fi

# Install system dependencies
echo "Installing system dependencies..."
apt update
apt install -y python3 python3-pip python3-venv nginx

# Create deployment directory
echo "Creating deployment directory..."
mkdir -p $DEPLOY_DIR
cp -r * $DEPLOY_DIR/

# Create virtual environment
echo "Creating Python virtual environment..."
cd $DEPLOY_DIR
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install Python dependencies from vendor
echo "Python dependencies already included in vendor/ directory"

# Configure nginx
echo "Configuring nginx..."
cp nginx.conf /etc/nginx/sites-available/$SERVICE_NAME
ln -sf /etc/nginx/sites-available/$SERVICE_NAME /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx

# Install systemd service
echo "Installing systemd service..."
cp $SERVICE_NAME.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable $SERVICE_NAME

# Start the service
echo "Starting service..."
systemctl restart $SERVICE_NAME

# Check status
echo "Checking service status..."
systemctl status $SERVICE_NAME --no-pager

echo "✅ Deployment complete!"
echo "Service should be running at: http://$(hostname -I | awk '{print $1}')"
echo "Check logs with: sudo journalctl -u $SERVICE_NAME -f"
EOF

chmod +x dist/deploy.sh

echo "✅ Production build complete in ./dist"
echo "=============================================="
echo "Files created:"
ls -la dist/
echo ""
echo "To test locally:"
echo "cd dist && python server.py"
echo ""
echo "To deploy to server:"
echo "1. Copy dist/ to your server: scp -r dist/ user@server:/tmp/"
echo "2. Run deployment script: sudo /tmp/dist/deploy.sh"
echo ""
echo "Manual deployment steps:"
echo "1. Copy dist/ to /opt/wan-video-viewer"
echo "2. Install system dependencies: sudo apt install python3-pip python3-venv nginx"
echo "3. Run the included deploy.sh script as root"
