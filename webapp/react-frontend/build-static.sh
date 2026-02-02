#!/bin/bash
# Build static site for deployment
# Usage: ./build-static.sh

set -e  # Exit on error

echo "========================================"
echo "🏗️  Building Static Site"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found"
    echo "   Run this script from webapp/react-frontend/ directory"
    exit 1
fi

# Check if data directory exists
if [ ! -d "public/data" ]; then
    echo "⚠️  Warning: public/data/ not found"
    echo "   Run export script first:"
    echo "   python ../../scripts/export_static_data.py BATCH_NAME"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get R2 URL from environment or use default
R2_URL="${VITE_R2_BASE_URL:-https://pub-c3b8273c4ebb40fbb357141ca4767c03.r2.dev}"

echo "📋 Build Configuration:"
echo "   VITE_STATIC_MODE=true"
echo "   VITE_R2_BASE_URL=$R2_URL"
echo ""

# Confirm before building
read -p "Proceed with build? (Y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Build cancelled."
    exit 0
fi

echo ""
echo "🔧 Installing dependencies..."
npm install

echo ""
echo "🏗️  Building for production..."
VITE_STATIC_MODE=true VITE_R2_BASE_URL="$R2_URL" npm run build

echo ""
echo "========================================"
echo "✅ Build Complete!"
echo "========================================"
echo ""
echo "📂 Output: dist/"
echo ""
echo "📊 Build size:"
du -sh dist/
echo ""
echo "📁 Build contents:"
ls -lh dist/ | tail -n +2
echo ""
echo "📌 Next steps:"
echo "   1. Test locally:"
echo "      npx serve dist"
echo ""
echo "   2. Deploy to Netlify:"
echo "      netlify deploy --prod --dir=dist"
echo "      OR drag dist/ folder to https://app.netlify.com/drop"
echo ""
echo "========================================"
