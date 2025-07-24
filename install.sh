#!/bin/bash
# Brain Source Localization Analysis Platform - Installation Script
# This script sets up the environment and installs all dependencies

set -e  # Exit on any error

echo "🧠 Brain Source Localization Analysis Platform - Installation"
echo "============================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d" " -f2 | cut -d"." -f1-2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv brain_analysis_env
source brain_analysis_env/bin/activate

echo "✅ Virtual environment created and activated"

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Dependencies installed successfully"

# Validate MNE installation
echo "🔍 Validating MNE installation..."
python3 -c "import mne; print(f'✅ MNE version: {mne.__version__}')"

# Validate PyVista installation
echo "🔍 Validating PyVista installation..."
python3 -c "import pyvista as pv; print(f'✅ PyVista version: {pv.__version__}')"

# Validate Dash installation
echo "🔍 Validating Dash installation..."
python3 -c "import dash; print(f'✅ Dash version: {dash.__version__}')"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p brain_images
mkdir -p custom_brain_images
mkdir -p exported_data
mkdir -p music_cache/icu

echo "✅ Directory structure created"

# Check for data files
echo "🔍 Checking for data files..."
if [ -d "music_cache/icu" ] && [ "$(ls -A music_cache/icu)" ]; then
    stc_count=$(find music_cache/icu -name "*.stc" | wc -l)
    echo "✅ Found $stc_count .stc data files"
else
    echo "⚠️  No data files found in music_cache/icu/"
    echo "   Place your .stc source estimate files there to begin analysis"
fi

# Create quick start script
echo "📝 Creating quick start script..."
cat > start_brain_analyzer.sh << 'EOF'
#!/bin/bash
# Quick start script for Brain Source Localization Analysis Platform

echo "🧠 Starting Brain Source Localization Analysis Platform..."

# Activate virtual environment
source brain_analysis_env/bin/activate

# Check if data exists
if [ ! -d "music_cache/icu" ] || [ ! "$(ls -A music_cache/icu)" ]; then
    echo "⚠️  Warning: No data files found!"
    echo "   Please place your .stc files in music_cache/icu/ directory"
    echo ""
fi

# Start the application
python3 modern_brain_explorer.py

EOF

chmod +x start_brain_analyzer.sh

echo "✅ Quick start script created: start_brain_analyzer.sh"

# Installation summary
echo ""
echo "🎉 Installation Complete!"
echo "========================="
echo ""
echo "📋 Next Steps:"
echo "   1. Place your .stc data files in: music_cache/icu/"
echo "   2. Run: ./start_brain_analyzer.sh"
echo "   3. Open browser to: http://localhost:8050"
echo ""
echo "🔧 Manual Start:"
echo "   source brain_analysis_env/bin/activate"
echo "   python3 modern_brain_explorer.py"
echo ""
echo "📚 Documentation: See README.md for detailed usage"
echo "🐛 Issues: Report at GitHub repository"
echo ""
echo "✨ Features Ready:"
echo "   • CAM ICU statistical analysis"
echo "   • Interactive brain visualization"
echo "   • Z-score and P-value thresholding"
echo "   • Data export with coordinates"
echo "   • ROI-based statistical summaries"

deactivate 2>/dev/null || true  # Deactivate if was activated 