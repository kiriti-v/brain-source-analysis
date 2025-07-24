# Brain Source Localization Analysis Platform - Docker Configuration
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p brain_images custom_brain_images exported_data music_cache/icu

# Set environment variables for headless rendering
ENV DISPLAY=:99
ENV PYVISTA_OFF_SCREEN=true
ENV MNE_3D_BACKEND=pyvista

# Expose port
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8050/ || exit 1

# Create startup script
RUN echo '#!/bin/bash\n\
echo "🧠 Starting Brain Source Localization Analysis Platform (Docker)"\n\
\n\
# Start Xvfb for headless rendering\n\
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &\n\
\n\
# Wait for Xvfb to start\n\
sleep 2\n\
\n\
# Check for data files\n\
if [ -d "music_cache/icu" ] && [ "$(ls -A music_cache/icu)" ]; then\n\
    stc_count=$(find music_cache/icu -name "*.stc" | wc -l)\n\
    echo "✅ Found $stc_count .stc data files"\n\
else\n\
    echo "⚠️  No data files found in music_cache/icu/"\n\
    echo "   Mount your data volume: -v /path/to/data:/app/music_cache/icu"\n\
fi\n\
\n\
# Start the application\n\
python3 modern_brain_explorer.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"] 