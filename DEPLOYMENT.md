# Deployment Guide
## Brain Source Localization Analysis Platform

This guide provides multiple deployment options for the Brain Source Localization Analysis Platform, from local development to production cloud deployment.

## 🚀 Quick Start Options

### Option 1: One-Click Local Setup
```bash
# Clone the repository
git clone <repository-url>
cd brain-source-analysis

# Run the automated installer
./install.sh

# Start the application
./start_brain_analyzer.sh
```

### Option 2: Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at: http://localhost:8050
```

### Option 3: Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run directly
python modern_brain_explorer.py
```

## 📋 Deployment Options

### 1. Local Development
**Best for**: Research, development, testing
```bash
./install.sh
./start_brain_analyzer.sh
```
- ✅ Easy setup with automated installer
- ✅ Virtual environment isolation
- ✅ Quick iteration and testing
- ⚠️ Single user access only

### 2. Shared Research Server
**Best for**: Small team collaboration, lab deployments
```bash
# Install on shared server
./install.sh

# Start with custom host/port
python cli.py web --host 0.0.0.0 --port 8050

# Or use Docker for isolation
docker-compose up
```
- ✅ Multi-user access
- ✅ Centralized data processing
- ✅ Resource sharing
- ⚠️ Requires server administration

### 3. Cloud Production Deployment
**Best for**: Multi-site studies, production research
```bash
# AWS/GCP/Azure deployment
docker-compose --profile production up

# With reverse proxy and SSL
# Customize nginx.conf for your domain
```
- ✅ High availability
- ✅ Scalable resources
- ✅ Professional SSL/security
- ⚠️ Requires cloud expertise

### 4. Container Orchestration
**Best for**: Large-scale, enterprise deployments
```bash
# Kubernetes deployment
kubectl apply -f k8s/

# Or Docker Swarm
docker stack deploy -c docker-compose.yml brain-analysis
```
- ✅ Auto-scaling
- ✅ Load balancing
- ✅ High availability
- ⚠️ Complex setup required

## 🛠️ Command Line Interface

The platform includes a comprehensive CLI for automation:

```bash
# List all available conditions
python cli.py list

# Run specific analysis
python cli.py analyze "ICU_CAM_ICU_Negative_Live_Music" "ICU_CAM_ICU_Negative_No_Music" -o results.json

# Batch export all comparisons
python cli.py batch-export --format csv --output-dir exports/

# Generate brain images
python cli.py generate-images --force

# Start web interface
python cli.py web --host 0.0.0.0 --port 8080
```

## 🔧 Configuration

### Environment Variables
```bash
# Application settings
export BRAIN_ANALYZER_PORT=8050
export BRAIN_ANALYZER_HOST=0.0.0.0
export BRAIN_ANALYZER_DEBUG=false

# Rendering settings
export PYVISTA_OFF_SCREEN=true
export MNE_3D_BACKEND=pyvista
export DISPLAY=:99
```

### Custom Configuration
Create a `config/settings.json`:
```json
{
  "default_statistical_method": "z_score",
  "default_threshold": 1.96,
  "max_upload_size": "500MB",
  "cache_enabled": true,
  "auto_generate_images": true
}
```

## 📊 Data Management

### Input Data Structure
```
music_cache/icu/
├── CAM_ICU_Negative_Live_Music_mean_stc-lh.stc
├── CAM_ICU_Negative_Live_Music_mean_stc-rh.stc
├── CAM_ICU_Negative_No_Music_mean_stc-lh.stc
└── ... (additional conditions)
```

### Output Management
```
brain_images/          # Pre-generated visualizations
custom_brain_images/   # Custom parameter images
exported_data/         # CSV/JSON exports
analysis_results/      # Batch processing results
```

## 🔒 Security Considerations

### Local/Lab Deployment
- Default setup uses localhost only
- No authentication required
- Suitable for trusted networks

### Production Deployment
```nginx
# nginx.conf example
server {
    listen 443 ssl;
    server_name brain-analysis.your-domain.com;
    
    ssl_certificate /etc/nginx/certs/cert.pem;
    ssl_certificate_key /etc/nginx/certs/key.pem;
    
    location / {
        proxy_pass http://brain-analyzer:8050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### HIPAA Compliance (if needed)
- Use encrypted storage for patient data
- Enable audit logging
- Implement user authentication
- Use VPN or private networks

## 📈 Performance Optimization

### Hardware Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 16GB RAM, 8 CPU cores
- **Storage**: 10GB+ for brain images and data

### Scaling Options
```yaml
# docker-compose.yml scaling
services:
  brain-analyzer:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
```

### Caching Strategy
- Pre-generate brain images: `python cli.py generate-images`
- Use SSD storage for cache directories
- Monitor memory usage with large datasets

## 🚨 Troubleshooting

### Common Issues

**PyVista rendering errors on macOS:**
```bash
export PYVISTA_OFF_SCREEN=true
# Or use the Docker deployment
```

**Port already in use:**
```bash
python cli.py web --port 8051
# Or: docker-compose down && docker-compose up
```

**Memory issues with large datasets:**
```bash
# Increase Docker memory limits
docker-compose --compatibility up
```

### Health Monitoring
```bash
# Docker health check
docker-compose ps

# Application logs
docker-compose logs brain-analyzer

# Resource usage
docker stats brain-source-analysis
```

## 📞 Support & Maintenance

### Regular Maintenance
```bash
# Update brain images monthly
python cli.py generate-images --force

# Clean old exports
find exported_data/ -name "*.csv" -mtime +30 -delete

# Update dependencies
pip install -r requirements.txt --upgrade
```

### Backup Strategy
```bash
# Critical directories to backup:
tar -czf backup_$(date +%Y%m%d).tar.gz \
    music_cache/ \
    brain_images/ \
    exported_data/ \
    config/
```

### Monitoring
- Monitor disk space (brain images can be large)
- Track memory usage during batch processing
- Log analysis completion times
- Monitor web application uptime

## 🎯 Production Checklist

- [ ] Data directories properly mounted
- [ ] SSL certificates configured (if public)
- [ ] Resource limits set appropriately
- [ ] Health checks enabled
- [ ] Backup strategy implemented
- [ ] Monitoring/logging configured
- [ ] Documentation updated for users
- [ ] Test all analysis workflows
- [ ] Performance benchmarking completed

---

**Last Updated**: 2024-12-19  
**Version**: 2.0.0  
**Status**: Production Ready ✅ 