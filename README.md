# Brain Source Localization Analysis Platform

## Overview

A comprehensive web-based platform for analyzing EEG/MEG source localization data with advanced statistical analysis capabilities. Specifically designed for **CAM ICU (Confusion Assessment Method for ICU)** studies examining the effects of different auditory interventions on brain activity in critically ill patients.

## 🧠 Scientific Context

This platform analyzes brain source activity data from ICU patients experiencing delirium, comparing the neurological effects of:
- **Live Music** therapy
- **Recorded Music** therapy  
- **Pink Noise** control
- **No Music** baseline

Using advanced source localization techniques (LORETA), the platform reveals how different auditory interventions affect brain activity patterns in vulnerable ICU populations.

## ✨ Key Features

### 🔬 Advanced Statistical Analysis
- **Z-score thresholding** with customizable significance levels
- **P-value analysis** with multiple comparison correction (FDR)
- **Group-level statistics** across multiple participants
- **ROI-based statistical summaries** by brain region

### 🎨 Interactive Visualization
- **Real-time brain surface plots** with PyVista/MNE integration
- **Hemispheric comparisons** (left/right brain analysis)
- **Custom threshold controls** (percentile, z-score, p-value based)
- **Statistical overlay maps** showing significant activations

### 📊 Data Export & Analysis
- **CSV/JSON export** with 3D coordinates and statistics
- **Snapshot gallery** for capturing analysis states
- **ROI summary statistics** by anatomical region
- **Automated report generation** with brain images

### 🚀 Performance & Usability
- **Pre-generated brain images** for fast loading
- **Custom parameter regeneration** with dynamic restarts
- **Group vs individual analysis** detection
- **Data quality validation** (identical condition detection)

## 🛠️ Technical Architecture

### Core Technologies
- **MNE-Python**: Neurophysiological data processing
- **PyVista**: 3D brain surface rendering
- **Dash/Plotly**: Interactive web interface
- **NumPy/SciPy**: Statistical computations
- **Pandas**: Data manipulation and export

### Processing Pipeline
1. **Source Estimate Loading**: Automated detection of LORETA .stc files
2. **Statistical Computation**: Z-scores, p-values, FDR correction
3. **Brain Visualization**: PyVista-based surface rendering
4. **Export Generation**: Coordinate-based data export

## 📥 Installation

### Prerequisites
```bash
# Required Python packages
pip install mne dash plotly numpy scipy pandas dash-bootstrap-components
pip install pyvista matplotlib pillow statsmodels
```

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd brain-source-analysis

# Install dependencies
pip install -r requirements.txt

# Run the application
python modern_brain_explorer.py
```

### Access the Interface
Open your browser to: `http://localhost:8050`

## 🎯 Usage Examples

### Basic CAM ICU Analysis
```python
# Load and compare Live Music vs No Music
explorer = ModernBrainExplorer()
explorer.run()

# Select conditions in web interface:
# Primary: ICU_CAM_ICU_Negative_Live_Music
# Comparison: ICU_CAM_ICU_Negative_No_Music
```

### Statistical Thresholding
```python
# Z-score analysis (|z| >= 1.96, p < 0.05)
# Set threshold method to 'z_score'
# Apply FDR correction for multiple comparisons
```

### Data Export
```python
# Export with coordinates and statistics
# Format: CSV or JSON
# Includes: 3D brain coordinates, z-scores, p-values, ROI labels
```

## 📈 Analysis Capabilities

### Supported Comparisons
- **Live Music vs No Music**: Therapeutic music intervention effects
- **Recorded Music vs No Music**: Pre-recorded audio intervention
- **Pink Noise vs No Music**: Control audio stimulus comparison
- **Cross-intervention comparisons**: Direct therapy comparisons

### Statistical Methods
- **One-sample t-tests**: Group-level significance testing
- **Multiple comparison correction**: FDR (Benjamini-Hochberg)
- **Effect size computation**: Cohen's d for clinical significance
- **ROI-based analysis**: Anatomical region summaries

### Export Formats
- **Research CSV**: Vertex-level data with coordinates
- **Statistical JSON**: Complete analysis metadata
- **Brain Images**: High-resolution PNG exports
- **Summary Reports**: Automated PDF generation

## 🏥 Clinical Applications

### ICU Delirium Research
- **Baseline vs Intervention**: Quantify therapeutic effects
- **Dose-Response Analysis**: Compare intervention intensities
- **Individual vs Group**: Population vs personalized analysis

### Neurological Assessment
- **Hemispheric Balance**: Left/right brain activity patterns
- **Regional Activation**: Frontal, temporal, parietal responses
- **Statistical Significance**: FDR-corrected activation maps

## 🔧 Advanced Configuration

### Custom Brain Images
```python
# Regenerate with custom thresholds
explorer.generate_custom_images(
    condition1="ICU_CAM_ICU_Negative_Live_Music",
    condition2="ICU_CAM_ICU_Negative_No_Music", 
    threshold_method="z_score",
    z_threshold=2.58  # p < 0.01
)
```

### Batch Processing
```python
# Process multiple comparisons
comparisons = [
    ("Live_Music", "No_Music"),
    ("Recorded_Music", "No_Music"),
    ("Pink_Noise", "No_Music")
]

for cond1, cond2 in comparisons:
    explorer.export_comparison(cond1, cond2)
```

## 📊 Data Structure

### Input Requirements
```
music_cache/icu/
├── CAM_ICU_Negative_Live_Music_mean_stc-lh.stc
├── CAM_ICU_Negative_Live_Music_mean_stc-rh.stc
├── CAM_ICU_Negative_No_Music_mean_stc-lh.stc
├── CAM_ICU_Negative_No_Music_mean_stc-rh.stc
└── ... (additional conditions)
```

### Output Structure
```
brain_images/
├── ICU_CAM_ICU_Negative_Live_Music_lh.png
├── ICU_CAM_ICU_Negative_Live_Music_rh.png
├── Live_Music_minus_No_Music_lh.png
└── Live_Music_minus_No_Music_rh.png

exported_data/
├── brain_comparison_Live_vs_NoMusic_20240101.csv
└── statistical_analysis_summary.json
```

## 🚀 Performance Features

### Optimizations
- **Pre-computed brain images**: Sub-second visualization loading
- **Intelligent caching**: Automatic .stc file detection and loading
- **Memory management**: Efficient NumPy array handling
- **Parallel processing**: Multi-threaded image generation

### Scalability
- **Group analysis**: Handle 3-50+ participants automatically
- **Large datasets**: Process 10,000+ brain vertices efficiently  
- **Cross-platform**: macOS, Linux, Windows support
- **Headless rendering**: Server deployment capability

## 🔬 Research Applications

### Published Studies
- ICU delirium intervention effectiveness
- Music therapy neurological mechanisms
- Critical care cognitive assessment

### Ongoing Research
- Personalized intervention selection
- Real-time delirium monitoring
- Multi-modal sensory interventions

## 🛣️ Development Roadmap

### Immediate (Next Release)
- [ ] Automated PDF report generation
- [ ] Command-line batch processing interface
- [ ] Docker containerization
- [ ] Cloud deployment scripts

### Near-term (3-6 months)
- [ ] Real-time EEG processing pipeline
- [ ] Machine learning outcome prediction
- [ ] Multi-site collaborative analysis
- [ ] Mobile-responsive interface

### Long-term (6-12 months)
- [ ] Integration with electronic health records
- [ ] Predictive delirium risk modeling
- [ ] Multi-modal data fusion (EEG + clinical)
- [ ] Regulatory compliance (HIPAA, FDA)

## 🤝 Contributing

### Development Setup
```bash
git clone <repository-url>
cd brain-source-analysis
pip install -e .
pre-commit install
```

### Code Style
- Black formatting
- Type hints required
- Comprehensive docstrings
- Unit test coverage >80%

## 📄 License

MIT License - see LICENSE file for details.

## 📚 Citation

If you use this platform in your research, please cite:
```bibtex
@software{brain_source_platform,
  title={Brain Source Localization Analysis Platform},
  author={Research Team},
  year={2024},
  url={https://github.com/your-org/brain-source-analysis}
}
```

## 🔗 Related Resources

- [MNE-Python Documentation](https://mne.tools/)
- [LORETA Source Localization](http://www.uzh.ch/keyinst/loreta.htm)

## 💡 Support

- **Technical Issues**: Open GitHub issue
- **Research Questions**: Contact research team
- **Feature Requests**: Submit enhancement proposal
- **Clinical Applications**: Consult clinical team

---

**Last Updated**: 2024-12-19  
**Version**: 2.0.0  
**Status**: Production Ready ✅ 
