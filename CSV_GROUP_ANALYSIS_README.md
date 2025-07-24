# CSV-Based Group Analysis System

## Overview
The brain analysis system now supports **CSV-based participant grouping** for multi-participant group-level analyses. This allows flexible grouping of participants by clinical criteria (e.g., CAM-ICU status) for powerful statistical comparisons.

## CSV Format

### Required Columns
- `participant_id`: Unique identifier for each participant
- `group`: Group classification (e.g., "CAM_ICU_Negative", "CAM_ICU_Positive")
- `condition`: Experimental condition (e.g., "Live_Music", "No_Music")
- `filename`: EEG file name (e.g., "MUS-101-LM.edf")
- `dataset`: Dataset type ("ICU" or "Outpatient")

### Example CSV (`participant_mapping.csv`)
```csv
participant_id,group,condition,filename,dataset
MUS-101,CAM_ICU_Negative,Live_Music,MUS-101-LM.edf,ICU
MUS-101,CAM_ICU_Negative,No_Music,MUS-101-NM.edf,ICU
MUS-103,CAM_ICU_Negative,Live_Music,MUS-103-LM.edf,ICU
MUS-103,CAM_ICU_Negative,No_Music,MUS-103-NM.edf,ICU
MUS-106,CAM_ICU_Negative,Live_Music,MUS-106-LM.edf,ICU
MUS-106,CAM_ICU_Negative,No_Music,MUS-106-NM.edf,ICU
```

## Usage

### 1. Create CSV File
Create `participant_mapping.csv` in the project root with your participant groupings.

### 2. Run Processing Pipeline
```bash
python demo_music_datasets.py
```

The system will:
- Load individual EEG files
- Process each participant's data individually
- Aggregate participants within each group
- Create group-level source estimates
- Generate key comparisons (e.g., Live Music vs No Music for CAM-ICU Negative)

### 3. Launch Web Interface
```bash
python modern_brain_explorer.py
```

The web interface will show:
- 🎯 **Group-level conditions** (averaged across participants)
- 👤 **Individual participant conditions**
- **Demo section** with key comparisons for team meetings
- **Group vs Individual** indicators in statistics

## Demo-Ready Comparisons

For your team meeting, the system generates these key comparisons:
1. **CAM-ICU Negative: Live Music vs No Music**
2. **CAM-ICU Negative: Recorded Music vs No Music**
3. **CAM-ICU Negative: Pink Noise vs No Music**

Each comparison aggregates across all participants in the CAM-ICU Negative group.

## Key Features

### Group-Level Analysis
- **Automatic aggregation** of participants within groups
- **Statistical power** through multi-participant averaging
- **Flexible grouping** by any clinical criteria

### Web Interface Enhancements
- **Visual indicators** for group vs individual data
- **Quick demo buttons** for key comparisons
- **Participant count** information in statistics
- **Prioritized display** of group-level conditions

### Backwards Compatibility
- **Fallback mode** if no CSV file is found
- **Maintains original** single-participant functionality
- **Same file structure** for cached results

## Clinical Applications

### CAM-ICU Grouping
- Compare **CAM-ICU Negative** vs **CAM-ICU Positive** participants
- Examine **music therapy effects** in different cognitive states
- Control for **delirium status** in analyses

### Other Grouping Possibilities
- **Age groups** (Young vs Old)
- **Gender** (Male vs Female)
- **Severity** (Mild vs Moderate vs Severe)
- **Time points** (Pre vs Post intervention)

## Files Generated

### Individual Level
- `music_cache/icu/MUS-101_Live_Music_stc.fif`
- `music_cache/icu/MUS-103_Live_Music_stc.fif`

### Group Level
- `music_cache/icu/CAM_ICU_Negative_Live_Music_stc.fif`
- `music_cache/icu/CAM_ICU_Negative_No_Music_stc.fif`

### Visualizations
- `music_results/comparisons/CAM_ICU_Negative_Live_Music_minus_CAM_ICU_Negative_No_Music_lh.png`
- `music_results/comparisons/CAM_ICU_Negative_Live_Music_minus_CAM_ICU_Negative_No_Music_rh.png`

## Status
✅ **Ready for Demo**: System implemented and tested with sample data
⏳ **Pending**: Addition of MUS-103 and MUS-106 EEG files for full demo

## Next Steps
1. Add MUS-103 and MUS-106 EEG files to `data/icu/` directory
2. Run processing pipeline to generate group-level results
3. Demo the key comparisons at team meeting
4. Expand to additional participant groups as needed 