#!/usr/bin/env python3
"""
Comparative Music EEG Analysis Pipeline
Processes ICU and Outpatient music datasets for sLORETA source localization
Now supports CSV-based participant grouping for multi-participant analyses
"""

import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import time
from scipy import stats
import pandas as pd  # Added for CSV reading

# Configure MNE to use a non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set up directories
base_dir = os.getcwd()
data_dir = os.path.join(base_dir, 'data')
cache_dir = os.path.join(base_dir, 'music_cache')
results_dir = os.path.join(base_dir, 'music_results')

# Create directories
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(cache_dir, 'icu'), exist_ok=True)
os.makedirs(os.path.join(cache_dir, 'outpatient'), exist_ok=True)
os.makedirs(os.path.join(results_dir, 'icu'), exist_ok=True)
os.makedirs(os.path.join(results_dir, 'outpatient'), exist_ok=True)
os.makedirs(os.path.join(results_dir, 'comparisons'), exist_ok=True)

# Load MNE sample dataset for anatomy
from mne.datasets import sample
data_path = sample.data_path()
subject = 'sample'
subjects_dir = os.path.join(data_path, 'subjects')

# Paths to anatomy files
bem_fname = os.path.join(data_path, 'subjects', 'sample', 'bem', 'sample-5120-5120-5120-bem-sol.fif')
src_fname = os.path.join(data_path, 'subjects', 'sample', 'bem', 'sample-oct-6-src.fif')
trans_fname = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif')

# File definitions
ICU_FILES = {
    'Live_Music': 'MUS-101-LM.edf',
    'Recorded_Music': 'MUS-101-RM.edf', 
    'Pink_Noise': 'MUS-101-PN.edf',
    'No_Music': 'MUS-101-NM.edf'
}

OUTPATIENT_FILES = {
    'AMH_Eyes_Closed': 'MOP-101-AMH-EC.edf',
    'AMH_Eyes_Open': 'MOP-101-AMH-EO.edf',
    'AML_Eyes_Closed': 'MOP-101-AML-EC.edf', 
    'AML_Eyes_Open': 'MOP-101-AML-EO.edf'
}

# CSV-based participant mapping functionality
def load_participant_mapping(csv_file='participant_mapping.csv'):
    """Load participant mapping from CSV file."""
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} not found. Using hardcoded file mappings.")
        return None
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded participant mapping from {csv_file}")
        print(f"Found {len(df)} participant-condition combinations")
        print(f"Groups: {df['group'].unique()}")
        print(f"Conditions: {df['condition'].unique()}")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def get_files_from_csv(participant_df, dataset_type='ICU'):
    """Extract file mappings from participant dataframe."""
    if participant_df is None:
        return {}
    
    # Filter by dataset type
    dataset_df = participant_df[participant_df['dataset'] == dataset_type]
    
    if dataset_df.empty:
        return {}
    
    # Create individual participant-condition mappings
    individual_files = {}
    for _, row in dataset_df.iterrows():
        key = f"{row['participant_id']}_{row['condition']}"
        individual_files[key] = row['filename']
    
    return individual_files

def get_group_mappings(participant_df, dataset_type='ICU'):
    """Get group-level mappings from participant dataframe."""
    if participant_df is None:
        return {}
    
    # Filter by dataset type
    dataset_df = participant_df[participant_df['dataset'] == dataset_type]
    
    if dataset_df.empty:
        return {}
    
    # Group by group and condition
    group_mappings = {}
    for (group, condition), group_data in dataset_df.groupby(['group', 'condition']):
        group_key = f"{group}_{condition}"
        group_mappings[group_key] = {
            'participants': group_data['participant_id'].tolist(),
            'files': group_data['filename'].tolist(),
            'group': group,
            'condition': condition
        }
    
    return group_mappings

def aggregate_source_estimates(stc_list, method='mean'):
    """Aggregate multiple source estimates across participants."""
    if not stc_list:
        return None
    
    if len(stc_list) == 1:
        return stc_list[0]
    
    # Ensure all STCs have the same structure
    vertices = stc_list[0].vertices
    n_vertices = stc_list[0].data.shape[0]
    n_times = stc_list[0].data.shape[1]
    
    # Stack all data
    all_data = np.zeros((len(stc_list), n_vertices, n_times))
    for i, stc in enumerate(stc_list):
        all_data[i] = stc.data
    
    # Aggregate
    if method == 'mean':
        aggregated_data = np.mean(all_data, axis=0)
    elif method == 'median':
        aggregated_data = np.median(all_data, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    # Create new source estimate
    aggregated_stc = mne.SourceEstimate(
        aggregated_data,
        vertices=vertices,
        tmin=stc_list[0].tmin,
        tstep=stc_list[0].tstep,
        subject=stc_list[0].subject
    )
    
    return aggregated_stc

def load_and_preprocess_edf(filepath, condition_name, dataset_type):
    """Load and preprocess EDF file with caching."""
    cache_file = os.path.join(cache_dir, dataset_type, f"{condition_name}_preprocessed.fif")
    
    if os.path.exists(cache_file):
        print(f"Loading cached {condition_name} from {cache_file}")
        return mne.io.read_raw_fif(cache_file, preload=True)
    
    print(f"Processing {condition_name} from {filepath}")
    start_time = time.time()
    
    # Load EDF file
    raw = mne.io.read_raw_edf(filepath, preload=True)
    
    # Print available channels for debugging
    print(f"Available channels: {raw.ch_names[:20]}...")  # Show first 20
    
    # Define standard EEG channel names we want to keep
    standard_eeg_channels = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'Oz',
        'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'A1', 'A2', 'T1', 'T2',
        'AF3', 'AF4', 'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2',
        'CP5', 'CP6', 'PO3', 'PO4', 'PO7', 'PO8', 'Fpz', 'P7', 'P8'
    ]
    
    # Find which standard EEG channels are actually present
    available_eeg = [ch for ch in raw.ch_names if any(std_ch.upper() == ch.upper() for std_ch in standard_eeg_channels)]
    
    if not available_eeg:
        # If no standard names found, try to pick channels that look like EEG
        # Look for channels that don't contain common non-EEG identifiers
        non_eeg_keywords = ['ECG', 'EOG', 'EMG', 'CHIN', 'LEG', 'VENT', 'DIF', 'DC', 'TRIG', 
                           'Flow', 'Pressure', 'Snore', 'Chest', 'Abdomen', 'PPG', 'SpO2', 
                           'LOC', 'ROC', 'X', 'Phase', 'RMI', 'RR', 'Position', 'Elevation', 
                           'Activity', 'Pleth', 'PTT', 'PR', 'PulseQuality']
        
        available_eeg = [ch for ch in raw.ch_names 
                        if not any(keyword in ch.upper() for keyword in non_eeg_keywords)]
        
        print(f"Found {len(available_eeg)} potential EEG channels: {available_eeg[:10]}...")
    
    if not available_eeg:
        raise ValueError(f"No EEG channels found in {filepath}")
    
    # Pick only the EEG channels we found
    raw.pick_channels(available_eeg)
    
    # Set channel types to EEG
    raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
    
    # Set montage and ensure all channels have locations
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False, on_missing='ignore')
        print(f"Set montage for {len(raw.info['chs'])} channels")
        
        # Check which channels have locations
        channels_with_locs = []
        for ch in raw.info['chs']:
            # Check if location is not NaN and not all zeros
            if not np.any(np.isnan(ch['loc'][:3])) and not np.allclose(ch['loc'][:3], 0):
                channels_with_locs.append(ch['ch_name'])
        
        print(f"Channels with locations: {len(channels_with_locs)}/{len(raw.ch_names)}")
        print(f"Channels with valid locations: {channels_with_locs}")
        
        # If some channels don't have locations, remove them
        if len(channels_with_locs) < len(raw.ch_names):
            print(f"Removing channels without locations...")
            raw.pick_channels(channels_with_locs)
            print(f"Kept {len(raw.ch_names)} channels with valid locations")
        
        # Ensure we have enough channels for source localization
        if len(raw.ch_names) < 10:
            raise ValueError(f"Too few EEG channels with locations: {len(raw.ch_names)}")
            
    except Exception as e:
        print(f"Could not set montage: {e}")
        raise ValueError("Cannot proceed without channel locations for source localization")
    
    # Set EEG reference
    raw.set_eeg_reference(projection=True)
    
    # Filter to alpha band (8-12 Hz) for comparison with sample dataset
    raw.filter(8., 12., fir_design='firwin')
    
    # Take a reasonable segment (30-60 seconds)
    if raw.times[-1] > 60:
        raw.crop(tmin=10, tmax=70)  # Skip first 10s, take 60s
    else:
        raw.crop(tmin=2)  # Skip first 2s, take rest
    
    # Save preprocessed data
    raw.save(cache_file, overwrite=True)
    print(f"Preprocessing {condition_name} took {time.time() - start_time:.1f}s")
    print(f"Final data shape: {raw._data.shape} ({len(raw.ch_names)} channels, {len(raw.times)} time points)")
    
    return raw

def compute_source_estimate(raw, condition_name, dataset_type):
    """Compute source estimate with caching and robust file checking."""
    cache_base = os.path.join(cache_dir, dataset_type, condition_name)
    stc_cache = f"{cache_base}_stc"
    stc_files = [f"{stc_cache}-lh.stc", f"{stc_cache}-rh.stc"]
    
    # Enhanced check - verify both files exist and are not empty
    if all(os.path.exists(f) and os.path.getsize(f) > 1000 for f in stc_files):  # Minimum 1KB size
        print(f"✅ Loading cached source estimate for {condition_name}")
        try:
            stc = mne.read_source_estimate(stc_cache)
            print(f"   Cached STC shape: {stc.data.shape}")
            return stc
        except Exception as e:
            print(f"⚠️  Error loading cached STC for {condition_name}: {e}")
            print(f"   Will recompute...")
            # Remove corrupted files
            for f in stc_files:
                if os.path.exists(f):
                    os.remove(f)
    
    print(f"🔄 Computing source estimate for {condition_name}")
    start_time = time.time()
    
    # Compute noise covariance
    noise_cov_cache = f"{cache_base}_noise_cov.pkl"
    if os.path.exists(noise_cov_cache):
        print(f"   Loading cached noise covariance...")
        with open(noise_cov_cache, 'rb') as f:
            noise_cov = pickle.load(f)
    else:
        print(f"   Computing noise covariance...")
        noise_cov = mne.compute_raw_covariance(raw)
        with open(noise_cov_cache, 'wb') as f:
            pickle.dump(noise_cov, f)
    
    # Compute forward solution
    fwd_cache = f"{cache_base}_fwd.fif"
    if os.path.exists(fwd_cache):
        print(f"   Loading cached forward solution...")
        fwd = mne.read_forward_solution(fwd_cache)
    else:
        print(f"   Computing forward solution...")
        fwd = mne.make_forward_solution(raw.info, trans=trans_fname, src=src_fname,
                                        bem=bem_fname, meg=False, eeg=True)
        mne.write_forward_solution(fwd_cache, fwd, overwrite=True)
    
    # Create inverse operator
    inv_cache = f"{cache_base}_inv.fif"
    if os.path.exists(inv_cache):
        print(f"   Loading cached inverse operator...")
        inverse_operator = mne.minimum_norm.read_inverse_operator(inv_cache)
    else:
        print(f"   Computing inverse operator...")
        inverse_operator = mne.minimum_norm.make_inverse_operator(
            raw.info, fwd, noise_cov, loose=0.2, depth=0.8)
        mne.minimum_norm.write_inverse_operator(inv_cache, inverse_operator)
    
    # Apply sLORETA (the memory-intensive step)
    print(f"   Applying sLORETA method...")
    try:
    stc = mne.minimum_norm.apply_inverse_raw(
        raw, inverse_operator, lambda2=1. / 9., method='sLORETA')
    
        # Save source estimate immediately
        print(f"   Saving source estimate...")
    stc.save(stc_cache, overwrite=True)
        
        # Clean up some memory
        del inverse_operator, fwd
        
        print(f"✅ Source localization for {condition_name} took {time.time() - start_time:.1f}s")
        print(f"   Final STC shape: {stc.data.shape}")
    
    return stc
        
    except Exception as e:
        print(f"❌ Error during source localization for {condition_name}: {e}")
        # Clean up any partial files
        for f in stc_files:
            if os.path.exists(f):
                os.remove(f)
        raise e

def compute_mean_power_stc(stc):
    """Compute mean alpha power across time."""
    stc_power = np.abs(stc.data) ** 2
    stc_mean_power = np.mean(stc_power, axis=1)
    
    return mne.SourceEstimate(
        stc_mean_power,
        vertices=stc.vertices,
        tmin=0,
        tstep=1,
        subject=subject
    )

def create_comparison_visualization(stc1, stc2, name1, name2, output_dir):
    """Create comparison visualizations between two conditions."""
    print(f"Creating comparison: {name1} vs {name2}")
    
    # Compute difference
    stc_diff = stc1.copy()
    stc_diff.data = stc1.data - stc2.data
    
    # Create visualizations for each condition and difference
    conditions = [(stc1, name1), (stc2, name2), (stc_diff, f"{name1}_minus_{name2}")]
    
    for stc, name in conditions:
        for hemi in ['lh', 'rh']:
            try:
                # Create extra large figure to give brain maximum room
                fig, ax = plt.subplots(figsize=(16, 14))
                
                # Give brain maximum breathing room - minimal margins
                plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
                
                # Plot brain visualization with maximum space parameters
                brain = mne.viz.plot_source_estimates(
                    stc, 
                    subject=subject, 
                    subjects_dir=subjects_dir,
                    hemi=hemi,
                    views='lat',
                    backend='matplotlib',
                    time_viewer=False,
                    figure=fig,
                    title=f"{name} - {hemi.upper()}\nAlpha Power ((nA/mm²)²)",
                    size=(1600, 1400),  # Very large size to prevent any cropping
                    background='white',
                    colorbar=True  # Include colorbar for value interpretation
                )
                
                # Expand axis limits to show complete brain with large margins
                current_xlim = ax.get_xlim()
                current_ylim = ax.get_ylim()
                
                # Add 25% margin on all sides for PDF quality
                x_range = current_xlim[1] - current_xlim[0]
                y_range = current_ylim[1] - current_ylim[0]
                
                ax.set_xlim(current_xlim[0] - 0.25 * x_range, current_xlim[1] + 0.25 * x_range)
                ax.set_ylim(current_ylim[0] - 0.25 * y_range, current_ylim[1] + 0.25 * y_range)
                
                # Save visualization with NO cropping
                viz_file = os.path.join(output_dir, f"{name}_{hemi}.png")
                fig.savefig(viz_file, dpi=300, 
                           bbox_inches=None,  # No cropping at all!
                           facecolor='white')
                plt.close(fig)
                print(f"Saved {name} {hemi} to: {viz_file}")
                
            except Exception as e:
                print(f"Visualization failed for {name} {hemi}: {e}")
                # Create fallback heatmap with maximum room
                fig, ax = plt.subplots(figsize=(16, 12))
                
                # Maximum breathing room for fallback too
                plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
                
                if hemi == 'lh':
                    data = stc.data[:len(stc.vertices[0])]
                else:
                    data = stc.data[len(stc.vertices[0]):]
                
                im = ax.imshow(data.reshape(-1, 1), aspect='auto', cmap='viridis')
                
                # Add colorbar with proper formatting and visible tick labels
                cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.02)
                cbar.set_label('Alpha Power ((nA/mm²)²)', fontsize=12, rotation=270, labelpad=25, weight='bold')
                
                # Set explicit tick values for better visibility
                data_min, data_max = np.min(data), np.max(data)
                ticks = np.linspace(data_min, data_max, 5)  # 5 evenly spaced ticks
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f'{tick:.3f}' for tick in ticks])
                cbar.ax.tick_params(labelsize=10, colors='black', width=1.5)
                
                # Ensure colorbar text is visible
                cbar.ax.yaxis.label.set_color('black')
                cbar.ax.yaxis.label.set_fontweight('bold')
                
                ax.set_title(f'{name} - {hemi.upper()} (Heatmap)', pad=30)
                
                # Add large margins around the heatmap
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                x_margin = (xlim[1] - xlim[0]) * 0.3
                y_margin = (ylim[1] - ylim[0]) * 0.3
                ax.set_xlim(xlim[0] - x_margin, xlim[1] + x_margin)
                ax.set_ylim(ylim[0] - y_margin, ylim[1] + y_margin)
                
                viz_file = os.path.join(output_dir, f"{name}_{hemi}_heatmap.png")
                fig.savefig(viz_file, dpi=300, 
                           bbox_inches=None,  # No cropping!
                           facecolor='white')
                plt.close(fig)

def analyze_datasets():
    """Main analysis function with CSV-based participant mapping and group analysis."""
    print("=== MUSIC EEG COMPARATIVE ANALYSIS WITH GROUP SUPPORT ===\n")
    
    # Load participant mapping from CSV
    participant_df = load_participant_mapping()
    
    # Store all results
    individual_results = {}
    group_results = {}
    
    if participant_df is not None:
        print("🎯 Using CSV-based participant mapping")
        
        # Process individual participants first
        print("\n📊 Processing Individual Participants...")
        
        for dataset_type in ['ICU', 'Outpatient']:
            print(f"\nProcessing {dataset_type} Dataset...")
            
            # Get individual file mappings
            individual_files = get_files_from_csv(participant_df, dataset_type)
            
            if not individual_files:
                print(f"No {dataset_type} files found in CSV")
                continue
            
            # Show progress info
            print(f"Found {len(individual_files)} participant-condition combinations")
            processed_count = 0
            
            # Process each individual participant-condition
            for participant_condition, filename in individual_files.items():
                filepath = os.path.join(data_dir, dataset_type.lower(), filename)
                
                if not os.path.exists(filepath):
                    print(f"⚠️  File not found: {filepath}")
                    continue
                
                print(f"\n[{processed_count + 1}/{len(individual_files)}] Processing: {participant_condition}")
                
                try:
                    # Load and preprocess
                    raw = load_and_preprocess_edf(filepath, participant_condition, dataset_type.lower())
                    
                    # Compute source estimate
                    stc = compute_source_estimate(raw, participant_condition, dataset_type.lower())
                    
                    # Compute mean power
                    stc_mean = compute_mean_power_stc(stc)
                    
                    individual_results[participant_condition] = {
                        'raw': raw,
                        'stc': stc,
                        'stc_mean': stc_mean,
                        'dataset': dataset_type,
                        'filename': filename
                    }
                    
                    processed_count += 1
                    print(f"✅ Processed: {participant_condition} ({processed_count}/{len(individual_files)} complete)")
                    
                except Exception as e:
                    print(f"❌ Error processing {participant_condition}: {e}")
                    print("   Continuing with next condition...")
            
            print(f"\n{dataset_type} Dataset Summary: {processed_count}/{len(individual_files)} conditions completed")
        
        # Process group-level aggregations
        print("\n🎯 Processing Group-Level Aggregations...")
        
        for dataset_type in ['ICU', 'Outpatient']:
            group_mappings = get_group_mappings(participant_df, dataset_type)
            
            if not group_mappings:
                print(f"No {dataset_type} groups found in CSV")
                continue
            
            for group_key, group_info in group_mappings.items():
                print(f"\nProcessing group: {group_key}")
                print(f"  Participants: {group_info['participants']}")
                
                # Collect individual STCs for this group
                individual_stcs = []
                individual_stc_means = []
                
                for participant_id in group_info['participants']:
                    participant_condition = f"{participant_id}_{group_info['condition']}"
                    
                    if participant_condition in individual_results:
                        individual_stcs.append(individual_results[participant_condition]['stc'])
                        individual_stc_means.append(individual_results[participant_condition]['stc_mean'])
                        print(f"  ✅ Added {participant_condition}")
                    else:
                        print(f"  ⚠️  Missing {participant_condition}")
                
                if individual_stcs:
                    # Aggregate across participants
                    group_stc = aggregate_source_estimates(individual_stcs, method='mean')
                    group_stc_mean = aggregate_source_estimates(individual_stc_means, method='mean')
                    
                    # Save group results
                    group_results[group_key] = {
                        'stc': group_stc,
                        'stc_mean': group_stc_mean,
                        'dataset': dataset_type,
                        'group': group_info['group'],
                        'condition': group_info['condition'],
                        'participants': group_info['participants'],
                        'n_participants': len(individual_stcs)
                    }
                    
                    # Save group-level source estimates to cache
                    cache_base = os.path.join(cache_dir, dataset_type.lower(), group_key)
                    group_stc.save(f"{cache_base}_stc", overwrite=True)
                    group_stc_mean.save(f"{cache_base}_mean_stc", overwrite=True)
                    
                    print(f"  ✅ Group {group_key} processed (n={len(individual_stcs)})")
                else:
                    print(f"  ❌ No valid participants for group {group_key}")
        
        # Create group-level comparisons
        print("\n🔍 Generating Group-Level Comparisons...")
        
        # Find CAM-ICU Negative comparisons
        cam_icu_negative_groups = {k: v for k, v in group_results.items() 
                                  if 'CAM_ICU_Negative' in k}
        
        if cam_icu_negative_groups:
            print(f"Found {len(cam_icu_negative_groups)} CAM-ICU Negative groups")
            
            # Generate key comparisons for demo
            comparisons = [
                ('CAM_ICU_Negative_Live_Music', 'CAM_ICU_Negative_No_Music'),
                ('CAM_ICU_Negative_Recorded_Music', 'CAM_ICU_Negative_No_Music'),
                ('CAM_ICU_Negative_Pink_Noise', 'CAM_ICU_Negative_No_Music')
            ]
            
            for group1, group2 in comparisons:
                if group1 in cam_icu_negative_groups and group2 in cam_icu_negative_groups:
                    print(f"Creating comparison: {group1} vs {group2}")
                    
                    create_comparison_visualization(
                        cam_icu_negative_groups[group1]['stc_mean'],
                        cam_icu_negative_groups[group2]['stc_mean'],
                        group1, group2,
                        os.path.join(results_dir, 'comparisons')
                    )
                    
                    print(f"  ✅ {group1} vs {group2} (n={cam_icu_negative_groups[group1]['n_participants']} vs n={cam_icu_negative_groups[group2]['n_participants']})")
        
        # Generate summary statistics
        print("\n📊 Generating Summary Statistics...")
        create_group_summary_report(individual_results, group_results, participant_df)
        
    else:
        print("🔄 Using hardcoded file mappings (legacy mode)")
        
        # Fallback to original hardcoded approach
    icu_results = {}
    outpatient_results = {}
    
    # Process ICU dataset
    print("Processing ICU Dataset...")
    for condition, filename in ICU_FILES.items():
        filepath = os.path.join(data_dir, 'icu', filename)
        
        # Load and preprocess
        raw = load_and_preprocess_edf(filepath, condition, 'icu')
        
        # Compute source estimate
        stc = compute_source_estimate(raw, condition, 'icu')
        
        # Compute mean power
        stc_mean = compute_mean_power_stc(stc)
        
        icu_results[condition] = {
            'raw': raw,
            'stc': stc,
            'stc_mean': stc_mean
        }
    
    # Process Outpatient dataset
    print("\nProcessing Outpatient Dataset...")
    for condition, filename in OUTPATIENT_FILES.items():
        filepath = os.path.join(data_dir, 'outpatient', filename)
        
        # Load and preprocess
        raw = load_and_preprocess_edf(filepath, condition, 'outpatient')
        
        # Compute source estimate
        stc = compute_source_estimate(raw, condition, 'outpatient')
        
        # Compute mean power
        stc_mean = compute_mean_power_stc(stc)
        
        outpatient_results[condition] = {
            'raw': raw,
            'stc': stc,
            'stc_mean': stc_mean
        }
    
    # Create meaningful comparisons
    print("\nGenerating Comparisons...")
    
    # ICU Comparisons: Music vs No Music
    print("ICU: Live Music vs No Music")
    create_comparison_visualization(
        icu_results['Live_Music']['stc_mean'],
        icu_results['No_Music']['stc_mean'],
        'ICU_Live_Music', 'ICU_No_Music',
        os.path.join(results_dir, 'comparisons')
    )
    
    print("ICU: Recorded Music vs Pink Noise")
    create_comparison_visualization(
        icu_results['Recorded_Music']['stc_mean'],
        icu_results['Pink_Noise']['stc_mean'],
        'ICU_Recorded_Music', 'ICU_Pink_Noise',
        os.path.join(results_dir, 'comparisons')
    )
    
    # Outpatient Comparisons: Eyes Closed vs Eyes Open
    print("Outpatient: AMH Eyes Closed vs Eyes Open")
    create_comparison_visualization(
        outpatient_results['AMH_Eyes_Closed']['stc_mean'],
        outpatient_results['AMH_Eyes_Open']['stc_mean'],
        'AMH_Eyes_Closed', 'AMH_Eyes_Open',
        os.path.join(results_dir, 'comparisons')
    )
    
    print("Outpatient: AML Eyes Closed vs Eyes Open")
    create_comparison_visualization(
        outpatient_results['AML_Eyes_Closed']['stc_mean'],
        outpatient_results['AML_Eyes_Open']['stc_mean'],
        'AML_Eyes_Closed', 'AML_Eyes_Open',
        os.path.join(results_dir, 'comparisons')
    )
    
    # Music Type Comparison (same attention state)
    print("Outpatient: AMH vs AML (Eyes Closed)")
    create_comparison_visualization(
        outpatient_results['AMH_Eyes_Closed']['stc_mean'],
        outpatient_results['AML_Eyes_Closed']['stc_mean'],
        'AMH_Eyes_Closed', 'AML_Eyes_Closed',
        os.path.join(results_dir, 'comparisons')
    )
    
    # Generate summary statistics
    print("\nGenerating Summary Statistics...")
    create_summary_report(icu_results, outpatient_results)
    
        individual_results = {**icu_results, **outpatient_results}
        group_results = {}
    
    return individual_results, group_results

def create_summary_report(icu_results, outpatient_results):
    """Create a summary report of findings."""
    report_file = os.path.join(results_dir, 'analysis_summary.txt')
    
    with open(report_file, 'w') as f:
        f.write("MUSIC EEG ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("DATASETS PROCESSED:\n")
        f.write(f"ICU Conditions: {list(icu_results.keys())}\n")
        f.write(f"Outpatient Conditions: {list(outpatient_results.keys())}\n\n")
        
        f.write("KEY COMPARISONS GENERATED:\n")
        f.write("1. ICU: Live Music vs No Music (therapeutic effect)\n")
        f.write("2. ICU: Recorded Music vs Pink Noise (music specificity)\n")
        f.write("3. Outpatient: Eyes Closed vs Eyes Open (alpha rhythm)\n")
        f.write("4. Outpatient: AMH vs AML (music tempo effects)\n\n")
        
        f.write("INTERPRETATION GUIDE:\n")
        f.write("- Positive differences (red/warm colors): Higher activity in first condition\n")
        f.write("- Negative differences (blue/cool colors): Higher activity in second condition\n")
        f.write("- Eyes Closed should show stronger occipital alpha (back of brain)\n")
        f.write("- Music conditions may show temporal/frontal differences\n\n")
        
        f.write("FILES GENERATED:\n")
        f.write("- Individual condition maps: music_results/comparisons/\n")
        f.write("- Difference maps: *_minus_* files show contrasts\n")
        f.write("- Both hemispheres (lh/rh) saved separately\n")
    
    print(f"Summary report saved to: {report_file}")

def create_group_summary_report(individual_results, group_results, participant_df):
    """Create a summary report of findings for group-level analysis."""
    report_file = os.path.join(results_dir, 'group_analysis_summary.txt')
    
    with open(report_file, 'w') as f:
        f.write("MUSIC EEG GROUP ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("PARTICIPANT MAPPING:\n")
        if participant_df is not None:
            groups = participant_df['group'].unique()
            for group in groups:
                group_participants = participant_df[participant_df['group'] == group]['participant_id'].unique()
                f.write(f"Group {group}: {list(group_participants)} (n={len(group_participants)})\n")
        f.write("\n")
        
        f.write("INDIVIDUAL RESULTS:\n")
        for name, result in individual_results.items():
            f.write(f"- {name}: {result['filename']}\n")
        f.write("\n")
        
        f.write("GROUP RESULTS:\n")
        for name, result in group_results.items():
            f.write(f"- {name}: {result['participants']} (n={result['n_participants']})\n")
        f.write("\n")
        
        f.write("KEY COMPARISONS GENERATED:\n")
        f.write("1. CAM-ICU Negative: Live Music vs No Music (therapeutic effect)\n")
        f.write("2. CAM-ICU Negative: Recorded Music vs No Music (music specificity)\n")
        f.write("3. CAM-ICU Negative: Pink Noise vs No Music (control comparison)\n\n")
        
        f.write("INTERPRETATION GUIDE:\n")
        f.write("- Positive differences (red/warm colors): Higher activity in first condition\n")
        f.write("- Negative differences (blue/cool colors): Higher activity in second condition\n")
        f.write("- Music conditions may show temporal/frontal differences\n")
        f.write("- CAM-ICU Negative participants have clearer cognition\n\n")
        
        f.write("FILES GENERATED:\n")
        f.write("- Group-level source estimates: music_cache/icu/group_key_stc.fif\n")
        f.write("- Group-level mean power: music_cache/icu/group_key_mean_stc.fif\n")
        f.write("- Comparison visualizations: music_results/comparisons/\n")
        f.write("- Both hemispheres (lh/rh) saved separately\n")
    
    print(f"Group analysis summary report saved to: {report_file}")

if __name__ == "__main__":
    start_time = time.time()
    
    print("Starting Music EEG Comparative Analysis...")
    individual_results, group_results = analyze_datasets()
    
    total_time = time.time() - start_time
    print(f"\nAnalysis complete! Total time: {total_time:.1f} seconds")
    print(f"Results saved to: {results_dir}")
    print("\nCheck the analysis_summary.txt file for interpretation guidance!") 