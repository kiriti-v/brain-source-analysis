#!/usr/bin/env python3
"""
Modern Brain Activity Explorer
Web-based interface with advanced features for brain visualization
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
# Removed make_subplots - now using simple direct image display
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import mne
from mne.datasets import sample
import base64
import io
from datetime import datetime
from scipy import stats
try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    print("Warning: statsmodels not available. FDR correction will not work.")
    multipletests = None

import json
import pickle
import signal
import sys
import threading
import time
from multiprocessing import Process
import socket
import platform

class ModernBrainExplorer:
    """Enhanced brain data explorer with group analysis support and better visualization."""
    
    def __init__(self, skip_image_generation=False):
        # Configure PyVista for headless rendering to prevent macOS VTK warnings
        self._configure_pyvista()
        
        # Initialize data structures
        self.data = {}
        self.roi_info = {}
        self.identical_conditions = set()
        self.app = None
        self.subject = 'fsaverage'
        self.subjects_dir = None  # Will be set only if needed
        self.use_custom_images = False
        self.custom_images_dir = "custom_brain_images"
        self.state_file = "ui_state.json"
        self.snapshots = []  # Store snapshot history
        self.skip_image_generation = skip_image_generation
        
        # Load the data
        self.load_data()
        
        # Setup the web app
        self.setup_app()
    
    def _configure_pyvista(self):
        """Configure PyVista for stable headless rendering."""
        try:
            import pyvista as pv
            import platform
            
            # Set up for headless rendering
            pv.OFF_SCREEN = True
            
            # Platform-specific configuration
            if platform.system() == 'Darwin':  # macOS
                # macOS-specific settings
                pv.global_theme.window_size = [800, 600]
                pv.global_theme.background = 'white'
                pv.global_theme.antialiasing = 'msaa'
                pv.global_theme.multi_samples = 4
                print("✅ PyVista configured for macOS headless rendering")
            else:
                # Linux systems
                try:
                    pv.start_xvfb()
                    print("✅ PyVista configured for Linux headless rendering")
                except:
                    print("⚠️  Xvfb not available, using default rendering")
            
        except Exception as e:
            print(f"⚠️  PyVista configuration warning: {e}")
            print("   Continuing with default settings...")
            
        # Set MNE to use minimal fsaverage download
        import mne
        mne.set_config('MNE_DATASETS_FSAVERAGE_PATH', None)  # Use cached if available
    
    def load_data(self):
        """Load ONLY the computed source estimates needed for main CAM ICU comparisons."""
        print("Loading CAM ICU comparison data...")
        cache_dir = "music_cache"
        self.data = {}
        self.roi_info = {}  # Store region of interest information
        self.identical_conditions = set()  # Track identical condition pairs
        
        # Define the main CAM ICU conditions we actually need
        cam_icu_conditions = [
            'CAM_ICU_Negative_Live_Music',
            'CAM_ICU_Negative_Recorded_Music', 
            'CAM_ICU_Negative_Pink_Noise',
            'CAM_ICU_Negative_No_Music'
        ]
        
        # Load only the main CAM ICU group-level results (mean power files)
        for dataset_name, data_dir in [('ICU', 'music_cache/icu')]:  # Only ICU data
            if not os.path.exists(data_dir):
                continue
                
            mean_stc_files = glob.glob(os.path.join(data_dir, "*_mean_stc-lh.stc"))
            
            for stc_file in mean_stc_files:
                condition_name = os.path.basename(stc_file).replace('_mean_stc-lh.stc', '')
                
                # Only load CAM ICU conditions
                if not any(cam_condition in condition_name for cam_condition in cam_icu_conditions):
                    continue
                
                stc_base = stc_file.replace('-lh.stc', '')
                full_name = f"{dataset_name}_{condition_name}"
                
                try:
                    stc = mne.read_source_estimate(stc_base)
                    # For mean power files, data is already power
                    stc_mean_power = stc.data.flatten() if stc.data.ndim > 1 else stc.data
                    
                    self.data[full_name] = {
                        'data': stc_mean_power,
                        'vertices': stc.vertices,
                        'raw_stc': stc,
                        'original_stc': stc,
                        'type': 'group'  # Mean power files are always group-level
                    }
                    
                    # Compute ROI statistics
                    self.roi_info[full_name] = self.compute_roi_stats(stc_mean_power, stc.vertices)
                    print(f"Loaded CAM ICU GROUP: {full_name}")
                    
                except Exception as e:
                    print(f"Error loading CAM ICU condition {condition_name}: {e}")
        
        # Also look for regular stc files (backup in case mean files don't exist)
        for dataset_name, data_dir in [('ICU', 'music_cache/icu')]:
            if not os.path.exists(data_dir):
                continue
                
            stc_files = glob.glob(os.path.join(data_dir, "*_stc-lh.stc"))
            
            for stc_file in stc_files:
                condition_name = os.path.basename(stc_file).replace('_stc-lh.stc', '')
                
                # Only load CAM ICU conditions that we haven't already loaded
                if not any(cam_condition in condition_name for cam_condition in cam_icu_conditions):
                    continue
                
                full_name = f"{dataset_name}_{condition_name}"
                if full_name in self.data:  # Skip if already loaded from mean files
                    continue
                
                stc_base = stc_file.replace('-lh.stc', '')
                
                try:
                    stc = mne.read_source_estimate(stc_base)
                    # Compute mean power
                    stc_power = np.abs(stc.data) ** 2
                    stc_mean_power = np.mean(stc_power, axis=1)
                    
                    self.data[full_name] = {
                        'data': stc_mean_power,
                        'vertices': stc.vertices,
                        'raw_stc': stc,
                        'original_stc': stc,
                        'type': 'group'  # These are group-level CAM ICU conditions
                    }
                    
                    # Compute ROI statistics
                    self.roi_info[full_name] = self.compute_roi_stats(stc_mean_power, stc.vertices)
                    print(f"Loaded CAM ICU GROUP (backup): {full_name}")
                    
                except Exception as e:
                    print(f"Error loading CAM ICU condition {condition_name}: {e}")
        
        if not self.data:
            raise ValueError("No CAM ICU conditions found! Check that the analysis has been run and files exist.")
        
        # Check for identical conditions
        self.detect_identical_conditions()
        
        print(f"✅ Loaded {len(self.data)} CAM ICU conditions (group-level only)")
        
        if self.identical_conditions:
            print(f"⚠️  Warning: Found {len(self.identical_conditions)} identical condition pairs")
        
        # Print loaded CAM ICU conditions
        print("\nCAM ICU conditions ready for analysis:")
        for condition in sorted(self.data.keys()):
            print(f"  - {condition}")
        
        # Check if we have the main comparison conditions
        required_conditions = [
            'ICU_CAM_ICU_Negative_Live_Music',
            'ICU_CAM_ICU_Negative_Recorded_Music', 
            'ICU_CAM_ICU_Negative_Pink_Noise',
            'ICU_CAM_ICU_Negative_No_Music'
        ]
        
        missing_conditions = [cond for cond in required_conditions if cond not in self.data]
        if missing_conditions:
            print(f"\n⚠️  Missing expected conditions:")
            for condition in missing_conditions:
                print(f"  - {condition}")
        else:
            print(f"\n✅ All main CAM ICU comparison conditions are available!")
        
        return self.data
    
    def detect_identical_conditions(self):
        """Detect pairs of conditions with identical data."""
        condition_names = list(self.data.keys())
        
        for i, cond1 in enumerate(condition_names):
            for j, cond2 in enumerate(condition_names[i+1:], i+1):
                data1 = self.data[cond1]['data']
                data2 = self.data[cond2]['data']
                
                # Check if data is identical (within numerical precision)
                if np.array_equal(data1, data2) or np.allclose(data1, data2, rtol=1e-12):
                    self.identical_conditions.add((cond1, cond2))
                    print(f"⚠️  Warning: {cond1} and {cond2} contain identical data!")

    def compute_roi_stats(self, data, vertices):
        """Compute region of interest statistics."""
        n_lh = len(vertices[0])
        lh_data = data[:n_lh]
        rh_data = data[n_lh:]
        
        return {
            'lh_mean': np.mean(lh_data),
            'lh_max': np.max(lh_data),
            'lh_std': np.std(lh_data),
            'rh_mean': np.mean(rh_data),
            'rh_max': np.max(rh_data),
            'rh_std': np.std(rh_data),
            'total_mean': np.mean(data),
            'total_max': np.max(data),
            'peak_vertices': np.argsort(data)[-10:]  # Top 10 vertices
        }
    
    def compute_z_scores(self, condition1, condition2=None, baseline_conditions=None):
        """
        Compute z-scores for statistical thresholding.
        
        Parameters:
        - condition1: Primary condition name
        - condition2: Comparison condition name (optional)
        - baseline_conditions: List of conditions to use as baseline for z-score computation
        """
        data1 = self.data[condition1]['data']
        
        if condition2:
            data2 = self.data[condition2]['data']
            diff_data = data1 - data2
        else:
            diff_data = data1
        
        # If baseline conditions are provided, use them to compute z-scores
        if baseline_conditions:
            baseline_data = []
            for baseline_cond in baseline_conditions:
                if baseline_cond in self.data:
                    baseline_data.append(self.data[baseline_cond]['data'])
            
            if baseline_data:
                baseline_stack = np.column_stack(baseline_data)
                baseline_mean = np.mean(baseline_stack, axis=1)
                baseline_std = np.std(baseline_stack, axis=1)
                
                # Avoid division by zero
                baseline_std[baseline_std == 0] = 1e-10
                z_scores = (diff_data - baseline_mean) / baseline_std
            else:
                # Fallback to simple z-score
                z_scores = stats.zscore(diff_data)
        else:
            # Simple z-score normalization
            z_scores = stats.zscore(diff_data)
        
        return z_scores, diff_data
    
    def compute_group_statistics(self, condition_a_list, condition_b_list):
        """
        Compute group-level statistics for multiple participants.
        
        Parameters:
        - condition_a_list: List of condition names for group A
        - condition_b_list: List of condition names for group B
        """
        # Collect individual difference maps
        diff_maps = []
        
        # Method 1: Individual differences then average
        for cond_a, cond_b in zip(condition_a_list, condition_b_list):
            if cond_a in self.data and cond_b in self.data:
                data_a = self.data[cond_a]['data']
                data_b = self.data[cond_b]['data']
                diff_map = data_a - data_b
                diff_maps.append(diff_map)
        
        if not diff_maps:
            return None, None, None
        
        diff_maps = np.array(diff_maps)
        
        # Group statistics
        group_mean = np.mean(diff_maps, axis=0)
        group_std = np.std(diff_maps, axis=0)
        group_sem = group_std / np.sqrt(len(diff_maps))
        
        # One-sample t-test against zero (testing if difference is significant)
        t_stats, p_values = stats.ttest_1samp(diff_maps, 0, axis=0)
        
        # Multiple comparison correction (FDR)
        if multipletests is not None:
            fdr_corrected = multipletests(p_values, alpha=0.05, method='fdr_bh')
            p_values_corrected = fdr_corrected[1]
            significant_mask = fdr_corrected[0]
        else:
            print("Warning: FDR correction not available. Using uncorrected p-values.")
            p_values_corrected = p_values
            significant_mask = p_values < 0.05
            fdr_corrected = (significant_mask, p_values_corrected)
        
        return {
            'group_mean': group_mean,
            'group_std': group_std,
            'group_sem': group_sem,
            't_stats': t_stats,
            'p_values': p_values,
            'p_values_corrected': p_values_corrected,
            'significant_mask': significant_mask,
            'n_participants': len(diff_maps)
        }, diff_maps, fdr_corrected
    
    def get_anatomical_coordinates(self, vertices):
        """
        Get anatomical coordinates for vertices using the sample brain.
        This is a simplified version - in practice you'd use proper atlases.
        """
        try:
            # Load sample source space for coordinate information
            data_path = sample.data_path()
            src_fname = os.path.join(data_path, 'subjects', 'sample', 'bem', 'sample-oct-6-src.fif')
            
            if os.path.exists(src_fname):
                src = mne.read_source_spaces(src_fname)
                
                # Get coordinates for left and right hemisphere
                lh_coords = src[0]['rr'][vertices[0]]  # Left hemisphere coordinates
                rh_coords = src[1]['rr'][vertices[1]]  # Right hemisphere coordinates
                
                # Combine coordinates
                all_coords = np.vstack([lh_coords, rh_coords])
                
                return all_coords
            else:
                # Fallback: generate approximate coordinates
                n_vertices = len(vertices[0]) + len(vertices[1])
                return np.random.randn(n_vertices, 3) * 0.1  # Rough brain-sized coordinates
                
        except Exception as e:
            print(f"Warning: Could not load anatomical coordinates: {e}")
            # Fallback coordinates
            n_vertices = len(vertices[0]) + len(vertices[1])
            return np.random.randn(n_vertices, 3) * 0.1
    
    def get_roi_labels(self, vertices):
        """
        Get anatomical region labels for vertices.
        This is a simplified version using basic hemisphere divisions.
        """
        n_lh = len(vertices[0])
        n_rh = len(vertices[1])
        
        # Create basic ROI labels
        roi_labels = []
        
        # Left hemisphere regions (simplified)
        lh_regions = ['LH_Frontal', 'LH_Parietal', 'LH_Temporal', 'LH_Occipital']
        for i in range(n_lh):
            region_idx = int((i / n_lh) * len(lh_regions))
            roi_labels.append(lh_regions[min(region_idx, len(lh_regions)-1)])
        
        # Right hemisphere regions (simplified)
        rh_regions = ['RH_Frontal', 'RH_Parietal', 'RH_Temporal', 'RH_Occipital']
        for i in range(n_rh):
            region_idx = int((i / n_rh) * len(rh_regions))
            roi_labels.append(rh_regions[min(region_idx, len(rh_regions)-1)])
        
        return roi_labels
    
    def export_data(self, condition1, condition2=None, include_coords=True, 
                   include_stats=True, export_format='csv'):
        """
        Export numerical data with coordinates and statistics.
        
        Parameters:
        - condition1: Primary condition
        - condition2: Comparison condition (optional)
        - include_coords: Include 3D coordinates
        - include_stats: Include statistical measures
        - export_format: 'csv' or 'json'
        """
        data1 = self.data[condition1]['data']
        vertices1 = self.data[condition1]['vertices']
        
        # Prepare data dictionary
        export_data = {
            'vertex_id': np.arange(len(data1)),
            'condition1_name': condition1,
            'condition1_value': data1
        }
        
        # Add comparison data if provided
        if condition2:
            data2 = self.data[condition2]['data']
            export_data['condition2_name'] = condition2
            export_data['condition2_value'] = data2
            export_data['difference'] = data1 - data2
        
        # Add coordinates if requested
        if include_coords:
            coords = self.get_anatomical_coordinates(vertices1)
            export_data['coord_x'] = coords[:, 0]
            export_data['coord_y'] = coords[:, 1]
            export_data['coord_z'] = coords[:, 2]
        
        # Add anatomical labels
        roi_labels = self.get_roi_labels(vertices1)
        export_data['anatomical_region'] = roi_labels
        
        # Add hemisphere information
        n_lh = len(vertices1[0])
        hemisphere = ['Left'] * n_lh + ['Right'] * (len(data1) - n_lh)
        export_data['hemisphere'] = hemisphere
        
        # Add statistical measures if requested
        if include_stats and condition2:
            z_scores, _ = self.compute_z_scores(condition1, condition2)
            export_data['z_score'] = z_scores
            
            # Compute p-values (assuming normal distribution)
            p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))  # Two-tailed test
            export_data['p_value'] = p_values
            
            # Add significance flags
            export_data['significant_p05'] = p_values < 0.05
            export_data['significant_p01'] = p_values < 0.01
            export_data['significant_p001'] = p_values < 0.001
        
        return export_data
    
    def compute_roi_summary(self, condition1, condition2=None):
        """
        Compute summary statistics by anatomical region.
        """
        data1 = self.data[condition1]['data']
        vertices1 = self.data[condition1]['vertices']
        roi_labels = self.get_roi_labels(vertices1)
        
        if condition2:
            data2 = self.data[condition2]['data']
            plot_data = data1 - data2
            analysis_type = 'difference'
        else:
            plot_data = data1
            analysis_type = 'single_condition'
        
        # Group by ROI
        unique_rois = list(set(roi_labels))
        roi_summary = {}
        
        for roi in unique_rois:
            roi_mask = np.array(roi_labels) == roi
            roi_data = plot_data[roi_mask]
            
            if len(roi_data) > 0:
                roi_summary[roi] = {
                    'n_vertices': len(roi_data),
                    'mean': np.mean(roi_data),
                    'std': np.std(roi_data),
                    'min': np.min(roi_data),
                    'max': np.max(roi_data),
                    'median': np.median(roi_data),
                    'q25': np.percentile(roi_data, 25),
                    'q75': np.percentile(roi_data, 75)
                }
                
                # Add statistical significance if it's a comparison
                if condition2:
                    z_scores, _ = self.compute_z_scores(condition1, condition2)
                    roi_z_scores = z_scores[roi_mask]
                    
                    roi_p_values = 2 * (1 - stats.norm.cdf(np.abs(roi_z_scores)))
                    
                    roi_summary[roi].update({
                        'mean_z_score': np.mean(roi_z_scores),
                        'max_z_score': np.max(np.abs(roi_z_scores)),
                        'n_significant_p05': np.sum(roi_p_values < 0.05),
                        'n_significant_p01': np.sum(roi_p_values < 0.01),
                        'percent_significant_p05': (np.sum(roi_p_values < 0.05) / len(roi_p_values)) * 100
                    })
        
        return roi_summary
    
    def is_comparison_valid(self, cond1, cond2):
        """Check if a comparison between two conditions is valid (not identical)."""
        if not cond2:  # Single condition view is always valid
            return True
        
        # Check if this pair is in our identical conditions set
        return not ((cond1, cond2) in self.identical_conditions or 
                   (cond2, cond1) in self.identical_conditions)
    
    def generate_brain_images(self, output_dir="brain_images"):
        """Generate brain images for CAM ICU conditions only."""
        
        # Check if image generation is disabled
        if self.skip_image_generation:
            print("⚡ Image generation skipped (quick mode)")
            return
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Check if key CAM ICU images already exist
        key_demo_files = [
            "ICU_CAM_ICU_Negative_Live_Music_minus_ICU_CAM_ICU_Negative_No_Music_lh.png",
            "ICU_CAM_ICU_Negative_Recorded_Music_minus_ICU_CAM_ICU_Negative_No_Music_lh.png",
            "ICU_CAM_ICU_Negative_Pink_Noise_minus_ICU_CAM_ICU_Negative_No_Music_lh.png"
        ]
        
        existing_demo_files = sum(1 for f in key_demo_files if os.path.exists(os.path.join(output_dir, f)))
        
        if existing_demo_files >= 2:  # If at least 2 key demo files exist
            print(f"🚀 Found {existing_demo_files} key CAM ICU images - skipping image generation")
            print("   Use --force-regenerate to regenerate all images")
            return
            
        print(f"🧠 Generating CAM ICU brain images...")
        
        # Generate individual condition images (CAM ICU only)
        print("\n📸 Generating CAM ICU condition images...")
        conditions = list(self.data.keys())
        for i, condition in enumerate(conditions):
            print(f"Processing ({i+1}/{len(conditions)}): {condition}...")
            try:
                self._generate_condition_images(condition, output_dir)
            except Exception as e:
                print(f"❌ Error generating {condition}: {e}")
                continue
        
        # Generate the key CAM ICU comparison images
        print("\n🎯 Generating key CAM ICU comparison images...")
        
        # Define the key CAM ICU comparisons
        key_comparisons = []
        
        # Find baseline (No Music) condition
        baseline_condition = None
        for condition in self.data.keys():
            if 'CAM_ICU_Negative_No_Music' in condition:
                baseline_condition = condition
                break
        
        if baseline_condition:
            # Key demo comparisons vs No Music
            for condition in self.data.keys():
                if condition != baseline_condition and 'CAM_ICU_Negative' in condition:
                    key_comparisons.append((condition, baseline_condition))
                    print(f"✅ Added key comparison: {condition} vs {baseline_condition}")
        
        # Generate only these key comparisons
        for i, (condition1, condition2) in enumerate(key_comparisons):
            print(f"🎯 Generating comparison ({i+1}/{len(key_comparisons)}): {condition1} vs {condition2}")
            try:
                self._generate_comparison_images(condition1, condition2, output_dir)
            except Exception as e:
                print(f"❌ Error generating comparison {condition1} vs {condition2}: {e}")
                continue
        
        print(f"\n✅ CAM ICU brain image generation complete!")
        print(f"📁 Images saved to: {output_dir}")
        print(f"🎯 Generated {len(key_comparisons)} key CAM ICU comparisons")
    
    def _generate_condition_images(self, condition, output_dir):
        """Generate PyVista brain images for a single condition with timeout protection."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("PyVista rendering timed out")
        
        # Get data
        data = self.data[condition]['data']
        vertices = self.data[condition]['vertices']
        
        # Better thresholding to avoid "brains on fire" - use 90th percentile
        vmin = np.percentile(data, 90)  # Only show top 10% of activity (more selective)
        vmax = np.percentile(data, 99)  # Avoid extreme outliers
        
        print(f"   Using thresholds: vmin={vmin:.4f} (90th percentile), vmax={vmax:.4f} (99th percentile)")
        
        # Create source estimate object
        stc_plot = mne.SourceEstimate(
            data,
            vertices=vertices,
            tmin=0,
            tstep=1,
            subject='fsaverage'
        )
        
        # Generate images for both hemispheres with timeout protection
        for hemi in ['lh', 'rh']:
            filename = os.path.join(output_dir, f"{condition}_{hemi}.png")
            
            # Skip if already exists
            if os.path.exists(filename):
                print(f"⚡ Skipping: {os.path.basename(filename)} (already exists)")
                continue
            
            try:
                # Set a timeout of 30 seconds per image
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
                
                # Generate the brain image
                brain = stc_plot.plot(
                    hemi=hemi,
                    views='lat',
                    backend='auto',
                    size=(800, 600),
                    clim={'kind': 'value', 'lims': [vmin, (vmin + vmax) / 2, vmax]},
                    colormap='hot',
                    background='white',
                    foreground='black'
                )
                
                # Save the image
                brain.save_image(filename)
                brain.close()
                
                # Clear the alarm
                signal.alarm(0)
                
                print(f"📸 Saved: {os.path.basename(filename)}")
                
            except TimeoutError:
                print(f"⏰ Timeout generating {os.path.basename(filename)} - skipping")
                continue
            except Exception as e:
                print(f"❌ Error generating {os.path.basename(filename)}: {e}")
                continue
            finally:
                # Always clear the alarm
                signal.alarm(0)
    
    def _generate_comparison_images(self, condition1, condition2, output_dir):
        """Generate PyVista brain images for condition comparison with timeout protection."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("PyVista rendering timed out")
        
        # Get data for both conditions
        data1 = self.data[condition1]['data']
        data2 = self.data[condition2]['data']
        vertices1 = self.data[condition1]['vertices']
        vertices2 = self.data[condition2]['vertices']
        
        # Calculate difference
        difference_data = data1 - data2
        
        # Use thresholding for difference plots (85th percentile)
        threshold_value = np.percentile(np.abs(difference_data), 85)
        vmin, vmax = -threshold_value, threshold_value
        
        print(f"   Difference thresholds: vmin={vmin:.4f}, vmax={vmax:.4f} (85th percentile)")
        
        # Create source estimate object for difference
        stc_diff = mne.SourceEstimate(
            difference_data,
            vertices=vertices1,  # Should be the same for both conditions
            tmin=0,
            tstep=1,
            subject='fsaverage'
        )
        
        # Generate images for both hemispheres with timeout protection
        for hemi in ['lh', 'rh']:
            filename = os.path.join(output_dir, f"{condition1}_minus_{condition2}_{hemi}.png")
            
            # Skip if already exists
            if os.path.exists(filename):
                print(f"⚡ Skipping: {os.path.basename(filename)} (already exists)")
                continue
            
            try:
                # Set a timeout of 30 seconds per image
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
                
                # Generate the brain image
                brain = stc_diff.plot(
                    hemi=hemi,
                    views='lat',
                    backend='auto',
                    size=(800, 600),
                    clim={'kind': 'value', 'lims': [vmin, (vmin + vmax) / 2, vmax]},
                    colormap='RdBu_r',
                    background='white',
                    foreground='black'
                )
                
                # Save the image
                brain.save_image(filename)
                brain.close()
                
                # Clear the alarm
                signal.alarm(0)
                
                print(f"📸 Saved: {os.path.basename(filename)}")
                
            except TimeoutError:
                print(f"⏰ Timeout generating {os.path.basename(filename)} - skipping")
                continue
            except Exception as e:
                print(f"❌ Error generating {os.path.basename(filename)}: {e}")
                continue
            finally:
                # Always clear the alarm
                signal.alarm(0)

    def create_brain_plot(self, condition1, condition2=None, vmin=None, vmax=None):
        """Create brain plot using ONLY pre-generated images - never generate on-the-fly."""
        
        # Check if we should use custom images first
        if hasattr(self, 'use_custom_images') and self.use_custom_images:
            # Try to load custom images
            if condition2:
                lh_custom = os.path.join(self.custom_images_dir, f"{condition1}_minus_{condition2}_lh_custom.png")
                rh_custom = os.path.join(self.custom_images_dir, f"{condition1}_minus_{condition2}_rh_custom.png")
            else:
                lh_custom = os.path.join(self.custom_images_dir, f"{condition1}_lh_custom.png")
                rh_custom = os.path.join(self.custom_images_dir, f"{condition1}_rh_custom.png")
            
            if os.path.exists(lh_custom) and os.path.exists(rh_custom):
                print(f"🎨 Using custom images for {condition1} vs {condition2 or 'None'}")
                return self.create_brain_plot_custom(condition1, condition2)
        
        # Use pre-generated images (original logic)
        if condition2:
            # Load difference image
            lh_filename = f"{condition1}_minus_{condition2}_lh.png"
            rh_filename = f"{condition1}_minus_{condition2}_rh.png"
            title = f"{condition1} - {condition2}"
        else:
            # Load single condition image
            lh_filename = f"{condition1}_lh.png"
            rh_filename = f"{condition1}_rh.png"
            title = condition1
        
        # Look for images in brain_images directory
        brain_images_dir = 'brain_images'
        lh_path = os.path.join(brain_images_dir, lh_filename)
        rh_path = os.path.join(brain_images_dir, rh_filename)
        
        if not os.path.exists(lh_path) or not os.path.exists(rh_path):
            # DO NOT generate images on-the-fly - show clear error instead
            print(f"❌ Pre-generated images not found: {lh_filename}, {rh_filename}")
            print(f"   Run: python modern_brain_explorer.py --generate-images first")
            
            # Return informative error plot
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                text=f"⚠️ Brain Images Not Pre-Generated<br><br>" +
                     f"Missing: {lh_filename}, {rh_filename}<br><br>" +
                     f"To fix this:<br>" +
                     f"1. Stop the web app (Ctrl+C)<br>" +
                     f"2. Run the pre-generation step<br>" +
                     f"3. Restart the web app<br><br>" +
                     f"This prevents PyVista threading conflicts.",
                showarrow=False,
                font=dict(size=14, color='#e74c3c'),
                bgcolor='#ffeaa7',
                bordercolor='#e74c3c',
                borderwidth=2
            )
            fig.update_layout(
                title="Images Not Pre-Generated",
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                plot_bgcolor='white',
                height=500
            )
            return fig
        
        # Load pre-generated images
        try:
            with open(lh_path, 'rb') as f:
                lh_base64 = base64.b64encode(f.read()).decode()
            
            with open(rh_path, 'rb') as f:
                rh_base64 = base64.b64encode(f.read()).decode()
        except Exception as e:
            print(f"❌ Error loading pre-generated images: {e}")
            
            # Return error plot
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                text=f"❌ Error Loading Brain Images<br><br>{str(e)}",
                showarrow=False,
                font=dict(size=14, color='red')
            )
            fig.update_layout(title="Image Loading Error")
            return fig
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add left hemisphere image
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{lh_base64}",
                xref="paper", yref="paper",
                x=0, y=1,
                sizex=0.48, sizey=1,
                xanchor="left", yanchor="top",
                layer="below"
            )
        )
        
        # Add right hemisphere image
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{rh_base64}",
                xref="paper", yref="paper",
                x=0.52, y=1,
                sizex=0.48, sizey=1,
                xanchor="left", yanchor="top",
                layer="below"
            )
        )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16, color='#2c3e50')),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=50, b=0),
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        print(f"✅ Loaded pre-generated image: {lh_filename}")
        print(f"✅ Loaded pre-generated image: {rh_filename}")
        
        return fig
    
    def _generate_custom_visualization(self, condition1, condition2, vmin, vmax, percentile, threshold_method, z_threshold, p_threshold):
        """Generate brain visualization with custom parameters (for regeneration)."""
        import os
        import matplotlib
        matplotlib.use('Agg')
        
        print(f"🎨 Generating custom visualization...")
        
        # Get data
        data1 = self.data[condition1]['data']
        vertices1 = self.data[condition1]['vertices']
        
        if condition2:
            data2 = self.data[condition2]['data']
            plot_data = data1 - data2
            title = f"Custom: {condition1} - {condition2}"
        else:
            plot_data = data1
            title = f"Custom: {condition1}"
            
        # Apply custom thresholds based on method
        if threshold_method == 'percentile':
            if vmin is None or vmax is None:
                custom_vmin = np.percentile(plot_data, percentile)
                custom_vmax = np.percentile(plot_data, 99)
            else:
                custom_vmin, custom_vmax = vmin, vmax
        elif threshold_method == 'z_score' and condition2:
            # Use z-score thresholding
            z_scores, _ = self.compute_z_scores(condition1, condition2)
            z_mask = np.abs(z_scores) >= z_threshold
            plot_data = plot_data.copy()
            plot_data[~z_mask] = 0
            title += f" (|Z| ≥ {z_threshold})"
            
            # Auto-scale for z-score filtered data
            non_zero_data = plot_data[plot_data != 0]
            if len(non_zero_data) > 0:
                custom_vmin, custom_vmax = np.min(non_zero_data), np.max(non_zero_data)
            else:
                custom_vmin, custom_vmax = -1, 1
        elif threshold_method == 'p_value' and condition2:
            # Use p-value thresholding
            z_scores, _ = self.compute_z_scores(condition1, condition2)
            from scipy import stats
            p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
            p_mask = p_values < p_threshold
            plot_data = plot_data.copy()
            plot_data[~p_mask] = 0
            title += f" (p < {p_threshold})"
            
            # Auto-scale for p-value filtered data
            non_zero_data = plot_data[plot_data != 0]
            if len(non_zero_data) > 0:
                custom_vmin, custom_vmax = np.min(non_zero_data), np.max(non_zero_data)
            else:
                custom_vmin, custom_vmax = -1, 1
        else:
            # Fallback to percentile method
            custom_vmin = np.percentile(plot_data, 75)
            custom_vmax = np.percentile(plot_data, 99)
            
        # Create source estimate object
        stc_plot = mne.SourceEstimate(
            plot_data,
            vertices=vertices1,
            tmin=0,
            tstep=1,
            subject=self.subject
        )
        
        # Generate images using PyVista (in separate process to avoid threading)
        images = {}
        for hemi in ['lh', 'rh']:
            try:
                print(f"   🧠 Generating {hemi} hemisphere...")
                
                # Use auto backend for custom generation
                brain = mne.viz.plot_source_estimates(
                    stc_plot,
                    subject=self.subject,
                    subjects_dir=self.subjects_dir,
                    hemi=hemi,
                    views='lat',
                    backend='auto',
                    time_viewer=False,
                    clim=dict(kind='value', lims=[custom_vmin, (custom_vmin+custom_vmax)/2, custom_vmax]),
                    colorbar=False,
                    background='white',
                    size=(800, 600)
                )
                
                # Get screenshot
                img_array = brain.screenshot()
                brain.close()
                
                # Convert to base64
                from PIL import Image
                from io import BytesIO
                import base64
                
                img = Image.fromarray(img_array)
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                images[hemi] = f"data:image/png;base64,{image_base64}"
                
                print(f"   ✅ {hemi} hemisphere complete")
                
            except Exception as e:
                print(f"   ❌ Error generating {hemi}: {e}")
                # Use placeholder
                images[hemi] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        # Create simple figure to display the custom images
        import plotly.graph_objects as go
        fig = go.Figure()
        
        # Add left hemisphere image
        fig.add_layout_image(
            dict(
                source=images['lh'],
                xref="paper", yref="paper",
                x=0, y=1,
                sizex=0.48, sizey=1,
                xanchor="left", yanchor="top",
                layer="below"
            )
        )
            
        # Add right hemisphere image  
        fig.add_layout_image(
            dict(
                source=images['rh'],
                xref="paper", yref="paper", 
                x=0.52, y=1,
                sizex=0.48, sizey=1,
                xanchor="left", yanchor="top",
                layer="below"
            )
        )
        
        # Clean layout
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16, color='#2c3e50')),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=50, b=0),
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_brain_plot_statistical(self, condition1, condition2, plot_data, stat_values, 
                                     threshold_method, threshold_value):
        """
        Create brain visualization with statistical thresholding.
        
        Parameters:
        - condition1: Primary condition name
        - condition2: Comparison condition name
        - plot_data: Data to plot (already thresholded)
        - stat_values: Statistical values (z-scores or p-values)
        - threshold_method: 'z_score' or 'p_value'
        - threshold_value: Threshold value used
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from io import BytesIO
        import base64
        
        vertices1 = self.data[condition1]['vertices']
        
        # Create title based on statistical method
        if threshold_method == 'z_score':
            title = f"Statistical Analysis (|Z| ≥ {threshold_value}): {condition1} - {condition2}"
            colormap = 'RdBu_r'
            stat_label = 'Z-Score'
        elif threshold_method == 'p_value':
            title = f"Statistical Analysis (p < {threshold_value}): {condition1} - {condition2}"
            colormap = 'RdBu_r'
            stat_label = 'P-Value'
        else:
            title = f"Statistical Analysis: {condition1} - {condition2}"
            colormap = 'RdBu_r'
            stat_label = 'Statistic'
        
        # Use adaptive scaling for statistical data
        non_zero_data = plot_data[plot_data != 0]
        if len(non_zero_data) > 0:
            vmin = np.min(non_zero_data)
            vmax = np.max(non_zero_data)
            # Ensure symmetric scale for difference data
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = -1, 1
        
        # Create source estimate object for proper brain visualization
        stc_plot = mne.SourceEstimate(
            plot_data,
            vertices=vertices1,
            tmin=0,
            tstep=1,
            subject=self.subject
        )
        
        # Create matplotlib figures for each hemisphere
        images = {}
        
        for hemi in ['lh', 'rh']:
            try:
                # Use auto backend for stable statistical rendering
                brain = mne.viz.plot_source_estimates(
                    stc_plot,
                    subject=self.subject,
                    subjects_dir=self.subjects_dir,
                    hemi=hemi,
                    views='lat',
                    backend='auto',
                    time_viewer=False,
                    clim=dict(kind='value', lims=[vmin, 0, vmax]),
                    colorbar=False,
                    background='white',
                    size=(800, 600)
                )
                
                # PyVista backend gives us perfect full-surface statistical screenshots
                img_array = brain.screenshot()  # Returns full-canvas PNG as NumPy array
                brain.close()  # Clean up PyVista resources
                
                # Convert NumPy array to PIL Image for processing
                from PIL import Image
                img = Image.fromarray(img_array)
                
                # Save debug version to inspect the clean PyVista statistical output
                debug_file = f"{hemi}_pyvista_statistical_brain.png"
                img.save(debug_file)
                print(f"🔍 Saved PyVista debug statistical brain: {debug_file}")
                
                # Convert to base64 for Dash/Plotly embedding
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                images[hemi] = f"data:image/png;base64,{image_base64}"
                
                print(f"Successfully created PyVista statistical brain image for {hemi} (full surface, no clipping)")
                
            except Exception as e:
                print(f"Error creating {hemi} statistical brain plot: {e}")
                # Fallback statistical visualization
                fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
                
                if hemi == 'lh':
                    data = plot_data[:len(vertices1[0])]
                else:
                    data = plot_data[len(vertices1[0]):]
                
                # Create brain-like 2D representation
                size = int(np.sqrt(len(data))) + 1
                brain_grid = np.full((size, size), np.nan)
                
                center_x, center_y = size//2, size//2
                for i, val in enumerate(data):
                    if val != 0:  # Only plot significant values
                        angle = (i / len(data)) * 2 * np.pi
                        radius_x = 0.35 * size * (0.9 + 0.1 * np.cos(2 * angle))
                        radius_y = 0.28 * size * (0.8 + 0.2 * np.cos(2 * angle))
                        
                        x = int(center_x + radius_x * np.cos(angle))
                        y = int(center_y + radius_y * np.sin(angle))
                        
                        if 0 <= x < size and 0 <= y < size:
                            brain_grid[y, x] = val
                
                im = ax.imshow(brain_grid, cmap=colormap, vmin=vmin, vmax=vmax,
                              aspect='equal', interpolation='bilinear')
                
                ax.set_title(f"{hemi.upper()} - {stat_label}", 
                            fontsize=14, fontweight='bold', pad=15)
                ax.axis('off')
                
                # Create colorbar positioned like the MNE version
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="4%", pad=0.1)
                
                cbar = plt.colorbar(im, cax=cax)
                cbar.set_label(stat_label, fontsize=10, rotation=270, 
                              labelpad=15, weight='bold', color='black')
                
                # Clean colorbar ticks
                ticks = np.linspace(vmin, vmax, 4)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f'{tick:.2f}' for tick in ticks])
                
                cbar.ax.tick_params(labelsize=9, colors='black', width=1, length=3)
                cbar.ax.yaxis.label.set_color('black')
                cbar.outline.set_edgecolor('black')
                cbar.outline.set_linewidth(1)
                
                # Convert to base64 - MAXIMUM padding to preserve brain edges
                buffer = BytesIO()
                fig.tight_layout()
                fig.savefig(buffer, format='png', dpi=100,
                           facecolor='white',     # Remove bbox_inches='tight'
                           bbox_inches=None,      # DO NOT auto-crop
                           pad_inches=1.0,        # Lots of breathing room
                           edgecolor='none')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                images[hemi] = f"data:image/png;base64,{image_base64}"
                
                print(f"Successfully created clean statistical fallback image for {hemi} (max padding)")
                plt.close(fig)
                buffer.close()
        
        # Simply display the clean PyVista statistical screenshots without complex plot machinery
        import plotly.graph_objects as go
        
        # Create minimal figure to display the PyVista statistical brain screenshots
        fig = go.Figure()
        
        # Add left hemisphere PyVista statistical screenshot
        fig.add_layout_image(
            dict(
                source=images['lh'],
                xref="paper", yref="paper",
                x=0, y=1,
                sizex=0.48, sizey=1,
                xanchor="left", yanchor="top",
                layer="below"
            )
        )
        
        # Add right hemisphere PyVista statistical screenshot  
        fig.add_layout_image(
            dict(
                source=images['rh'],
                xref="paper", yref="paper", 
                x=0.52, y=1,
                sizex=0.48, sizey=1,
                xanchor="left", yanchor="top",
                layer="below"
            )
        )
        
        # Simple clean layout - just display the PyVista statistical screenshots
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16, color='#2c3e50')),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=50, b=0),
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def setup_app(self):
        """Set up the Dash web application."""
        
        # Initialize Dash app with Bootstrap theme
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title = "Brain Source Localization Analysis"
        
        # Define the layout
        self.app.layout = dbc.Container([
            
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Brain Source Localization Analysis Platform", 
                           className="text-center mb-4",
                           style={'color': '#2c3e50', 'fontWeight': 'bold'}),
                    html.Div(id='system-status', className="text-center mb-3")
                ])
            ]),
            
            # Main content
            dbc.Row([
                
                # Control Panel (Left Sidebar)
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Controls", className="text-center")),
                        dbc.CardBody([
                            
                            # Condition Selection
                            html.H5("📊 Condition Selection", style={'color': '#34495e'}),
                            html.Hr(),
                            
                            html.Label("Primary Condition:", style={'fontWeight': 'bold'}),
                            dcc.Dropdown(
                                id='condition1-dropdown',
                                options=[
                                    {'label': f"{k.replace('_', ' ')}", 'value': k} 
                                    for k in sorted(self.data.keys(), key=lambda x: (self.data[x]['type'] != 'group', x))
                                ],
                                value=list(self.data.keys())[0],
                                style={'marginBottom': '10px'}
                            ),
                            
                            html.Label("Comparison Condition (Optional):", style={'fontWeight': 'bold'}),
                            dcc.Dropdown(
                                id='condition2-dropdown',
                                options=[{'label': 'None', 'value': ''}] + 
                                        [
                                            {'label': f"{k.replace('_', ' ')}", 'value': k} 
                                            for k in sorted(self.data.keys(), key=lambda x: (self.data[x]['type'] != 'group', x))
                                        ],
                                value='',
                                style={'marginBottom': '10px'}
                            ),
                            
                            # Data Quality Warning
                            html.Div(id='data-quality-warning', style={'marginBottom': '20px'}),
                            
                            # Threshold Controls
                            html.H5("🎚️ Threshold Controls", style={'color': '#34495e'}),
                            html.Hr(),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Min Value:", style={'fontWeight': 'bold'}),
                                    dbc.Input(
                                        id='vmin-input',
                                        type='number',
                                        step=0.001,
                                        value=0.0,  # Default value
                                        style={'marginBottom': '10px'}
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("Max Value:", style={'fontWeight': 'bold'}),
                                    dbc.Input(
                                        id='vmax-input',
                                        type='number',
                                        step=0.001,
                                        value=1.0,  # Default value
                                        style={'marginBottom': '10px'}
                                    )
                                ], width=6)
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label([
                                        "Activation Threshold: ",
                                        html.Span("ⓘ", id="percentile-tooltip", style={'cursor': 'pointer', 'color': '#007bff'})
                                    ], style={'fontWeight': 'bold'}),
                                    dbc.Tooltip(
                                        "75% = show only top 25% strongest activations (like PDF). "
                                        "50% = show top 50% of activity. "
                                        "Lower values = show more of the brain.",
                                        target="percentile-tooltip"
                                    ),
                                    dbc.Input(
                                        id='percentile-input',
                                        type='number',
                                        value=75,  # Default to more selective like PDF
                                        min=0,
                                        max=95,
                                        style={'marginBottom': '10px'}
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Button("Auto Scale", id='auto-scale-btn', color='primary', size='sm',
                                              style={'marginTop': '25px'})
                                ], width=6)
                            ]),
                            
                            dbc.Button("Update Visualization", id='update-btn', color='success', 
                                      className='w-100 mb-3'),
                            
                            # Parameter Change Detection & Regeneration
                            html.Div(id='parameter-status', style={'marginBottom': '10px'}),
                            
                            dbc.Button("🔄 Regenerate with New Settings", 
                                      id='regenerate-btn', 
                                      color='warning', 
                                      className='w-100 mb-3',
                                      style={'display': 'none'}),  # Initially hidden
                            
                            # Statistical Controls
                            html.H5("📊 Statistical Analysis", style={'color': '#34495e'}),
                            html.Hr(),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Threshold Method:", style={'fontWeight': 'bold'}),
                                    dcc.Dropdown(
                                        id='threshold-method',
                                        options=[
                                            {'label': 'Percentile', 'value': 'percentile'},
                                            {'label': 'Z-Score', 'value': 'z_score'},
                                            {'label': 'P-Value', 'value': 'p_value'}
                                        ],
                                        value='percentile',
                                        style={'marginBottom': '10px'}
                                    )
                                ], width=12)
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Z-Score Threshold:", style={'fontWeight': 'bold'}),
                                    dbc.Input(
                                        id='z-threshold-input',
                                        type='number',
                                        value=1.96,  # p < 0.05
                                        step=0.1,
                                        disabled=True,
                                        style={'marginBottom': '10px'}
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Label("P-Value Threshold:", style={'fontWeight': 'bold'}),
                                    dbc.Input(
                                        id='p-threshold-input',
                                        type='number',
                                        value=0.05,
                                        step=0.001,
                                        disabled=True,
                                        style={'marginBottom': '10px'}
                                    )
                                ], width=6)
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    dcc.Checklist(
                                        id='correction-checklist',
                                        options=[
                                            {'label': ' Apply Multiple Comparison Correction', 'value': 'fdr'}
                                        ],
                                        value=[],
                                        style={'marginBottom': '10px'}
                                    )
                                ])
                            ]),
                            
                            # Export Controls
                            html.H5("💾 Export & Snapshots", style={'color': '#34495e'}),
                            html.Hr(),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Save PNG", id='save-png-btn', color='info', size='sm')
                                ], width=4),
                                dbc.Col([
                                    dbc.Button("Export Data", id='export-data-btn', color='success', size='sm')
                                ], width=4),
                                dbc.Col([
                                    dbc.Button("Take Snapshot", id='snapshot-btn', color='warning', size='sm')
                                ], width=4)
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Export Format:", style={'fontWeight': 'bold', 'marginTop': '10px'}),
                                    dcc.Dropdown(
                                        id='export-format',
                                        options=[
                                            {'label': 'CSV', 'value': 'csv'},
                                            {'label': 'JSON', 'value': 'json'}
                                        ],
                                        value='csv',
                                        style={'marginBottom': '10px'}
                                    )
                                ], width=6),
                                dbc.Col([
                                    dcc.Checklist(
                                        id='export-options',
                                        options=[
                                            {'label': ' Include Coordinates', 'value': 'coords'},
                                            {'label': ' Include Statistics', 'value': 'stats'}
                                        ],
                                        value=['coords', 'stats'],
                                        style={'marginTop': '25px'}
                                    )
                                ], width=6)
                            ]),
                            
                            html.Div(id='download-link', style={'marginTop': '10px'}),
                            
                            # ROI Information
                            html.H5("📍 Region Info", style={'color': '#34495e', 'marginTop': '20px'}),
                            html.Hr(),
                            html.Div(id='roi-info', style={'fontSize': '12px'})
                            
                        ])
                    ])
                ], width=3),
                

                
                # Main Visualization Area
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(
                                id='brain-plot',
                                style={'height': '500px'},
                                config={
                                    'displayModeBar': True,
                                    'toImageButtonOptions': {
                                        'format': 'png',
                                        'filename': 'brain_activity',
                                        'height': 500,
                                        'width': 800,
                                        'scale': 2
                                    }
                                }
                            )
                        ])
                    ]),
                    
                    # Data Range Information
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("📈 Data Statistics", style={'color': '#34495e'}),
                            html.Div(id='data-stats')
                        ])
                    ], style={'marginTop': '20px'}),
                    
                    # ROI Summary Statistics
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("🏷️ ROI Summary Statistics", style={'color': '#34495e'}),
                            html.Div(id='roi-summary-stats')
                        ])
                    ], style={'marginTop': '20px'})
                    
                ], width=6),
                
                # Demo Section (if group conditions are available)
                *([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4("Key Comparisons", className="text-center")),
                            dbc.CardBody([
                                html.Div(id='demo-comparisons'),
                                html.Hr(),
                                html.P("Quick Actions:", style={'fontWeight': 'bold', 'color': '#2c3e50'}),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button("Live Music vs No Music", id='demo-btn-1', color='primary', size='sm')
                                    ], width=12, className="mb-2"),
                                    dbc.Col([
                                        dbc.Button("Recorded Music vs No Music", id='demo-btn-2', color='primary', size='sm')
                                    ], width=12, className="mb-2"),
                                    dbc.Col([
                                        dbc.Button("Pink Noise vs No Music", id='demo-btn-3', color='primary', size='sm')
                                    ], width=12, className="mb-2")
                                ])
                            ])
                        ])
                    ], width=3)
                ] if sum(1 for v in self.data.values() if v['type'] == 'group') > 0 else [])
            ]),
            
            # Snapshot Gallery
            dbc.Row([
                dbc.Col([
                    html.H4("📸 Snapshot Gallery", style={'color': '#2c3e50', 'marginTop': '30px'}),
                    html.Div(id='snapshot-gallery')
                ])
            ], style={'marginTop': '20px'}),
            
            # Hidden div to store data
            html.Div(id='hidden-div', style={'display': 'none'})
            
        ], fluid=True)
        
        # Set up callbacks
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """Set up Dash callbacks for interactivity."""
        
        # Demo comparisons display
        @self.app.callback(
            Output('demo-comparisons', 'children'),
            [Input('condition1-dropdown', 'value')]  # Trigger on page load
        )
        def update_demo_comparisons(condition1):
            cam_icu_conditions = [k for k in self.data.keys() if 'CAM_ICU_Negative' in k]
            
            if len(cam_icu_conditions) < 2:
                return [html.P("No CAM ICU comparisons available", style={'color': '#6c757d'})]
            
            # Show available conditions
            demo_items = []
            for condition in sorted(cam_icu_conditions):
                condition_display = condition.replace('ICU_CAM_ICU_Negative_', '').replace('_', ' ')
                demo_items.append(
                    html.Li(f"✅ {condition_display}", style={'color': '#28a745'})
                )
            
            return [
                html.P(f"CAM ICU conditions loaded (n=3 participants each):", 
                       style={'fontSize': '12px', 'fontWeight': 'bold'}),
                html.Ul(demo_items, style={'fontSize': '12px', 'marginBottom': '10px'})
            ]
        
        # Demo button callbacks
        @self.app.callback(
            [Output('condition1-dropdown', 'value'),
             Output('condition2-dropdown', 'value')],
            [Input('demo-btn-1', 'n_clicks'),
             Input('demo-btn-2', 'n_clicks'),
             Input('demo-btn-3', 'n_clicks')],
            prevent_initial_call=True
        )
        def handle_demo_buttons(btn1, btn2, btn3):
            ctx = callback_context
            if not ctx.triggered:
                raise PreventUpdate
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # Find the appropriate CAM ICU conditions
            live_music = next((k for k in self.data.keys() if 'CAM_ICU_Negative_Live_Music' in k), None)
            recorded_music = next((k for k in self.data.keys() if 'CAM_ICU_Negative_Recorded_Music' in k), None)
            pink_noise = next((k for k in self.data.keys() if 'CAM_ICU_Negative_Pink_Noise' in k), None)
            no_music = next((k for k in self.data.keys() if 'CAM_ICU_Negative_No_Music' in k), None)
            
            if button_id == 'demo-btn-1' and live_music and no_music:
                return live_music, no_music
            elif button_id == 'demo-btn-2' and recorded_music and no_music:
                return recorded_music, no_music
            elif button_id == 'demo-btn-3' and pink_noise and no_music:
                return pink_noise, no_music
            else:
                raise PreventUpdate
        
        @self.app.callback(
            Output('system-status', 'children'),
            [Input('condition1-dropdown', 'value')]  # Trigger on page load
        )
        def update_system_status(condition1):
            status_badges = []
            
            # Main status badge - all conditions are group-level CAM ICU
            status_badges.append(
                dbc.Badge(f"✅ {len(self.data)} CAM ICU conditions loaded (group-level)", 
                         color="success", className="me-2")
            )
            
            # Warning badge if identical conditions
            if self.identical_conditions:
                identical_pairs = len(self.identical_conditions)
                status_badges.append(
                    dbc.Badge(f"⚠️ {identical_pairs} invalid comparison{'s' if identical_pairs > 1 else ''}", 
                             color="warning", className="me-2")
                )
            
            # Demo-ready badge
            cam_icu_negative_conditions = [k for k in self.data.keys() if 'CAM_ICU_Negative' in k]
            if len(cam_icu_negative_conditions) >= 2:
                status_badges.append(
                    dbc.Badge(f"Ready for CAM ICU Analysis!", 
                             color="info", className="me-2")
                )
            
            return status_badges
        
        @self.app.callback(
            Output('data-quality-warning', 'children'),
            [Input('condition1-dropdown', 'value'),
             Input('condition2-dropdown', 'value')]
        )
        def update_data_quality_warning(condition1, condition2):
            if not condition1 or not condition2:
                return []
            
            if not self.is_comparison_valid(condition1, condition2):
                return [
                    dbc.Alert([
                        html.H6("⚠️ Data Quality Warning", className="alert-heading"),
                        html.P([
                            f"Conditions '{condition1.replace('_', ' ')}' and '{condition2.replace('_', ' ')}' ",
                            "contain identical data. This comparison will show zero difference everywhere."
                        ]),
                        html.Hr(),
                        html.P([
                            "This usually indicates a data collection issue. ",
                            "Consider using different conditions for comparison."
                        ], className="mb-0")
                    ], color="warning", dismissable=True)
                ]
            
            return []

        @self.app.callback(
            [Output('vmin-input', 'value'),
             Output('vmax-input', 'value')],
            [Input('auto-scale-btn', 'n_clicks'),
             Input('percentile-input', 'value'),
             Input('condition1-dropdown', 'value')],
            prevent_initial_call=True
        )
        def auto_scale_thresholds(n_clicks, percentile, condition1):
            if not condition1 or percentile is None:
                raise PreventUpdate
            
            data = self.data[condition1]['data']
            # Use the percentile as a threshold - only show activations above this percentile
            vmin = np.percentile(data, percentile)  # Lower threshold
            vmax = np.percentile(data, 99)  # Keep max at 99th percentile
            
            return vmin, vmax
        
        @self.app.callback(
            [Output('brain-plot', 'figure'),
             Output('data-stats', 'children'),
             Output('roi-info', 'children')],
            [Input('update-btn', 'n_clicks'),
             Input('condition1-dropdown', 'value')],  # Added this to trigger on condition change
            [State('condition2-dropdown', 'value'),
             State('vmin-input', 'value'),
             State('vmax-input', 'value'),
             State('percentile-input', 'value'),
             State('threshold-method', 'value'),
             State('z-threshold-input', 'value'),
             State('p-threshold-input', 'value'),
             State('correction-checklist', 'value')]
        )
        def update_visualization(n_clicks, condition1, condition2, vmin, vmax, percentile,
                               threshold_method, z_threshold, p_threshold, correction_options):
            if not condition1:
                # Return empty/default values instead of raising error
                empty_fig = go.Figure()
                empty_fig.update_layout(title="Select a condition to begin")
                return empty_fig, [], []

            # Only check for parameter changes if the user explicitly clicked the update button
            # (not on initial load or condition change)
            ctx = callback_context
            if ctx.triggered and any("update-btn" in trigger["prop_id"] for trigger in ctx.triggered):
                # Check if parameters have changed from defaults
                default_percentile = 75
                default_z_threshold = 1.96
                default_p_threshold = 0.05
                default_threshold_method = 'percentile'
                
                # Get default vmin/vmax for current condition
                data = self.data[condition1]['data']
                default_vmin = np.percentile(data, default_percentile)
                default_vmax = np.percentile(data, 99)
                    
                # Check if any parameters have changed significantly
                parameters_changed = (
                    abs((vmin or default_vmin) - default_vmin) > 0.001 or
                    abs((vmax or default_vmax) - default_vmax) > 0.001 or
                    percentile != default_percentile or
                    threshold_method != default_threshold_method or
                    (threshold_method == 'z_score' and z_threshold != default_z_threshold) or
                    (threshold_method == 'p_value' and p_threshold != default_p_threshold)
                )
                
                # If parameters changed, trigger regeneration automatically
                if parameters_changed:
                    print(f"🔄 Parameters changed - triggering regeneration...")
                    print(f"   Condition: {condition1} vs {condition2 or 'None'}")
                    print(f"   Method: {threshold_method}")
                    print(f"   Parameters: vmin={vmin}, vmax={vmax}, percentile={percentile}")
                    
                    try:
                        # Save state and trigger restart
                        self.save_ui_state(condition1, condition2, vmin, vmax, percentile, threshold_method, z_threshold, p_threshold)
                        
                        print("🛑 Stopping web service for custom image generation...")
                        
                        # Start new process with custom parameters and different port
                        import subprocess
                        
                        subprocess.Popen([
                            sys.executable, __file__, '--restart-custom'
                        ], cwd=os.getcwd())
                        
                        print("🔄 Started restart process, shutting down current instance...")
                        
                        # Give the subprocess a moment to start
                        time.sleep(2)
                        
                        # Exit current process
                        os._exit(0)  # Force exit without cleanup
                        
                    except Exception as e:
                        print(f"❌ Error during regeneration: {e}")
                        # Fall through to normal visualization
            
            # Check if this is an invalid comparison (identical conditions)
            condition2 = condition2 if condition2 else None
            if condition2 and not self.is_comparison_valid(condition1, condition2):
                # Create warning figure for identical conditions
                warning_fig = go.Figure()
                warning_fig.add_annotation(
                    x=0.5, y=0.5,
                    text="⚠️ Warning: Identical Conditions<br><br>" +
                         f"'{condition1}' and '{condition2}' contain identical data.<br>" +
                         "This comparison will show zero difference everywhere.<br><br>" +
                         "Possible causes:<br>" +
                         "• Recording error (same data saved twice)<br>" +
                         "• Participant didn't change state<br>" +
                         "• Data corruption during transfer<br><br>" +
                         "Please select different conditions to compare.",
                    showarrow=False,
                    font=dict(size=14, color='#e74c3c'),
                    bgcolor='#ffeaa7',
                    bordercolor='#e74c3c',
                    borderwidth=2
                )
                warning_fig.update_layout(
                    title="Invalid Comparison: Identical Conditions",
                    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    plot_bgcolor='white',
                    height=500
                )
                
                # Return warning with empty stats
                warning_stats = [
                    html.P("⚠️ Invalid Comparison", style={'fontWeight': 'bold', 'color': '#e74c3c'}),
                    html.P("These conditions contain identical data.", style={'color': '#e74c3c'}),
                    html.P("Difference will be zero everywhere.", style={'color': '#e74c3c'})
                ]
                
                warning_roi = [
                    html.P("⚠️ No ROI Analysis", style={'fontWeight': 'bold', 'color': '#e74c3c'}),
                    html.P("Cannot analyze regions for identical conditions.", style={'color': '#e74c3c'})
                ]
                
                return warning_fig, warning_stats, warning_roi
            
            # Apply statistical thresholding based on method
            if threshold_method == 'z_score' and condition2:
                # Compute z-scores and apply thresholding
                z_scores, plot_data = self.compute_z_scores(condition1, condition2)
                
                # Apply FDR correction if requested
                if 'fdr' in correction_options:
                    from scipy import stats
                    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
                    if multipletests is not None:
                        fdr_corrected = multipletests(p_values, alpha=0.05, method='fdr_bh')
                        significant_mask = fdr_corrected[0]
                    else:
                        print("Warning: FDR correction not available. Using uncorrected p-values.")
                        significant_mask = p_values < 0.05
                    
                    # Only show significant voxels
                    plot_data_masked = plot_data.copy()
                    plot_data_masked[~significant_mask] = 0
                    
                    # Create the plot with masked data
                    fig = self.create_brain_plot_statistical(condition1, condition2, plot_data_masked, 
                                                            z_scores, threshold_method, z_threshold)
                    stats_title = f"Z-Score Analysis (FDR corrected): {condition1} - {condition2}"
                else:
                    # Apply z-score threshold
                    z_mask = np.abs(z_scores) >= z_threshold
                    plot_data_masked = plot_data.copy()
                    plot_data_masked[~z_mask] = 0
                    
                    fig = self.create_brain_plot_statistical(condition1, condition2, plot_data_masked, 
                                                            z_scores, threshold_method, z_threshold)
                    stats_title = f"Z-Score Analysis (|z| >= {z_threshold}): {condition1} - {condition2}"
                
            elif threshold_method == 'p_value' and condition2:
                # Compute z-scores and convert to p-values
                z_scores, plot_data = self.compute_z_scores(condition1, condition2)
                from scipy import stats
                p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
                
                # Apply FDR correction if requested
                if 'fdr' in correction_options:
                    if multipletests is not None:
                        fdr_corrected = multipletests(p_values, alpha=p_threshold, method='fdr_bh')
                        significant_mask = fdr_corrected[0]
                        stats_title = f"P-Value Analysis (FDR corrected, p < {p_threshold}): {condition1} - {condition2}"
                    else:
                        print("Warning: FDR correction not available. Using uncorrected p-values.")
                        significant_mask = p_values < p_threshold
                        stats_title = f"P-Value Analysis (uncorrected, p < {p_threshold}): {condition1} - {condition2}"
                else:
                    significant_mask = p_values < p_threshold
                    stats_title = f"P-Value Analysis (p < {p_threshold}): {condition1} - {condition2}"
                
                # Only show significant voxels
                plot_data_masked = plot_data.copy()
                plot_data_masked[~significant_mask] = 0
                
                fig = self.create_brain_plot_statistical(condition1, condition2, plot_data_masked, 
                                                        p_values, threshold_method, p_threshold)
                
            else:
                # Use default brain plot (loads pre-generated or custom images)
                fig = self.create_brain_plot(condition1, condition2, vmin, vmax)
                
                # Generate statistics for default method
                data1 = self.data[condition1]['data']
                if condition2:
                    plot_data = data1 - self.data[condition2]['data']
                    stats_title = f"Difference Statistics ({condition1} - {condition2})"
                else:
                    plot_data = data1
                    stats_title = f"Condition Statistics ({condition1})"
            
            stats = [
                html.P(f"{stats_title}", style={'fontWeight': 'bold'}),
                html.P(f"Data Type: {'Group-level' if self.data[condition1]['type'] == 'group' else 'Individual'}", 
                       style={'color': '#e74c3c' if self.data[condition1]['type'] == 'group' else '#3498db'}),
                html.P(f"Min: {np.min(plot_data):.4f}"),
                html.P(f"Max: {np.max(plot_data):.4f}"),
                html.P(f"Mean: {np.mean(plot_data):.4f}"),
                html.P(f"Std: {np.std(plot_data):.4f}"),
                html.P(f"Range: {np.max(plot_data) - np.min(plot_data):.4f}")
            ]
            
            # Add comparison type information
            if condition2:
                comparison_type = []
                cond1_type = self.data[condition1]['type']
                cond2_type = self.data[condition2]['type']
                
                if cond1_type == 'group' and cond2_type == 'group':
                    comparison_type.append(html.P("Group vs Group Comparison", style={'color': '#e74c3c', 'fontWeight': 'bold'}))
                elif cond1_type == 'individual' and cond2_type == 'individual':
                    comparison_type.append(html.P("Individual vs Individual Comparison", style={'color': '#3498db'}))
                else:
                    comparison_type.append(html.P("⚠️ Mixed Group/Individual Comparison", style={'color': '#f39c12', 'fontWeight': 'bold'}))
                
                stats.extend(comparison_type)
            
            # ROI information
            roi_stats = self.roi_info[condition1]
            roi_info = [
                html.P("Left Hemisphere:", style={'fontWeight': 'bold', 'color': '#e74c3c'}),
                html.P(f"Mean: {roi_stats['lh_mean']:.4f}"),
                html.P(f"Max: {roi_stats['lh_max']:.4f}"),
                html.P("Right Hemisphere:", style={'fontWeight': 'bold', 'color': '#3498db'}),
                html.P(f"Mean: {roi_stats['rh_mean']:.4f}"),
                html.P(f"Max: {roi_stats['rh_max']:.4f}")
            ]
            
            return fig, stats, roi_info
        
        @self.app.callback(
            Output('snapshot-gallery', 'children'),
            [Input('snapshot-btn', 'n_clicks')],
            [State('brain-plot', 'figure'),
             State('condition1-dropdown', 'value'),
             State('condition2-dropdown', 'value')]
        )
        def take_snapshot(n_clicks, figure, condition1, condition2):
            if not n_clicks:
                return []
            
            # Create snapshot info
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            snapshot_info = {
                'timestamp': timestamp,
                'condition1': condition1,
                'condition2': condition2,
                'figure': figure
            }
            
            self.snapshots.append(snapshot_info)
            
            # Create gallery display
            gallery_items = []
            for i, snapshot in enumerate(self.snapshots[-5:]):  # Show last 5 snapshots
                title = f"Snapshot {len(self.snapshots) - 5 + i + 1}: {snapshot['timestamp']}"
                conditions = f"{snapshot['condition1']}"
                if snapshot['condition2']:
                    conditions += f" vs {snapshot['condition2']}"
                
                gallery_items.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6(title, className="card-title"),
                                html.P(conditions, className="card-text"),
                                dcc.Graph(
                                    figure=snapshot['figure'],
                                    style={'height': '200px'},
                                    config={'displayModeBar': False}
                                )
                            ])
                        ])
                    ], width=2)
                )
            
            return dbc.Row(gallery_items)
        
        # Enable/disable threshold inputs based on method
        @self.app.callback(
            [Output('z-threshold-input', 'disabled'),
             Output('p-threshold-input', 'disabled'),
             Output('percentile-input', 'disabled')],
            [Input('threshold-method', 'value')]
        )
        def update_threshold_inputs(method):
            if method == 'z_score':
                return False, True, True
            elif method == 'p_value':
                return True, False, True
            else:  # percentile
                return True, True, False
        
        # ROI Summary Statistics
        @self.app.callback(
            Output('roi-summary-stats', 'children'),
            [Input('update-btn', 'n_clicks'),
             Input('condition1-dropdown', 'value')],
            [State('condition2-dropdown', 'value')]
        )
        def update_roi_summary(n_clicks, condition1, condition2):
            if not condition1:
                return []
            
            condition2 = condition2 if condition2 else None
            
            # Check for invalid comparison
            if condition2 and not self.is_comparison_valid(condition1, condition2):
                return [html.P("⚠️ Invalid comparison: identical conditions", 
                             style={'color': '#e74c3c', 'fontWeight': 'bold'})]
            
            roi_summary = self.compute_roi_summary(condition1, condition2)
            
            if not roi_summary:
                return [html.P("No ROI data available")]
            
            # Create summary table
            summary_components = []
            
            for roi, stats in roi_summary.items():
                roi_card = dbc.Card([
                    dbc.CardHeader(html.H6(roi.replace('_', ' '), style={'margin': '0'})),
                    dbc.CardBody([
                        html.P(f"Vertices: {stats['n_vertices']}", style={'margin': '2px 0'}),
                        html.P(f"Mean: {stats['mean']:.4f}", style={'margin': '2px 0'}),
                        html.P(f"Range: {stats['min']:.4f} to {stats['max']:.4f}", style={'margin': '2px 0'}),
                        
                        # Add statistical significance if comparison
                        *([html.P(f"Significant (p<0.05): {stats['percent_significant_p05']:.1f}%", 
                                 style={'margin': '2px 0', 'color': '#e74c3c' if stats['percent_significant_p05'] > 10 else '#2c3e50'}),
                          html.P(f"Max |Z|: {stats['max_z_score']:.2f}", 
                                 style={'margin': '2px 0'})]
                         if condition2 and 'max_z_score' in stats else [])
                    ])
                ], style={'marginBottom': '10px'})
                
                summary_components.append(roi_card)
            
            return summary_components
        
        # Parameter Change Detection
        @self.app.callback(
            [Output('parameter-status', 'children'),
             Output('regenerate-btn', 'style')],
            [Input('vmin-input', 'value'),
             Input('vmax-input', 'value'),
             Input('percentile-input', 'value'),
             Input('threshold-method', 'value'),
             Input('z-threshold-input', 'value'),
             Input('p-threshold-input', 'value'),
             Input('condition1-dropdown', 'value'),
             Input('condition2-dropdown', 'value')]
        )
        def detect_parameter_changes(vmin, vmax, percentile, threshold_method, z_threshold, p_threshold, condition1, condition2):
            # Check if parameters differ from defaults used in pre-generated images
            default_percentile = 75
            default_z_threshold = 1.96
            default_p_threshold = 0.05
            default_threshold_method = 'percentile'
            
            # Get default vmin/vmax for current condition
            if condition1:
                data = self.data[condition1]['data']
                default_vmin = np.percentile(data, default_percentile)
                default_vmax = np.percentile(data, 99)
            else:
                default_vmin, default_vmax = 0.0, 1.0
                
            # Check if any parameters have changed
            parameters_changed = (
                abs((vmin or 0) - default_vmin) > 0.001 or
                abs((vmax or 1) - default_vmax) > 0.001 or
                percentile != default_percentile or
                threshold_method != default_threshold_method or
                (threshold_method == 'z_score' and z_threshold != default_z_threshold) or
                (threshold_method == 'p_value' and p_threshold != default_p_threshold)
            )
            
            if parameters_changed:
                status_message = [
                    dbc.Alert([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        "⚡ Parameters changed from defaults. ",
                        html.Strong("Click 'Regenerate' to apply new settings.")
                    ], color="info", dismissable=False, className="mb-2")
                ]
                button_style = {'display': 'block'}  # Show regenerate button
            else:
                status_message = []
                button_style = {'display': 'none'}   # Hide regenerate button
                
            return status_message, button_style

        # Custom Image Regeneration with Restart
        @self.app.callback(
            Output('parameter-status', 'children', allow_duplicate=True),
            [Input('regenerate-btn', 'n_clicks')],
            [State('condition1-dropdown', 'value'),
             State('condition2-dropdown', 'value'),
             State('vmin-input', 'value'),
             State('vmax-input', 'value'),
             State('percentile-input', 'value'),
             State('threshold-method', 'value'),
             State('z-threshold-input', 'value'),
             State('p-threshold-input', 'value')],
            prevent_initial_call=True
        )
        def regenerate_with_restart(n_clicks, condition1, condition2, vmin, vmax, percentile, threshold_method, z_threshold, p_threshold):
            if not n_clicks or not condition1:
                raise PreventUpdate
                
            print(f"🔄 Initiating restart for custom image generation...")
            
            # Request restart with custom parameters
            try:
                # Use threading to avoid blocking the callback
                restart_thread = threading.Thread(
                    target=self.request_restart_with_custom_images,
                    args=(condition1, condition2, vmin, vmax, percentile, threshold_method, z_threshold, p_threshold)
                )
                restart_thread.daemon = True
                restart_thread.start()
                
                return [
                    dbc.Alert([
                        html.I(className="fas fa-sync-alt fa-spin me-2"),
                        html.Strong("🔄 Regenerating with custom parameters..."),
                        html.Br(),
                        "The application will restart momentarily with your new brain images."
                    ], color="info", dismissable=False)
                ]
                
            except Exception as e:
                print(f"❌ Error initiating restart: {e}")
                return [
                    dbc.Alert([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        html.Strong("❌ Restart failed"),
                        html.Br(),
                        f"Error: {str(e)}"
                    ], color="danger", dismissable=True)
                ]

        # Data Export
        @self.app.callback(
            Output('download-link', 'children'),
            [Input('export-data-btn', 'n_clicks')],
            [State('condition1-dropdown', 'value'),
             State('condition2-dropdown', 'value'),
             State('export-format', 'value'),
             State('export-options', 'value')]
        )
        def export_data_callback(n_clicks, condition1, condition2, export_format, export_options):
            if not n_clicks or not condition1:
                return []
            
            condition2 = condition2 if condition2 else None
            include_coords = 'coords' in export_options
            include_stats = 'stats' in export_options
            
            # Check for invalid comparison
            if condition2 and not self.is_comparison_valid(condition1, condition2):
                return [
                    dbc.Alert("⚠️ Cannot export: identical conditions selected", 
                             color="warning", dismissable=True)
                ]
            
            try:
                # Export the data
                export_data = self.export_data(
                    condition1, condition2, 
                    include_coords=include_coords,
                    include_stats=include_stats,
                    export_format=export_format
                )
                
                # Create filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if condition2:
                    filename = f"brain_comparison_{condition1}_vs_{condition2}_{timestamp}.{export_format}"
                    analysis_type = "comparison"
                else:
                    filename = f"brain_analysis_{condition1}_{timestamp}.{export_format}"
                    analysis_type = "single_condition"
                
                # Convert to appropriate format
                if export_format == 'csv':
                    import pandas as pd
                    df = pd.DataFrame(export_data)
                    csv_string = df.to_csv(index=False)
                    
                    # Create download link
                    csv_base64 = base64.b64encode(csv_string.encode()).decode()
                    download_url = f"data:text/csv;base64,{csv_base64}"
                    
                elif export_format == 'json':
                    # Convert numpy types to python types for JSON serialization
                    json_data = {}
                    for key, value in export_data.items():
                        if isinstance(value, np.ndarray):
                            json_data[key] = value.tolist()
                        elif isinstance(value, np.number):
                            json_data[key] = value.item()
                        else:
                            json_data[key] = value
                    
                    json_string = json.dumps(json_data, indent=2)
                    json_base64 = base64.b64encode(json_string.encode()).decode()
                    download_url = f"data:application/json;base64,{json_base64}"
                
                # Return download link
                return [
                    dbc.Alert([
                        html.H6("✅ Export Ready", className="alert-heading"),
                        html.P(f"Analysis type: {analysis_type}"),
                        html.P(f"Data points: {len(export_data['vertex_id'])}"),
                        html.P(f"Includes: {', '.join(['coordinates' if include_coords else '', 'statistics' if include_stats else '']).strip(', ')}"),
                        html.A(
                            "📥 Download File",
                            id="download-link-a",
                            download=filename,
                            href=download_url,
                            className="btn btn-success btn-sm",
                            style={'marginTop': '10px'}
                        )
                    ], color="success", dismissable=True)
                ]
                
            except Exception as e:
                return [
                    dbc.Alert(f"❌ Export failed: {str(e)}", color="danger", dismissable=True)
                ]
    
    def run(self, debug=True, host='127.0.0.1', port=8050):
        """Run the web application."""
        
        # Find available port if default is busy
        available_port = self.find_available_port(port)
        if available_port != port:
            print(f"🔄 Port {port} busy, using port {available_port}")
            port = available_port
        
        print(f"🚀 Starting Modern Brain Explorer (CAM ICU Focus)...")
        print(f"📱 Open your browser to: http://{host}:{port}")
        print(f"Loaded {len(self.data)} CAM ICU conditions (group-level)")
        print(f"✨ Statistical Features:")
        print(f"   📊 Z-score and P-value thresholding")
        print(f"   🔬 Multiple comparison correction (FDR)")
        print(f"   📋 ROI-based statistical summaries")
        print(f"   💾 Data export with coordinates & statistics")
        print(f"📈 Export formats: CSV, JSON with 3D coordinates")
        if self.identical_conditions:
            print(f"⚠️  Warning: {len(self.identical_conditions)} identical condition pairs detected")
        
        try:
            self.app.run(debug=debug, host=host, port=port)
        except OSError as e:
            if "Address already in use" in str(e) or "Socket operation" in str(e):
                print(f"⚠️ Port {port} conflict detected, trying different port...")
                new_port = self.find_available_port(port + 1)
                print(f"🔄 Retrying on port {new_port}")
                self.app.run(debug=debug, host=host, port=new_port)
            else:
                raise e

    def save_ui_state(self, condition1, condition2, vmin, vmax, percentile, threshold_method, z_threshold, p_threshold):
        """Save current UI state for restart."""
        state = {
            'condition1': condition1,
            'condition2': condition2,
            'vmin': vmin,
            'vmax': vmax,
            'percentile': percentile,
            'threshold_method': threshold_method,
            'z_threshold': z_threshold,
            'p_threshold': p_threshold,
            'timestamp': time.time()
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"💾 Saved UI state: {condition1} vs {condition2 or 'None'}")

    def load_ui_state(self):
        """Load saved UI state after restart."""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Check if state is recent (within last 5 minutes)
            if time.time() - state['timestamp'] < 300:  # 5 minutes
                print(f"📂 Loaded UI state: {state['condition1']} vs {state.get('condition2') or 'None'}")
                return state
            else:
                print("⏰ Saved state is too old, using defaults")
                return None
                
        except (FileNotFoundError, json.JSONDecodeError):
            print("📄 No saved state found, using defaults")
            return None

    def generate_custom_images(self, condition1, condition2, vmin, vmax, percentile, threshold_method, z_threshold, p_threshold):
        """Generate custom brain images with specified parameters using PyVista."""
        
        # Create custom images directory
        os.makedirs(self.custom_images_dir, exist_ok=True)
        
        print(f"🎨 Generating custom brain images...")
        print(f"   Condition: {condition1} vs {condition2 or 'None'}")
        print(f"   Parameters: vmin={vmin}, vmax={vmax}, percentile={percentile}")
        print(f"   Method: {threshold_method}")
        
        try:
            # Generate images with custom parameters
            if condition2:
                # Generate difference images
                self._generate_custom_difference_images(condition1, condition2, vmin, vmax, percentile, threshold_method, z_threshold, p_threshold)
            else:
                # Generate single condition images
                self._generate_custom_condition_images(condition1, vmin, vmax, percentile, threshold_method, z_threshold, p_threshold)
            
            print("✅ Custom images generated successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error generating custom images: {e}")
            return False

    def _generate_custom_condition_images(self, condition, vmin, vmax, percentile, threshold_method, z_threshold, p_threshold):
        """Generate custom images for a single condition."""
        
        data = self.data[condition]['data']
        vertices = self.data[condition]['vertices']
        
        # Apply thresholding based on method
        if threshold_method == 'percentile':
            if vmin is None or vmax is None:
                vmin = np.percentile(data, percentile)
                vmax = np.percentile(data, 99)
            plot_data = data.copy()
            plot_data[plot_data < vmin] = 0
        else:
            plot_data = data.copy()
            if vmin is None: vmin = np.min(plot_data)
            if vmax is None: vmax = np.max(plot_data)
        
        # Create source estimate
        stc = mne.SourceEstimate(
            plot_data,
            vertices=vertices,
            tmin=0,
            tstep=1,
            subject=self.subject
        )
        
        # Generate images for both hemispheres
        for hemi in ['lh', 'rh']:
            filename = f"{condition}_{hemi}_custom.png"
            filepath = os.path.join(self.custom_images_dir, filename)
            
            brain = mne.viz.plot_source_estimates(
                stc,
                subject=self.subject,
                subjects_dir=self.subjects_dir,
                hemi=hemi,
                views='lat',
                backend='auto',
                time_viewer=False,
                clim=dict(kind='value', lims=[vmin, (vmin+vmax)/2, vmax]),
                colorbar=False,
                background='white',
                size=(800, 600)
            )
            
            img_array = brain.screenshot()
            brain.close()
            
            from PIL import Image
            img = Image.fromarray(img_array)
            img.save(filepath)
            
            print(f"   ✅ Generated {filename}")

    def _generate_custom_difference_images(self, condition1, condition2, vmin, vmax, percentile, threshold_method, z_threshold, p_threshold):
        """Generate custom difference images between two conditions."""
        
        data1 = self.data[condition1]['data']
        data2 = self.data[condition2]['data']
        vertices = self.data[condition1]['vertices']
        
        # Apply statistical analysis based on method
        if threshold_method == 'z_score':
            z_scores, plot_data = self.compute_z_scores(condition1, condition2)
            z_mask = np.abs(z_scores) >= z_threshold
            plot_data[~z_mask] = 0
            
            if vmin is None or vmax is None:
                abs_max = np.max(np.abs(plot_data))
                vmin, vmax = -abs_max, abs_max
                
        elif threshold_method == 'p_value':
            z_scores, plot_data = self.compute_z_scores(condition1, condition2)
            from scipy import stats
            p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
            p_mask = p_values < p_threshold
            plot_data[~p_mask] = 0
            
            if vmin is None or vmax is None:
                abs_max = np.max(np.abs(plot_data))
                vmin, vmax = -abs_max, abs_max
                
        else:  # percentile
            plot_data = data1 - data2
            if vmin is None or vmax is None:
                threshold_val = np.percentile(np.abs(plot_data), percentile)
                plot_data[np.abs(plot_data) < threshold_val] = 0
                abs_max = np.max(np.abs(plot_data))
                vmin, vmax = -abs_max, abs_max
        
        # Create source estimate
        stc = mne.SourceEstimate(
            plot_data,
            vertices=vertices,
            tmin=0,
            tstep=1,
            subject=self.subject
        )
        
        # Generate images for both hemispheres
        for hemi in ['lh', 'rh']:
            filename = f"{condition1}_minus_{condition2}_{hemi}_custom.png"
            filepath = os.path.join(self.custom_images_dir, filename)
            
            brain = mne.viz.plot_source_estimates(
                stc,
                subject=self.subject,
                subjects_dir=self.subjects_dir,
                hemi=hemi,
                views='lat',
                backend='auto',
                time_viewer=False,
                clim=dict(kind='value', lims=[vmin, 0, vmax]),
                colorbar=False,
                background='white',
                size=(800, 600)
            )
            
            img_array = brain.screenshot()
            brain.close()
            
            from PIL import Image
            img = Image.fromarray(img_array)
            img.save(filepath)
            
            print(f"   ✅ Generated {filename}")

    def create_brain_plot_custom(self, condition1, condition2=None):
        """Create brain plot using custom generated images."""
        
        # Determine which images to load
        if condition2:
            lh_filename = f"{condition1}_minus_{condition2}_lh_custom.png"
            rh_filename = f"{condition1}_minus_{condition2}_rh_custom.png"
            title = f"Custom Analysis: {condition1} - {condition2}"
        else:
            lh_filename = f"{condition1}_lh_custom.png"
            rh_filename = f"{condition1}_rh_custom.png"
            title = f"Custom Analysis: {condition1}"
        
        # Load custom images
        lh_path = os.path.join(self.custom_images_dir, lh_filename)
        rh_path = os.path.join(self.custom_images_dir, rh_filename)
        
        if not os.path.exists(lh_path) or not os.path.exists(rh_path):
            # Fallback to regular images if custom don't exist
            return self.create_brain_plot(condition1, condition2)
        
        # Load and encode images
        with open(lh_path, 'rb') as f:
            lh_base64 = base64.b64encode(f.read()).decode()
        
        with open(rh_path, 'rb') as f:
            rh_base64 = base64.b64encode(f.read()).decode()
        
        # Create figure
        fig = go.Figure()
        
        # Add left hemisphere image
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{lh_base64}",
                xref="paper", yref="paper",
                x=0, y=1,
                sizex=0.48, sizey=1,
                xanchor="left", yanchor="top",
                layer="below"
            )
        )
        
        # Add right hemisphere image
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{rh_base64}",
                xref="paper", yref="paper",
                x=0.52, y=1,
                sizex=0.48, sizey=1,
                xanchor="left", yanchor="top",
                layer="below"
            )
        )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16, color='#2c3e50')),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=50, b=0),
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        print(f"✅ Loaded custom brain visualization: {title}")
        return fig

    def request_restart_with_custom_images(self, condition1, condition2, vmin, vmax, percentile, threshold_method, z_threshold, p_threshold):
        """Request restart with custom image generation."""
        
        # Save current UI state
        self.save_ui_state(condition1, condition2, vmin, vmax, percentile, threshold_method, z_threshold, p_threshold)
        
        print("🛑 Stopping web service for custom image generation...")
        
        # Start new process with custom parameters
        import subprocess
        
        try:
            # Start the restart process
            subprocess.Popen([
                sys.executable, __file__, '--restart-custom'
            ], cwd=os.getcwd())
            
            print("🔄 Started restart process, shutting down current instance...")
            
            # Give the subprocess a moment to start
            time.sleep(1)
            
            # Exit current process
            os._exit(0)  # Force exit without cleanup
            
        except Exception as e:
            print(f"❌ Error starting restart process: {e}")
            # Fallback - just exit
            os._exit(1)

    def find_available_port(self, start_port=8050, max_attempts=10):
        """Find an available port starting from start_port."""
        
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
                    return port
            except OSError:
                continue
        
        # If no port found, return default
        return start_port

def main():
    """Main entry point with restart capability."""
    
    # Check if this is a restart with custom parameters
    if len(sys.argv) > 1 and sys.argv[1] == '--restart-custom':
        print("🔄 Restarting with custom image generation...")
        
        # Initialize explorer
        explorer = ModernBrainExplorer()
        
        # Load saved state
        state = explorer.load_ui_state()
        
        if state:
            # Generate custom images with saved parameters
            success = explorer.generate_custom_images(
                state['condition1'], 
                state.get('condition2'),
                state['vmin'], 
                state['vmax'], 
                state['percentile'],
                state['threshold_method'],
                state['z_threshold'],
                state['p_threshold']
            )
            
            if success:
                print("✅ Custom images generated successfully!")
                # Set flag to use custom images
                explorer.use_custom_images = True
                
                # Start the app
                explorer.run()
            else:
                print("❌ Failed to generate custom images")
        else:
            print("❌ No saved state found")
            
    else:
        print("🧠 Starting Modern Brain Explorer (CAM ICU Focus)...")
        
        # Initialize explorer
        explorer = ModernBrainExplorer()
        
        # Pre-generate ONLY the CAM ICU comparison images
        try:
            explorer.generate_brain_images()
        except Exception as e:
            print(f"❌ Error during image generation: {e}")
            print("⚠️  Continuing with existing images...")
        
        # Start the web app
        print("\n🚀 Starting web application...")
        explorer.run()

if __name__ == "__main__":
    main() 