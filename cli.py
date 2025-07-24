#!/usr/bin/env python3
"""
Command-Line Interface for Brain Source Localization Analysis Platform
Provides batch processing and automation capabilities
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BrainAnalysisCLI:
    """Command-line interface for brain analysis operations."""
    
    def __init__(self):
        self.explorer = None
    
    def init_explorer(self):
        """Initialize the brain explorer."""
        if self.explorer is None:
            from modern_brain_explorer import ModernBrainExplorer
            self.explorer = ModernBrainExplorer(skip_image_generation=False)
        return self.explorer
    
    def list_conditions(self):
        """List all available conditions."""
        explorer = self.init_explorer()
        
        print("🧠 Available Brain Conditions:")
        print("="*50)
        
        for condition in sorted(explorer.data.keys()):
            condition_type = explorer.data[condition]['type']
            data_shape = explorer.data[condition]['data'].shape
            
            print(f"✅ {condition}")
            print(f"   Type: {condition_type}")
            print(f"   Data points: {data_shape}")
            print()
    
    def batch_export(self, output_dir="batch_exports", format="csv", include_coords=True, include_stats=True):
        """Export all possible condition comparisons."""
        explorer = self.init_explorer()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"📦 Starting batch export to: {output_path}")
        
        # Get CAM ICU conditions
        cam_icu_conditions = [k for k in explorer.data.keys() if 'CAM_ICU_Negative' in k]
        
        if len(cam_icu_conditions) < 2:
            print("❌ Need at least 2 CAM ICU conditions for batch export")
            return
        
        # Find baseline condition
        baseline = None
        for condition in cam_icu_conditions:
            if 'No_Music' in condition:
                baseline = condition
                break
        
        if not baseline:
            print("⚠️  No baseline (No_Music) condition found, using first condition as baseline")
            baseline = cam_icu_conditions[0]
        
        # Export all comparisons vs baseline
        exported_count = 0
        for condition in cam_icu_conditions:
            if condition == baseline:
                continue
                
            try:
                print(f"📊 Exporting: {condition} vs {baseline}")
                
                # Generate safe filename
                safe_cond = condition.replace('ICU_CAM_ICU_Negative_', '').replace('_', '-')
                safe_base = baseline.replace('ICU_CAM_ICU_Negative_', '').replace('_', '-')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                filename = f"comparison_{safe_cond}_vs_{safe_base}_{timestamp}.{format}"
                filepath = output_path / filename
                
                # Export data
                export_data = explorer.export_data(
                    condition, baseline,
                    include_coords=include_coords,
                    include_stats=include_stats,
                    export_format=format
                )
                
                # Save to file
                if format == 'csv':
                    import pandas as pd
                    df = pd.DataFrame(export_data)
                    df.to_csv(filepath, index=False)
                elif format == 'json':
                    # Convert numpy arrays to lists for JSON serialization
                    json_data = {}
                    for key, value in export_data.items():
                        if hasattr(value, 'tolist'):
                            json_data[key] = value.tolist()
                        else:
                            json_data[key] = value
                    
                    with open(filepath, 'w') as f:
                        json.dump(json_data, f, indent=2)
                
                print(f"✅ Exported: {filename}")
                exported_count += 1
                
            except Exception as e:
                print(f"❌ Failed to export {condition} vs {baseline}: {e}")
        
        print(f"\n🎉 Batch export complete! Exported {exported_count} comparisons to {output_path}")
    
    def generate_images(self, output_dir="brain_images", force=False):
        """Generate brain images for all conditions and comparisons."""
        explorer = self.init_explorer()
        
        if force or not Path(output_dir).exists():
            print(f"🎨 Generating brain images to: {output_dir}")
            explorer.generate_brain_images(output_dir)
        else:
            print(f"⚡ Images already exist in {output_dir}. Use --force to regenerate.")
    
    def run_analysis(self, condition1, condition2=None, output_file=None, statistical_method="percentile", threshold=75):
        """Run a specific analysis and save results."""
        explorer = self.init_explorer()
        
        if condition1 not in explorer.data:
            print(f"❌ Condition '{condition1}' not found")
            self.list_conditions()
            return
        
        if condition2 and condition2 not in explorer.data:
            print(f"❌ Condition '{condition2}' not found")
            self.list_conditions()
            return
        
        print(f"🔬 Running analysis: {condition1}" + (f" vs {condition2}" if condition2 else ""))
        
        # Perform statistical analysis
        if condition2:
            if statistical_method == "z_score":
                z_scores, diff_data = explorer.compute_z_scores(condition1, condition2)
                results = {
                    'analysis_type': 'z_score_comparison',
                    'condition1': condition1,
                    'condition2': condition2,
                    'z_scores': z_scores.tolist(),
                    'difference_data': diff_data.tolist(),
                    'significant_vertices': int((abs(z_scores) >= 1.96).sum()),
                    'threshold': 1.96
                }
            else:
                # Standard difference analysis
                data1 = explorer.data[condition1]['data']
                data2 = explorer.data[condition2]['data']
                diff_data = data1 - data2
                
                results = {
                    'analysis_type': 'difference_comparison',
                    'condition1': condition1,
                    'condition2': condition2,
                    'difference_data': diff_data.tolist(),
                    'mean_difference': float(diff_data.mean()),
                    'max_difference': float(diff_data.max()),
                    'min_difference': float(diff_data.min())
                }
        else:
            # Single condition analysis
            data = explorer.data[condition1]['data']
            roi_stats = explorer.roi_info[condition1]
            
            # Convert NumPy types in roi_stats to JSON-serializable types
            json_roi_stats = {}
            for key, value in roi_stats.items():
                if hasattr(value, 'tolist'):
                    json_roi_stats[key] = value.tolist()
                elif isinstance(value, (int, float)):
                    json_roi_stats[key] = float(value)
                else:
                    json_roi_stats[key] = value
            
            results = {
                'analysis_type': 'single_condition',
                'condition': condition1,
                'data': data.tolist(),
                'roi_statistics': json_roi_stats,
                'data_type': explorer.data[condition1]['type']
            }
        
        # Add metadata
        results['timestamp'] = datetime.now().isoformat()
        results['software_version'] = '2.0.0'
        
        # Save results
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ Analysis results saved to: {output_path}")
        else:
            # Print summary to console
            print("\n📊 Analysis Summary:")
            print("="*30)
            if 'mean_difference' in results:
                print(f"Mean difference: {results['mean_difference']:.4f}")
                print(f"Max difference: {results['max_difference']:.4f}")
                print(f"Min difference: {results['min_difference']:.4f}")
            if 'significant_vertices' in results:
                print(f"Significant vertices: {results['significant_vertices']}")
    
    def web_server(self, host="127.0.0.1", port=8050, debug=False):
        """Start the web interface."""
        explorer = self.init_explorer()
        
        print(f"🚀 Starting web server at http://{host}:{port}")
        explorer.run(host=host, port=port, debug=debug)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Brain Source Localization Analysis Platform - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                              # List all conditions
  %(prog)s web                               # Start web interface
  %(prog)s analyze cond1 cond2 -o results.json  # Run specific analysis
  %(prog)s batch-export --format csv         # Export all comparisons
  %(prog)s generate-images --force           # Regenerate all brain images
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List conditions
    list_parser = subparsers.add_parser('list', help='List all available conditions')
    
    # Web server
    web_parser = subparsers.add_parser('web', help='Start web interface')
    web_parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    web_parser.add_argument('--port', type=int, default=8050, help='Port to bind to')
    web_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Analysis
    analyze_parser = subparsers.add_parser('analyze', help='Run specific analysis')
    analyze_parser.add_argument('condition1', help='Primary condition')
    analyze_parser.add_argument('condition2', nargs='?', help='Comparison condition (optional)')
    analyze_parser.add_argument('-o', '--output', help='Output file for results')
    analyze_parser.add_argument('--method', choices=['percentile', 'z_score'], 
                               default='percentile', help='Statistical method')
    analyze_parser.add_argument('--threshold', type=float, default=75, 
                               help='Threshold value')
    
    # Batch export
    batch_parser = subparsers.add_parser('batch-export', help='Export all comparisons')
    batch_parser.add_argument('--output-dir', default='batch_exports', 
                             help='Output directory')
    batch_parser.add_argument('--format', choices=['csv', 'json'], default='csv',
                             help='Export format')
    batch_parser.add_argument('--no-coords', action='store_true',
                             help='Skip coordinate data')
    batch_parser.add_argument('--no-stats', action='store_true',
                             help='Skip statistical data')
    
    # Generate images
    image_parser = subparsers.add_parser('generate-images', help='Generate brain images')
    image_parser.add_argument('--output-dir', default='brain_images',
                             help='Output directory for images')
    image_parser.add_argument('--force', action='store_true',
                             help='Force regeneration of existing images')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = BrainAnalysisCLI()
    
    try:
        if args.command == 'list':
            cli.list_conditions()
        
        elif args.command == 'web':
            cli.web_server(host=args.host, port=args.port, debug=args.debug)
        
        elif args.command == 'analyze':
            cli.run_analysis(
                args.condition1, args.condition2, 
                args.output, args.method, args.threshold
            )
        
        elif args.command == 'batch-export':
            cli.batch_export(
                args.output_dir, args.format,
                include_coords=not args.no_coords,
                include_stats=not args.no_stats
            )
        
        elif args.command == 'generate-images':
            cli.generate_images(args.output_dir, args.force)
        
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 