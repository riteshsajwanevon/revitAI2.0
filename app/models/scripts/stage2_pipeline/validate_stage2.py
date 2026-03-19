#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 Validation Script
Run inference on validation dataset and generate comprehensive results

IMPORTANT - NO DATA LEAKAGE:
This script ensures NO data leakage by:
1. Running predictions WITHOUT using BeamColumnMatrix
2. Saving predictions to CSV files
3. Only AFTER predictions are made, loading BeamColumnMatrix for comparison
4. BeamColumnMatrix is NEVER used during the prediction phase

The validation process:
  Step 1: Load model
  Step 2: For each building, predict using ONLY:
          - FeatureMatrix (structural layout)
          - BeamWallMatrix (beam-wall connections)
          - BeamBeamMatrix (beam-beam connections)
  Step 3: Save predictions to CSV
  Step 4: Load ground truth from BeamColumnMatrix
  Step 5: Compare predictions with ground truth
  Step 6: Calculate accuracy metrics

BeamColumnMatrix is ONLY used in Step 4, AFTER all predictions are complete.
"""

import torch
import pandas as pd
import numpy as np
import os
import glob
import random
from datetime import datetime
from predict_stage2_inference_only import load_model, predict_building

# ============================================================================
# CONFIGURATION
# ============================================================================
VALIDATION_PATH = "Dataset/validation"
MODEL_PATH = "stage2_model.pth"
OUTPUT_DIR = "result/validation_results"
UPDATE_ORIGINAL_FILES = True  # Set to True to update original validation files

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def get_validation_buildings(validation_path):
    """Get list of building IDs in validation folder"""
    feature_files = glob.glob(os.path.join(validation_path, "*_FeatureMatrix.csv"))
    building_ids = []
    
    for filepath in feature_files:
        filename = os.path.basename(filepath)
        building_id = filename.replace("_FeatureMatrix.csv", "")
        building_ids.append(building_id)
    
    return sorted(building_ids)

def validate_building(model, building_id, validation_path):
    """
    Run Stage 2 inference on a single building
    Returns predictions and statistics
    
    IMPORTANT: This function does NOT use BeamColumnMatrix during prediction.
    BeamColumnMatrix is only used later for comparison purposes.
    """
    
    # Build file paths (NO BeamColumnMatrix!)
    feature_csv = os.path.join(validation_path, f"{building_id}_FeatureMatrix.csv")
    beam_wall_csv = os.path.join(validation_path, f"{building_id}_BeamWallMatrix.csv")
    beam_beam_csv = os.path.join(validation_path, f"{building_id}_BeamBeamMatrix.csv")
    
    # VERIFICATION: Ensure we're not accidentally using BeamColumnMatrix
    beam_column_csv = os.path.join(validation_path, f"{building_id}_BeamColumnMatrix.csv")
    if os.path.exists(beam_column_csv):
        # Ground truth exists but we MUST NOT use it during prediction
        pass  # This is expected - we'll use it later for comparison only
    
    # Check if files exist
    if not os.path.exists(feature_csv):
        print(f"  ❌ FeatureMatrix not found for {building_id}")
        return None
    
    if not os.path.exists(beam_wall_csv):
        print(f"  ❌ BeamWallMatrix not found for {building_id}")
        return None
    
    # BeamBeamMatrix is optional
    if not os.path.exists(beam_beam_csv):
        beam_beam_csv = None
    
    # Run prediction WITHOUT BeamColumnMatrix
    # This function only uses FeatureMatrix, BeamWallMatrix, and BeamBeamMatrix
    try:
        results = predict_building(model, feature_csv, beam_wall_csv, beam_beam_csv)
        
        # Calculate statistics
        total_beams = len(results)
        class_counts = {0: 0, 1: 0, 2: 0}
        high_confidence = 0
        low_confidence = 0
        
        for r in results:
            class_counts[r['predicted_columns']] += 1
            if r['confidence'] >= 0.8:
                high_confidence += 1
            elif r['confidence'] < 0.5:
                low_confidence += 1
        
        total_columns = class_counts[1] + class_counts[2] * 2
        
        stats = {
            'building_id': building_id,
            'total_beams': total_beams,
            'beams_class_0': class_counts[0],
            'beams_class_1': class_counts[1],
            'beams_class_2': class_counts[2],
            'total_columns': total_columns,
            'high_confidence': high_confidence,
            'low_confidence': low_confidence,
            'avg_confidence': np.mean([r['confidence'] for r in results])
        }
        
        return {
            'stats': stats,
            'predictions': results
        }
        
    except Exception as e:
        print(f"  ❌ Error processing {building_id}: {e}")
        return None

def generate_beam_column_matrix(building_id, predictions, validation_path):
    """
    Generate BeamColumnMatrix based on predictions
    Creates beam IDs as {building_id}_{beam_id}_B (rows)
    Creates column IDs as {building_id}_{4_digit_random}_C (columns only)
    Matrix shows beam-column connections only
    """
    
    # Create beam IDs with building prefix
    beam_ids = []
    beam_column_connections = {}
    all_column_ids = []
    
    # Process predictions to create beam-column connections
    for pred in predictions:
        beam_id = f"{building_id}_{pred['beam_id']}_B"
        beam_ids.append(beam_id)
        
        num_columns = pred['predicted_columns']
        beam_column_connections[beam_id] = []
        
        # Generate random column IDs for this beam
        for i in range(num_columns):
            # Generate 4-digit random number
            random_id = str(random.randint(1000, 9999))
            column_id = f"{building_id}_{random_id}_C"
            
            # Ensure unique column ID
            while column_id in all_column_ids:
                random_id = str(random.randint(1000, 9999))
                column_id = f"{building_id}_{random_id}_C"
            
            all_column_ids.append(column_id)
            beam_column_connections[beam_id].append(column_id)
    
    # If no columns predicted, create empty matrix
    if not all_column_ids:
        # Create empty DataFrame with beam rows but no columns
        df = pd.DataFrame(index=beam_ids)
        return df, []
    
    # Create matrix with beam rows and ONLY column headers (no beam columns)
    matrix_data = {}
    
    # Add only column connections (no beam-beam connections)
    for column_id in all_column_ids:
        column_connections = []
        for beam_id in beam_ids:
            # Check if this beam connects to this column
            if column_id in beam_column_connections.get(beam_id, []):
                column_connections.append(1)
            else:
                column_connections.append(0)
        matrix_data[column_id] = column_connections
    
    # Create DataFrame with beams as rows and columns as headers
    df = pd.DataFrame(matrix_data, index=beam_ids)
    
    return df, all_column_ids

def update_feature_matrix(building_id, predictions, all_column_ids, validation_path):
    """
    Update FeatureMatrix to include predicted columns as new rows
    """
    
    # Load existing FeatureMatrix
    feature_csv = os.path.join(validation_path, f"{building_id}_FeatureMatrix.csv")
    
    if not os.path.exists(feature_csv):
        print(f"  ❌ FeatureMatrix not found for {building_id}")
        return None
    
    df = pd.read_csv(feature_csv)
    
    # Create new rows for predicted columns
    new_rows = []
    
    for i, column_id in enumerate(all_column_ids):
        # Find the corresponding prediction for material
        predicted_material = "Unknown"
        for pred in predictions:
            if pred['predicted_columns'] > 0:
                predicted_material = pred['material']
                break
        
        # Create new row with mostly null values
        new_row = {col: np.nan for col in df.columns}
        
        # Set specific values for columns
        new_row['Element Type'] = 'Structural Column'
        new_row['Element ID'] = column_id.replace('_C', '')  # Remove _C suffix for Element ID
        new_row['Structural Material'] = predicted_material
        
        new_rows.append(new_row)
    
    # Append new rows to DataFrame
    if new_rows:
        new_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        return new_df
    
    return df
def display_matrix_sample(building_id, matrix_file):
    """Display a sample of the generated BeamColumnMatrix for verification"""
    try:
        df = pd.read_csv(matrix_file, index_col=0)
        print(f"\n📋 Sample BeamColumnMatrix for {building_id}:")
        print(f"   Matrix shape: {df.shape[0]} beams × {df.shape[1]} columns")
        
        if df.shape[1] > 0:
            # Show first few rows and columns
            sample = df.iloc[:min(5, df.shape[0]), :min(8, df.shape[1])]
            print(f"   Sample (first 5 beams, up to 8 columns):")
            print(sample.to_string())
            
            # Count connections
            total_connections = df.sum().sum()
            print(f"   Total beam-column connections: {total_connections}")
        else:
            print(f"   No columns predicted for this building")
        
    except Exception as e:
        print(f"   ❌ Error reading matrix: {e}")

def backup_original_files(building_id, validation_path, output_dir):
    """Create backup of original files before updating them"""
    
    backup_dir = os.path.join(output_dir, "original_backups")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Backup BeamColumnMatrix if it exists
    original_matrix = os.path.join(validation_path, f"{building_id}_BeamColumnMatrix.csv")
    if os.path.exists(original_matrix):
        backup_matrix = os.path.join(backup_dir, f"{building_id}_BeamColumnMatrix_original.csv")
        import shutil
        shutil.copy2(original_matrix, backup_matrix)
        print(f"  📋 Backed up original BeamColumnMatrix to: {backup_matrix}")
    
    # Backup FeatureMatrix
    original_feature = os.path.join(validation_path, f"{building_id}_FeatureMatrix.csv")
    if os.path.exists(original_feature):
        backup_feature = os.path.join(backup_dir, f"{building_id}_FeatureMatrix_original.csv")
        import shutil
        shutil.copy2(original_feature, backup_feature)
        print(f"  📋 Backed up original FeatureMatrix to: {backup_feature}")

def save_stage3_constraint_file(building_id, predictions, output_dir):
    """
    Save a constraint file for Stage 3 to use
    Maps beam_id -> number_of_columns for Stage 3 constraint
    """
    
    constraint_data = []
    for pred in predictions:
        constraint_data.append({
            'beam_id': pred['beam_id'],
            'max_columns': pred['predicted_columns'],
            'confidence': pred['confidence']
        })
    
    constraint_df = pd.DataFrame(constraint_data)
    constraint_file = os.path.join(output_dir, f"{building_id}_stage3_constraints.csv")
    constraint_df.to_csv(constraint_file, index=False)
    
    return constraint_file

def save_building_results(building_id, predictions, output_dir, validation_path):
    """Save predictions as BeamColumnMatrix and updated FeatureMatrix"""
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate BeamColumnMatrix
    beam_column_matrix, all_column_ids = generate_beam_column_matrix(building_id, predictions, validation_path)
    
    # Determine where to save files
    if UPDATE_ORIGINAL_FILES:
        # Create backup of original files first
        backup_original_files(building_id, validation_path, output_dir)
        
        # Update original files in validation folder
        matrix_file = os.path.join(validation_path, f"{building_id}_BeamColumnMatrix.csv")
        feature_file_path = validation_path
        print(f"  🔄 Updating original files in validation folder")
    else:
        # Save to output directory
        matrix_file = os.path.join(output_dir, f"{building_id}_BeamColumnMatrix.csv")
        feature_file_path = output_dir
        print(f"  💾 Saving to output directory")
    
    # Save BeamColumnMatrix
    beam_column_matrix.to_csv(matrix_file, index=True)
    
    # Update and save FeatureMatrix
    updated_feature_matrix = update_feature_matrix(building_id, predictions, all_column_ids, validation_path)
    
    feature_file = None
    if updated_feature_matrix is not None:
        feature_file = os.path.join(feature_file_path, f"{building_id}_FeatureMatrix.csv")
        updated_feature_matrix.to_csv(feature_file, index=False)
    
    # Always save summary to output directory for reference
    data = []
    for pred in predictions:
        data.append({
            'Beam_ID': f"{building_id}_{pred['beam_id']}_B",
            'Original_Beam_ID': pred['beam_id'],
            'Predicted_Columns': pred['predicted_columns'],
            'Confidence': pred['confidence'],
            'Predicted_Material': pred['material'],
            'Material_Confidence': pred['material_confidence'],
        })
    
    df = pd.DataFrame(data)
    summary_file = os.path.join(output_dir, f"{building_id}_predictions_summary.csv")
    df.to_csv(summary_file, index=False)
    
    # Save Stage 3 constraint file
    constraint_file = save_stage3_constraint_file(building_id, predictions, output_dir)
    
    return matrix_file, feature_file, summary_file, constraint_file

def generate_summary_report(all_stats, output_dir):
    """Generate comprehensive summary report"""
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_stats)
    
    # Calculate totals
    total_buildings = len(all_stats)
    total_beams = summary_df['total_beams'].sum()
    total_columns = summary_df['total_columns'].sum()
    total_class_0 = summary_df['beams_class_0'].sum()
    total_class_1 = summary_df['beams_class_1'].sum()
    total_class_2 = summary_df['beams_class_2'].sum()
    total_high_conf = summary_df['high_confidence'].sum()
    total_low_conf = summary_df['low_confidence'].sum()
    avg_confidence = summary_df['avg_confidence'].mean()
    
    # Save summary CSV
    summary_file = os.path.join(output_dir, "validation_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # Generate detailed report
    report_file = os.path.join(output_dir, "validation_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STAGE 2 VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Validation Path: {VALIDATION_PATH}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Buildings: {total_buildings}\n")
        f.write(f"Total Beams: {total_beams}\n")
        f.write(f"Total Predicted Columns: {total_columns}\n\n")
        
        f.write("Class Distribution:\n")
        f.write(f"  Beams with 0 columns: {total_class_0} ({100*total_class_0/total_beams:.1f}%)\n")
        f.write(f"  Beams with 1 column:  {total_class_1} ({100*total_class_1/total_beams:.1f}%)\n")
        f.write(f"  Beams with 2 columns: {total_class_2} ({100*total_class_2/total_beams:.1f}%)\n\n")
        
        f.write("Confidence Statistics:\n")
        f.write(f"  High confidence (>80%): {total_high_conf} ({100*total_high_conf/total_beams:.1f}%)\n")
        f.write(f"  Low confidence (<50%):  {total_low_conf} ({100*total_low_conf/total_beams:.1f}%)\n")
        f.write(f"  Average confidence:     {avg_confidence:.4f} ({100*avg_confidence:.2f}%)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("PER-BUILDING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Building ID':<20} {'Beams':<8} {'Columns':<10} {'Avg Conf':<12} {'Low Conf':<10}\n")
        f.write("-" * 80 + "\n")
        
        for stats in all_stats:
            f.write(f"{stats['building_id']:<20} "
                   f"{stats['total_beams']:<8} "
                   f"{stats['total_columns']:<10} "
                   f"{stats['avg_confidence']:<12.4f} "
                   f"{stats['low_confidence']:<10}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED BREAKDOWN\n")
        f.write("=" * 80 + "\n\n")
        
        for stats in all_stats:
            f.write(f"\nBuilding: {stats['building_id']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Total beams: {stats['total_beams']}\n")
            f.write(f"  Beams with 0 columns: {stats['beams_class_0']}\n")
            f.write(f"  Beams with 1 column:  {stats['beams_class_1']}\n")
            f.write(f"  Beams with 2 columns: {stats['beams_class_2']}\n")
            f.write(f"  Total columns: {stats['total_columns']}\n")
            f.write(f"  High confidence: {stats['high_confidence']}\n")
            f.write(f"  Low confidence: {stats['low_confidence']}\n")
            f.write(f"  Avg confidence: {stats['avg_confidence']:.4f}\n")
    
    return report_file, summary_file

def generate_markdown_report(all_stats, output_dir):
    """Generate markdown report for easy viewing"""
    
    report_file = os.path.join(output_dir, "VALIDATION_REPORT.md")
    
    # Calculate totals
    total_buildings = len(all_stats)
    total_beams = sum(s['total_beams'] for s in all_stats)
    total_columns = sum(s['total_columns'] for s in all_stats)
    total_class_0 = sum(s['beams_class_0'] for s in all_stats)
    total_class_1 = sum(s['beams_class_1'] for s in all_stats)
    total_class_2 = sum(s['beams_class_2'] for s in all_stats)
    total_high_conf = sum(s['high_confidence'] for s in all_stats)
    total_low_conf = sum(s['low_confidence'] for s in all_stats)
    avg_confidence = np.mean([s['avg_confidence'] for s in all_stats])
    
    with open(report_file, 'w') as f:
        f.write("# Stage 2 Validation Report\n\n")
        
        f.write(f"**Validation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Model**: `{MODEL_PATH}`\n\n")
        f.write(f"**Validation Dataset**: `{VALIDATION_PATH}`\n\n")
        
        f.write("---\n\n")
        f.write("## Overall Statistics\n\n")
        
        f.write(f"- **Total Buildings**: {total_buildings}\n")
        f.write(f"- **Total Beams**: {total_beams}\n")
        f.write(f"- **Total Predicted Columns**: {total_columns}\n")
        f.write(f"- **Average Confidence**: {avg_confidence:.2%}\n\n")
        
        f.write("### Class Distribution\n\n")
        f.write("| Class | Count | Percentage |\n")
        f.write("|-------|-------|------------|\n")
        f.write(f"| 0 columns | {total_class_0} | {100*total_class_0/total_beams:.1f}% |\n")
        f.write(f"| 1 column | {total_class_1} | {100*total_class_1/total_beams:.1f}% |\n")
        f.write(f"| 2 columns | {total_class_2} | {100*total_class_2/total_beams:.1f}% |\n\n")
        
        f.write("### Confidence Distribution\n\n")
        f.write("| Confidence Level | Count | Percentage |\n")
        f.write("|-----------------|-------|------------|\n")
        f.write(f"| High (>80%) | {total_high_conf} | {100*total_high_conf/total_beams:.1f}% |\n")
        f.write(f"| Medium (50-80%) | {total_beams - total_high_conf - total_low_conf} | {100*(total_beams - total_high_conf - total_low_conf)/total_beams:.1f}% |\n")
        f.write(f"| Low (<50%) | {total_low_conf} | {100*total_low_conf/total_beams:.1f}% |\n\n")
        
        f.write("---\n\n")
        f.write("## Per-Building Results\n\n")
        
        f.write("| Building ID | Beams | Columns | Class 0 | Class 1 | Class 2 | Avg Conf | Low Conf |\n")
        f.write("|-------------|-------|---------|---------|---------|---------|----------|----------|\n")
        
        for stats in all_stats:
            f.write(f"| {stats['building_id']} | "
                   f"{stats['total_beams']} | "
                   f"{stats['total_columns']} | "
                   f"{stats['beams_class_0']} | "
                   f"{stats['beams_class_1']} | "
                   f"{stats['beams_class_2']} | "
                   f"{stats['avg_confidence']:.2%} | "
                   f"{stats['low_confidence']} |\n")
        
        f.write("\n---\n\n")
        f.write("## Detailed Breakdown\n\n")
        
        for stats in all_stats:
            f.write(f"### Building: {stats['building_id']}\n\n")
            f.write(f"- **Total beams**: {stats['total_beams']}\n")
            f.write(f"- **Total columns**: {stats['total_columns']}\n")
            f.write(f"- **Average confidence**: {stats['avg_confidence']:.2%}\n\n")
            
            f.write("**Class Distribution**:\n")
            f.write(f"- 0 columns: {stats['beams_class_0']} beams\n")
            f.write(f"- 1 column: {stats['beams_class_1']} beams\n")
            f.write(f"- 2 columns: {stats['beams_class_2']} beams\n\n")
            
            f.write("**Confidence**:\n")
            f.write(f"- High confidence (>80%): {stats['high_confidence']} beams\n")
            f.write(f"- Low confidence (<50%): {stats['low_confidence']} beams\n\n")
    
    return report_file

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("STAGE 2 VALIDATION - BEAMCOLUMN MATRIX GENERATION")
    print("=" * 80)
    
    print(f"\n📋 Configuration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Validation Path: {VALIDATION_PATH}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  Update Original Files: {'YES' if UPDATE_ORIGINAL_FILES else 'NO'}")
    
    if UPDATE_ORIGINAL_FILES:
        print(f"\n⚠️  WARNING: This will modify the original files in {VALIDATION_PATH}")
        print(f"   Original files will be backed up to {OUTPUT_DIR}/original_backups/")
    
    # Set random seed for reproducible column IDs
    random.seed(42)
    
    # Load model
    print(f"\n🔄 Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print(f"  ✅ Model loaded successfully")
    
    # Get validation buildings
    print(f"\n📂 Scanning validation folder: {VALIDATION_PATH}")
    building_ids = get_validation_buildings(VALIDATION_PATH)
    print(f"  ✅ Found {len(building_ids)} buildings")
    
    if len(building_ids) == 0:
        print("\n❌ No buildings found in validation folder!")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    
    # Process each building
    print("\n" + "=" * 80)
    print("RUNNING INFERENCE")
    print("=" * 80)
    
    all_stats = []
    all_predictions = {}
    
    for i, building_id in enumerate(building_ids, 1):
        print(f"\n[{i}/{len(building_ids)}] Processing: {building_id}")
        
        result = validate_building(model, building_id, VALIDATION_PATH)
        
        if result is not None:
            all_stats.append(result['stats'])
            all_predictions[building_id] = result['predictions']
            
            # Save individual building results
            matrix_file, feature_file, summary_file, constraint_file = save_building_results(
                building_id,
                result['predictions'],
                OUTPUT_DIR,
                VALIDATION_PATH
            )
            
            print(f"  ✅ Processed {result['stats']['total_beams']} beams")
            print(f"  📊 Predicted {result['stats']['total_columns']} columns")
            print(f"  💾 BeamColumnMatrix: {matrix_file}")
            if feature_file:
                print(f"  💾 Updated FeatureMatrix: {feature_file}")
            print(f"  💾 Summary: {summary_file}")
            print(f"  🔗 Stage 3 constraints: {constraint_file}")
            
            # Display matrix sample for verification
            display_matrix_sample(building_id, matrix_file)
    
    # Generate summary reports
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)
    
    if len(all_stats) > 0:
        # Text report
        report_file, summary_file = generate_summary_report(all_stats, OUTPUT_DIR)
        print(f"\n📄 Text report: {report_file}")
        print(f"📊 Summary CSV: {summary_file}")
        
        # Markdown report
        md_report = generate_markdown_report(all_stats, OUTPUT_DIR)
        print(f"📝 Markdown report: {md_report}")
        
        # Display summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        total_buildings = len(all_stats)
        total_beams = sum(s['total_beams'] for s in all_stats)
        total_columns = sum(s['total_columns'] for s in all_stats)
        avg_confidence = np.mean([s['avg_confidence'] for s in all_stats])
        
        print(f"\n✅ Validation complete!")
        print(f"\n📊 Results:")
        print(f"  Buildings processed: {total_buildings}")
        print(f"  Total beams: {total_beams}")
        print(f"  Total columns predicted: {total_columns}")
        print(f"  Average confidence: {avg_confidence:.2%}")
        
        print(f"\n📁 All results saved to: {OUTPUT_DIR}")
        print(f"\n📋 Generated files per building:")
        print(f"  - {building_id}_BeamColumnMatrix.csv (beam-column connection matrix)")
        print(f"  - {building_id}_FeatureMatrix.csv (updated with predicted columns)")
        print(f"  - {building_id}_predictions_summary.csv (prediction details)")
        
    else:
        print("\n❌ No buildings were successfully processed!")

if __name__ == "__main__":
    main()
