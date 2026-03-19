#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 Inference Script (INCREMENTAL MODE) - BATCH VERSION
Predicts column counts only for beams with missing column information
If BeamColumnMatrix is provided with some columns, only predicts for beams without columns
Supports multiple building IDs and saves results in result/incremental directory
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import pandas as pd
import numpy as np
import os
import glob
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = r"E:/ModelCreation/Models/stage2_model.pth"
DATASET_PATH = r"E:/ModelCreation/Dataset/test"
OUTPUT_DIR = "result/incremental/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

def discover_buildings(dataset_path):
    """
    Automatically discover all building IDs from FeatureMatrix files in the dataset path
    """
    pattern = os.path.join(dataset_path, "*_FeatureMatrix.csv")
    feature_files = glob.glob(pattern)
    
    building_ids = []
    for file_path in feature_files:
        filename = os.path.basename(file_path)
        building_id = filename.replace("_FeatureMatrix.csv", "")
        building_ids.append(building_id)
    
    building_ids.sort()  # Sort for consistent ordering
    return building_ids

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Material mapping (from training)
MATERIAL_MAPPING = {
    "Concrete": 1, "Steel": 2, "Wood": 3, "Masonry": 4,
    "Composite": 5, "Other": 6, "Generic": 7, "Unknown": 0
}
MATERIAL_NAMES = {v: k for k, v in MATERIAL_MAPPING.items()}

def encode_material(val):
    v = str(val).strip() if pd.notna(val) else "Unknown"
    return MATERIAL_MAPPING.get(v, MATERIAL_MAPPING["Unknown"])

def norm_id(x):
    """Normalize element ID"""
    return str(x).split("_")[-1] if "_" in str(x) else str(x)

def extract_element_id(label):
    """Extract element ID from label"""
    parts = str(label).split("_")
    if len(parts) >= 2:
        return parts[1]
    return str(label)

def encode_node_type(element_type):
    """Encode element type as one-hot vector"""
    if "Framing" in element_type or "Beam" in element_type:
        return [1, 0, 0]
    elif "Wall" in element_type:
        return [0, 1, 0]
    elif "Column" in element_type:
        return [0, 0, 1]
    else:
        return [0, 0, 0]

def adjacency_to_edges(adj_df, node_map):
    """Convert adjacency matrix to edge list"""
    edges = []
    
    row_ids = adj_df.iloc[:, 0].values
    col_ids = adj_df.columns[1:]
    
    for i, row_label in enumerate(row_ids):
        src_elem = extract_element_id(row_label)
        if src_elem not in node_map:
            continue
        
        src_node = node_map[src_elem]
        
        for j, col_label in enumerate(col_ids):
            if adj_df.iloc[i, j+1] == 1:
                tgt_elem = extract_element_id(col_label)
                
                if tgt_elem in node_map:
                    tgt_node = node_map[tgt_elem]
                    edges.append([src_node, tgt_node])
                    edges.append([tgt_node, src_node])
    
    return edges

def identify_beams_with_columns(bc_df):
    """
    Identify which beams already have column information
    Returns: dict {beam_id: column_count}
    """
    beams_with_columns = {}
    
    row_ids = bc_df.iloc[:, 0].values
    
    for i, row_label in enumerate(row_ids):
        beam_id = extract_element_id(row_label)
        
        # Count columns for this beam
        column_count = int(bc_df.iloc[i, 1:].sum())
        
        if column_count > 0:
            beams_with_columns[beam_id] = column_count
    
    return beams_with_columns

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class Stage2BeamModel(torch.nn.Module):
    def __init__(self, in_dim, hidden=32, classes=3):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.head = torch.nn.Linear(hidden, classes)
        self.mat_head = torch.nn.Linear(hidden, 8)  # Material prediction head (8 classes)
    
    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, 0.35, self.training)
        x = F.relu(self.c2(x, edge_index))
        return self.head(x), self.mat_head(x)

# ============================================================================
# GRAPH BUILDING (INFERENCE MODE)
# ============================================================================

def build_graph_inference(model_id, dataset_path="Dataset/", beam_column_df=None):
    """
    Build graph for inference using the same feature engineering as the working script
    If beam_column_df is provided, identify beams that already have columns
    """
    
    feat_file = os.path.join(dataset_path, f"{model_id}_FeatureMatrix.csv")
    bw_file = os.path.join(dataset_path, f"{model_id}_BeamWallMatrix.csv")
    bb_file = os.path.join(dataset_path, f"{model_id}_BeamBeamMatrix.csv")
    
    df = pd.read_csv(feat_file)
    bw_df = pd.read_csv(bw_file)
    bb_df = pd.read_csv(bb_file) if os.path.exists(bb_file) else None
    
    # Identify beams with existing columns
    beams_with_columns = {}
    if beam_column_df is not None:
        beams_with_columns = identify_beams_with_columns(beam_column_df)
        print(f"\n📊 Found {len(beams_with_columns)} beams with existing column information:")
        for beam_id, count in beams_with_columns.items():
            print(f"  Beam {beam_id}: {count} columns (will skip prediction)")
    
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Remove column nodes (matching training)
    df = df[~df["Element Type"].str.contains("Column", case=False, na=False)]
    
    # Separate walls and beams
    wall_df = df[df["Element Type"].str.contains("Wall", case=False, na=False)]
    beam_df = df[df["Element Type"].str.contains("Beam|Framing", case=False, na=False)]
    
    # Features (matching training exactly)
    WALL_FEATS = ["Start X", "Start Y", "Start Z", "End X", "End Y", "End Z", "Length", "Area"]
    BEAM_FEATS = ["Start X", "Start Y", "Start Z", "End X", "End Y", "End Z",
                  "Length", "Volume", "Entity Start Level", "Entity End Level"]
    
    wall_x = torch.tensor(wall_df[WALL_FEATS].values, dtype=torch.float)
    beam_x = torch.tensor(beam_df[BEAM_FEATS].values, dtype=torch.float)
    
    # Normalized beam length (matching training)
    bl = beam_df["Length"].values
    bl_norm = bl / (np.mean(bl) + 1e-6)
    bl_feat = torch.tensor(bl_norm, dtype=torch.float).unsqueeze(1)
    beam_x = torch.cat([beam_x, bl_feat], dim=1)

    # Structural Material encoded feature (beam only)
    mat_ids = beam_df["Structural Material"].apply(encode_material).values
    mat_feat = torch.tensor(mat_ids, dtype=torch.float).unsqueeze(1)
    beam_x = torch.cat([beam_x, mat_feat], dim=1)
    
    # Pad dimensions (matching training)
    max_dim = max(wall_x.shape[1], beam_x.shape[1])
    wall_x = F.pad(wall_x, (0, max_dim - wall_x.shape[1]))
    beam_x = F.pad(beam_x, (0, max_dim - beam_x.shape[1]))
    
    # Type flag (matching training)
    wall_x = torch.cat([wall_x, torch.ones((wall_x.size(0), 1))], dim=1)
    beam_x = torch.cat([beam_x, torch.zeros((beam_x.size(0), 1))], dim=1)
    
    x = torch.cat([wall_x, beam_x], dim=0)
    
    # Node IDs
    beam_ids = beam_df["Element ID"].astype(str).apply(norm_id).tolist()
    wall_ids = wall_df["Element ID"].astype(str).apply(norm_id).tolist()
    
    node_ids = wall_ids + beam_ids
    id_to_idx = {eid: i for i, eid in enumerate(node_ids)}
    
    # Build edges + support stats
    edges = []
    support_degree = torch.zeros(len(node_ids))
    endpoint_support = torch.zeros(len(node_ids))
    
    # Beam-Wall edges
    for _, row in bw_df.iterrows():
        b = norm_id(row.iloc[0])
        if b not in id_to_idx:
            continue
        
        bi = id_to_idx[b]
        connected = 0
        
        for w_raw, val in row.iloc[1:].items():
            if val == 1:
                w = norm_id(w_raw)
                if w in id_to_idx:
                    wi = id_to_idx[w]
                    edges.append([bi, wi])
                    edges.append([wi, bi])
                    connected += 1
        
        support_degree[bi] = connected
        endpoint_support[bi] = 1 if connected >= 2 else 0
    
    # Beam-Beam edges
    if bb_df is not None:
        for _, row in bb_df.iterrows():
            b1 = norm_id(row.iloc[0])
            if b1 not in id_to_idx:
                continue
            
            b1i = id_to_idx[b1]
            
            for b2_raw, val in row.iloc[1:].items():
                if val == 1:
                    b2 = norm_id(b2_raw)
                    if b2 in id_to_idx:
                        b2i = id_to_idx[b2]
                        edges.append([b1i, b2i])
                        edges.append([b2i, b1i])
    
    edge_index = (
        torch.tensor(edges, dtype=torch.long).T
        if edges else
        torch.arange(x.size(0)).repeat(2, 1)
    )
    
    # Append support features (matching training)
    x = torch.cat([
        x,
        support_degree.unsqueeze(1),
        endpoint_support.unsqueeze(1)
    ], dim=1)
    
    data = Data(x=x, edge_index=edge_index)
    
    # Track beam indices and IDs
    beam_indices = list(range(len(wall_ids), len(node_ids)))
    beam_ids_list = beam_ids
    
    # Track which beams need prediction
    beams_to_predict = []
    for beam_id in beam_ids:
        if beam_id in beams_with_columns:
            beams_to_predict.append(False)  # Skip - already has columns
        else:
            beams_to_predict.append(True)   # Predict - missing columns
    
    return data, beam_indices, beam_ids_list, beams_to_predict, beams_with_columns

# ============================================================================
# INFERENCE
# ============================================================================

def predict_incremental(model_id, dataset_path="Dataset/", beam_column_file=None, output_dir="result/incremental/"):
    """
    Run incremental prediction
    If beam_column_file is provided, only predict for beams without columns
    """
    
    print(f"\n🔮 Running Stage 2 Inference (INCREMENTAL MODE)")
    print(f"Building ID: {model_id}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load BeamColumnMatrix if provided
    beam_column_df = None
    if beam_column_file and os.path.exists(beam_column_file):
        print(f"\n📂 Loading existing BeamColumnMatrix: {beam_column_file}")
        beam_column_df = pd.read_csv(beam_column_file)
    else:
        print(f"\n📂 No existing BeamColumnMatrix provided - will predict for all beams")
    
    # Build graph
    print(f"\n🔨 Building graph...")
    data, beam_indices, beam_ids, beams_to_predict, beams_with_columns = build_graph_inference(
        model_id, dataset_path, beam_column_df
    )
    
    total_beams = len(beam_indices)
    beams_to_predict_count = sum(beams_to_predict)
    beams_to_skip_count = total_beams - beams_to_predict_count
    
    print(f"  Total beams: {total_beams}")
    print(f"  Beams to predict: {beams_to_predict_count}")
    print(f"  Beams to skip (already have columns): {beams_to_skip_count}")
    
    # Load model
    print(f"\n📦 Loading model: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        in_channels = checkpoint.get('in_dim', data.num_node_features)
        model = Stage2BeamModel(in_channels, hidden=32, classes=3)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state dict
        in_channels = data.num_node_features
        model = Stage2BeamModel(in_channels, hidden=32, classes=3)
        model.load_state_dict(checkpoint)
    
    model.to(DEVICE)
    model.eval()
    
    print(f"  ✅ Model loaded (input channels: {in_channels})")
    
    # Run inference
    print(f"\n🚀 Running predictions...")
    
    with torch.no_grad():
        out, mat_out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)
        mat_probs = F.softmax(mat_out, dim=1)
    
    # Extract predictions for beams
    beam_predictions = []
    
    for i, (beam_idx, beam_id, should_predict) in enumerate(zip(beam_indices, beam_ids, beams_to_predict)):
        if should_predict:
            # Predict for this beam
            beam_probs = probs[beam_idx].cpu().numpy()
            predicted_class = int(beam_probs.argmax())
            confidence = float(beam_probs[predicted_class])
            
            # Material predictions
            beam_mat_probs = mat_probs[beam_idx].cpu().numpy()
            predicted_material_class = int(beam_mat_probs.argmax())
            material_confidence = float(beam_mat_probs[predicted_material_class])
            predicted_material_name = MATERIAL_NAMES.get(predicted_material_class, "Unknown")
            
            beam_predictions.append({
                'beam_id': f"{model_id}_{beam_id}_B",
                'predicted_columns': predicted_class,
                'confidence': confidence,
                'prob_0': float(beam_probs[0]),
                'prob_1': float(beam_probs[1]),
                'prob_2': float(beam_probs[2]),
                'predicted_material': predicted_material_name,
                'material_confidence': material_confidence,
                'source': 'predicted'
            })
        else:
            # Use existing column count
            existing_count = beams_with_columns[beam_id]
            
            beam_predictions.append({
                'beam_id': f"{model_id}_{beam_id}_B",
                'predicted_columns': existing_count,
                'confidence': 1.0,  # Known value
                'prob_0': 1.0 if existing_count == 0 else 0.0,
                'prob_1': 1.0 if existing_count == 1 else 0.0,
                'prob_2': 1.0 if existing_count == 2 else 0.0,
                'predicted_material': 'Unknown',  # No material prediction for existing
                'material_confidence': 0.0,
                'source': 'existing'
            })
    
    # Create results dataframe
    results_df = pd.DataFrame(beam_predictions)
    
    # Save results
    output_file = os.path.join(output_dir, f"{model_id}_predictions_incremental.csv")
    results_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Predictions complete!")
    print(f"💾 Results saved to: {output_file}")
    
    # Summary
    print(f"\n📊 Summary:")
    predicted_beams = results_df[results_df['source'] == 'predicted']
    existing_beams = results_df[results_df['source'] == 'existing']
    
    print(f"  Total beams: {len(results_df)}")
    print(f"  Newly predicted: {len(predicted_beams)}")
    print(f"  Existing (kept): {len(existing_beams)}")
    
    if len(predicted_beams) > 0:
        print(f"\n📊 Newly Predicted Distribution:")
        for col_count in [0, 1, 2]:
            count = len(predicted_beams[predicted_beams['predicted_columns'] == col_count])
            pct = count / len(predicted_beams) * 100
            print(f"  {col_count} columns: {count} beams ({pct:.1f}%)")
        
        avg_conf = predicted_beams['confidence'].mean()
        print(f"\n📊 Average Confidence: {avg_conf:.2%}")
        
        # Material prediction summary
        avg_mat_conf = predicted_beams['material_confidence'].mean()
        print(f"📊 Average Material Confidence: {avg_mat_conf:.2%}")
        
        print(f"\n📊 Predicted Materials:")
        material_counts = predicted_beams['predicted_material'].value_counts()
        for material, count in material_counts.items():
            pct = count / len(predicted_beams) * 100
            print(f"  {material}: {count} beams ({pct:.1f}%)")
    
    if len(existing_beams) > 0:
        print(f"\n📊 Existing (Kept) Distribution:")
        for col_count in [0, 1, 2]:
            count = len(existing_beams[existing_beams['predicted_columns'] == col_count])
            if count > 0:
                print(f"  {col_count} columns: {count} beams")
    
    return results_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run batch incremental predictions"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stage 2 Incremental Prediction')
    parser.add_argument('--dataset-path', type=str, default=DATASET_PATH,
                        help='Path to dataset directory (default: ../Dataset/test/)')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='Output directory for results (default: result/incremental/)')
    
    args = parser.parse_args()
    
    # Use command line arguments or defaults
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    
    print("=" * 80)
    print("STAGE 2: PER-BEAM COLUMN CLASSIFICATION (INCREMENTAL MODE - BATCH)")
    print("=" * 80)
    print("\n⚠️  INCREMENTAL MODE: Only predicts for beams without existing column information")
    print("   If BeamColumnMatrix exists, beams with columns will be skipped")
    
    # Discover buildings from dataset
    print(f"\n🔍 Discovering buildings from: {dataset_path}")
    building_ids = discover_buildings(dataset_path)
    
    if not building_ids:
        print(f"❌ No buildings found in {dataset_path}")
        print("   Make sure the directory contains *_FeatureMatrix.csv files")
        return {}
    
    print(f"✅ Found {len(building_ids)} buildings: {building_ids[:10]}{'...' if len(building_ids) > 10 else ''}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Process each building
    total_buildings = len(building_ids)
    successful_predictions = 0
    all_results = {}
    
    print(f"\n📊 Processing {total_buildings} buildings...")
    print("="*80)
    
    for i, building_id in enumerate(building_ids, 1):
        print(f"\n[{i}/{total_buildings}] Processing Building: {building_id}")
        print("-" * 60)
        
        # Check if required files exist
        feature_csv = os.path.join(dataset_path, f'{building_id}_FeatureMatrix.csv')
        beam_wall_csv = os.path.join(dataset_path, f'{building_id}_BeamWallMatrix.csv')
        beam_column_csv = os.path.join(dataset_path, f'{building_id}_BeamColumnMatrix.csv')
        
        required_files = [feature_csv, beam_wall_csv]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"❌ Error: Missing required files for {building_id}:")
            for f in missing_files:
                print(f"   - {os.path.basename(f)}")
            print(f"⏭️  Skipping building {building_id}")
            continue
        
        try:
            # Check if BeamColumnMatrix exists for incremental mode
            beam_column_file = beam_column_csv if os.path.exists(beam_column_csv) else None
            
            if beam_column_file:
                print(f"📂 Found existing BeamColumnMatrix - will use incremental mode")
            else:
                print(f"📂 No existing BeamColumnMatrix - will predict for all beams")
            
            # Run incremental prediction
            results = predict_incremental(
                building_id, 
                dataset_path, 
                beam_column_file=beam_column_file,
                output_dir=output_dir
            )
            
            # Store results
            all_results[building_id] = results
            
            # Display summary for this building
            total_beams = len(results)
            predicted_beams = results[results['source'] == 'predicted']
            existing_beams = results[results['source'] == 'existing']
            total_columns = results['predicted_columns'].sum()
            avg_conf = results['confidence'].mean()
            avg_mat_conf = predicted_beams['material_confidence'].mean() if len(predicted_beams) > 0 else 0.0
            
            print(f"✅ Building {building_id} processed:")
            print(f"   - Total beams: {total_beams}")
            print(f"   - Newly predicted: {len(predicted_beams)}")
            print(f"   - Existing (kept): {len(existing_beams)}")
            print(f"   - Total columns: {total_columns}")
            print(f"   - Average confidence: {avg_conf:.2%}")
            print(f"   - Average material confidence: {avg_mat_conf:.2%}")
            
            successful_predictions += 1
            
        except Exception as e:
            print(f"❌ Error processing building {building_id}: {str(e)}")
            print(f"⏭️  Skipping building {building_id}")
            continue
    
    # Final summary
    print("\n" + "="*80)
    print("BATCH INCREMENTAL PREDICTION SUMMARY")
    print("="*80)
    
    print(f"\n📊 Overall Results:")
    print(f"   Buildings requested: {total_buildings}")
    print(f"   Buildings processed: {successful_predictions}")
    print(f"   Buildings failed: {total_buildings - successful_predictions}")
    
    if successful_predictions > 0:
        # Calculate aggregate statistics
        total_beams_all = sum(len(results) for results in all_results.values())
        total_predicted_all = sum(len(results[results['source'] == 'predicted']) for results in all_results.values())
        total_existing_all = sum(len(results[results['source'] == 'existing']) for results in all_results.values())
        total_columns_all = sum(results['predicted_columns'].sum() for results in all_results.values())
        avg_conf_all = sum(results['confidence'].mean() * len(results) for results in all_results.values()) / total_beams_all
        
        # Calculate material confidence for predicted beams only
        all_predicted_beams = pd.concat([results[results['source'] == 'predicted'] for results in all_results.values() if len(results[results['source'] == 'predicted']) > 0])
        avg_mat_conf_all = all_predicted_beams['material_confidence'].mean() if len(all_predicted_beams) > 0 else 0.0
        
        print(f"\n📈 Aggregate Statistics:")
        print(f"   Total beams: {total_beams_all}")
        print(f"   Newly predicted: {total_predicted_all}")
        print(f"   Existing (kept): {total_existing_all}")
        print(f"   Total columns: {total_columns_all}")
        print(f"   Overall average confidence: {avg_conf_all:.2%}")
        print(f"   Overall average material confidence: {avg_mat_conf_all:.2%}")
        
        # Show material distribution across all predicted beams
        if len(all_predicted_beams) > 0:
            print(f"\n📊 Overall Material Distribution (Predicted Beams Only):")
            material_counts = all_predicted_beams['predicted_material'].value_counts()
            for material, count in material_counts.items():
                pct = count / len(all_predicted_beams) * 100
                print(f"   {material}: {count} beams ({pct:.1f}%)")
        
        # Show per-building breakdown
        print(f"\n📋 Per-Building Breakdown:")
        print("Building ID    Beams    Predicted    Existing    Columns    Avg Conf    Mat Conf")
        print("-" * 78)
        for building_id, results in all_results.items():
            beams = len(results)
            predicted = len(results[results['source'] == 'predicted'])
            existing = len(results[results['source'] == 'existing'])
            columns = results['predicted_columns'].sum()
            avg_conf = results['confidence'].mean()
            predicted_beams = results[results['source'] == 'predicted']
            avg_mat_conf = predicted_beams['material_confidence'].mean() if len(predicted_beams) > 0 else 0.0
            print(f"{building_id:<12} {beams:<8} {predicted:<11} {existing:<10} {columns:<9} {avg_conf:.2%}      {avg_mat_conf:.2%}")
        
        # Generate summary report
        summary_file = os.path.join(output_dir, "incremental_summary.csv")
        summary_data = []
        for building_id, results in all_results.items():
            predicted_beams = results[results['source'] == 'predicted']
            avg_mat_conf = predicted_beams['material_confidence'].mean() if len(predicted_beams) > 0 else 0.0
            
            summary_data.append({
                'building_id': building_id,
                'total_beams': len(results),
                'predicted_beams': len(predicted_beams),
                'existing_beams': len(results[results['source'] == 'existing']),
                'total_columns': results['predicted_columns'].sum(),
                'avg_confidence': results['confidence'].mean(),
                'avg_material_confidence': avg_mat_conf,
                'columns_0': len(results[results['predicted_columns'] == 0]),
                'columns_1': len(results[results['predicted_columns'] == 1]),
                'columns_2': len(results[results['predicted_columns'] == 2])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        print(f"\n💾 Summary report saved to: {summary_file}")
    
    print(f"\n✅ Batch incremental prediction complete!")
    print(f"📁 All results saved to: {output_dir}")
    
    return all_results

if __name__ == "__main__":
    main()
