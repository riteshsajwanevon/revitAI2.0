#!/usr/bin/env python3
"""
Stage 3 Validation with Stage 2 Constraint
This script uses Stage 2 predictions to limit Stage 3 coordinate outputs.

If Stage 2 predicts N columns for a beam, Stage 3 will output exactly N coordinates.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import ast
from scipy.signal import find_peaks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and data paths
MODEL_PATH = "column_predictor_no_leakage.pth"
VALIDATION_SIGNAL_DIR = "Dataset/validation/ml_signal"
FEATURE_DIR = "Dataset/validation"
STAGE2_RESULTS_DIR = "result/validation_results"  # Where Stage 2 saved BeamColumnMatrix

PEAK_THRESHOLD = 0.35

# ============================================================
# CNN MODEL (Same as original)
# ============================================================

class ImprovedColumnPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2,64,7,padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64,128,5,padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128,128,5,padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128,64,5,padding=2)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64,1,1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout(self.relu(self.bn4(self.conv4(x))))
        x = self.sigmoid(self.conv5(x))
        return x.squeeze(1)

# ============================================================
# Load Stage 2 Predictions
# ============================================================

def load_stage2_predictions(building_id):
    """Load Stage 2 predictions for a building"""
    
    # Try to load from summary file first
    summary_file = os.path.join(STAGE2_RESULTS_DIR, f"{building_id}_predictions_summary.csv")
    
    if os.path.exists(summary_file):
        df = pd.read_csv(summary_file)
        # Create mapping from original beam ID to predicted columns
        stage2_predictions = {}
        for _, row in df.iterrows():
            original_beam_id = str(row['Original_Beam_ID'])
            predicted_columns = int(row['Predicted_Columns'])
            stage2_predictions[original_beam_id] = predicted_columns
        return stage2_predictions
    
    # Fallback: try to infer from BeamColumnMatrix
    matrix_file = os.path.join(STAGE2_RESULTS_DIR, f"{building_id}_BeamColumnMatrix.csv")
    
    if os.path.exists(matrix_file):
        df = pd.read_csv(matrix_file, index_col=0)
        stage2_predictions = {}
        
        for beam_full_id in df.index:
            # Extract original beam ID from format: building_id_beam_id_B
            parts = beam_full_id.split('_')
            if len(parts) >= 3 and parts[-1] == 'B':
                original_beam_id = parts[1]  # Get the beam_id part
                # Count columns (non-zero values in the row)
                predicted_columns = int(df.loc[beam_full_id].sum())
                stage2_predictions[original_beam_id] = predicted_columns
        
        return stage2_predictions
    
    print(f"⚠️  No Stage 2 predictions found for building {building_id}")
    return {}

# ============================================================
# Constrained Peak Extraction
# ============================================================

def extract_constrained_peaks(signal, max_peaks):
    """
    Extract peaks from signal, but limit to max_peaks based on Stage 2 prediction
    """
    
    if max_peaks == 0:
        return np.array([])
    
    # Get all potential peaks
    all_peaks, properties = find_peaks(signal, height=PEAK_THRESHOLD, distance=5)
    
    # Include edges if they meet threshold
    edge_peaks = []
    if signal[0] >= PEAK_THRESHOLD:
        edge_peaks.append(0)
    if signal[-1] >= PEAK_THRESHOLD:
        edge_peaks.append(len(signal)-1)
    
    # Combine all peaks
    all_peaks = np.concatenate([edge_peaks, all_peaks])
    all_peaks = np.unique(all_peaks)  # Remove duplicates
    
    # If we have fewer or equal peaks than max_peaks, return all
    if len(all_peaks) <= max_peaks:
        return all_peaks
    
    # If we have more peaks than allowed, select the strongest ones
    peak_heights = signal[all_peaks]
    
    # Get indices of the strongest peaks
    strongest_indices = np.argsort(peak_heights)[-max_peaks:]
    
    # Return the strongest peaks, sorted by position
    selected_peaks = all_peaks[strongest_indices]
    selected_peaks.sort()
    
    return selected_peaks

# ============================================================
# Utility Functions
# ============================================================

def index_to_xyz(idx, beam_row):
    """Convert signal index to XYZ coordinates"""
    sx, sy, sz = beam_row["Start X"], beam_row["Start Y"], beam_row["Start Z"]
    ex, ey, ez = beam_row["End X"], beam_row["End Y"], beam_row["End Z"]
    
    relative = idx / 127.0
    
    x = sx + relative * (ex - sx)
    y = sy + relative * (ey - sy)
    z = sz + relative * (ez - sz)
    
    return x, y, z

# ============================================================
# Load Model
# ============================================================

print("Loading CNN model...")
model = ImprovedColumnPredictor().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded")

# ============================================================
# Main Processing
# ============================================================

results = []
stage2_stats = {
    'total_beams': 0,
    'constrained_beams': 0,
    'unconstrained_beams': 0,
    'total_predictions': 0,
    'stage2_limited_predictions': 0
}

signal_files = [
    f for f in os.listdir(VALIDATION_SIGNAL_DIR)
    if f.endswith("_BeamSignal128.csv")
]

print(f"Found validation buildings: {len(signal_files)}")

for file in signal_files:
    building_id = file.split("_")[0]
    
    print(f"\nProcessing building: {building_id}")
    
    # Load Stage 2 predictions for this building
    stage2_predictions = load_stage2_predictions(building_id)
    print(f"  Stage 2 predictions loaded for {len(stage2_predictions)} beams")
    
    signal_file = os.path.join(VALIDATION_SIGNAL_DIR, file)
    feature_file = f"{FEATURE_DIR}/{building_id}_FeatureMatrix.csv"
    
    if not os.path.exists(feature_file):
        feature_file = f"{FEATURE_DIR}/validation/{building_id}_FeatureMatrix.csv"
    
    if not os.path.exists(feature_file):
        print("  ❌ FeatureMatrix missing")
        continue
    
    signal_df = pd.read_csv(signal_file)
    feature_df = pd.read_csv(feature_file)
    
    beam_df = feature_df[
        feature_df["Element Type"].str.contains("Beam|Framing", case=False)
    ]
    
    beam_map = {
        str(row["Element ID"]): row
        for _, row in beam_df.iterrows()
    }
    
    building_beams = 0
    building_constrained = 0
    building_predictions = 0
    building_limited = 0
    
    for _, row in signal_df.iterrows():
        beam_id = str(row["beam_id"]).split("_")[1] if "_" in str(row["beam_id"]) else str(row["beam_id"])
        
        if beam_id not in beam_map:
            continue
        
        building_beams += 1
        beam_row = beam_map[beam_id]
        
        # Get Stage 2 constraint for this beam
        max_columns = stage2_predictions.get(beam_id, None)
        
        # Prepare input signals
        wall_signal = np.array(ast.literal_eval(row["wall_signal"]))
        beam_signal = np.array(ast.literal_eval(row["beam_signal"]))
        X = np.stack([wall_signal, beam_signal])
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Get CNN prediction
        with torch.no_grad():
            pred_signal = model(X_tensor).cpu().numpy()[0]
        
        # Extract peaks with or without constraint
        if max_columns is not None:
            # Use Stage 2 constraint
            peaks = extract_constrained_peaks(pred_signal, max_columns)
            building_constrained += 1
            building_limited += max_columns
            print(f"    Beam {beam_id}: Stage 2 limit = {max_columns}, Found peaks = {len(peaks)}")
        else:
            # No constraint available, use original method
            peaks, _ = find_peaks(pred_signal, height=PEAK_THRESHOLD, distance=5)
            
            # Include edges
            if pred_signal[0] >= PEAK_THRESHOLD:
                peaks = np.insert(peaks, 0, 0)
            if pred_signal[-1] >= PEAK_THRESHOLD:
                peaks = np.append(peaks, len(pred_signal)-1)
            
            print(f"    Beam {beam_id}: No Stage 2 constraint, Found peaks = {len(peaks)}")
        
        building_predictions += len(peaks)
        
        # Convert peaks to coordinates
        for idx in peaks:
            x, y, z = index_to_xyz(idx, beam_row)
            
            results.append({
                "building_id": building_id,
                "beam_id": row["beam_id"],
                "original_beam_id": beam_id,
                "signal_index": int(idx),
                "x": x,
                "y": y,
                "z": z,
                "stage2_constraint": max_columns,
                "constrained": max_columns is not None
            })
    
    print(f"  📊 Building {building_id} summary:")
    print(f"    Total beams processed: {building_beams}")
    print(f"    Beams with Stage 2 constraint: {building_constrained}")
    print(f"    Total coordinates predicted: {building_predictions}")
    print(f"    Stage 2 limited coordinates: {building_limited}")
    
    # Update global stats
    stage2_stats['total_beams'] += building_beams
    stage2_stats['constrained_beams'] += building_constrained
    stage2_stats['unconstrained_beams'] += (building_beams - building_constrained)
    stage2_stats['total_predictions'] += building_predictions
    stage2_stats['stage2_limited_predictions'] += building_limited

# ============================================================
# Save Results
# ============================================================

output_file = "ColumnPrediction_Stage2_Constrained.csv"
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("STAGE 2 CONSTRAINT SUMMARY")
print("=" * 80)

print(f"Total beams processed: {stage2_stats['total_beams']}")
print(f"Beams with Stage 2 constraint: {stage2_stats['constrained_beams']}")
print(f"Beams without constraint: {stage2_stats['unconstrained_beams']}")
print(f"Total coordinates predicted: {stage2_stats['total_predictions']}")
print(f"Coordinates limited by Stage 2: {stage2_stats['stage2_limited_predictions']}")

constraint_percentage = (stage2_stats['constrained_beams'] / stage2_stats['total_beams']) * 100 if stage2_stats['total_beams'] > 0 else 0
print(f"Constraint coverage: {constraint_percentage:.1f}%")

print(f"\n📄 Results saved to: {output_file}")

# Save summary statistics
summary_stats = pd.DataFrame([stage2_stats])
summary_stats.to_csv("Stage2_Constraint_Summary.csv", index=False)
print(f"📊 Summary statistics saved to: Stage2_Constraint_Summary.csv")