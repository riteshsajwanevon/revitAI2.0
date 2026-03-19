#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 Training Script: Per-Beam Column Classification
Predicts number of columns (0, 1, or 2) for each beam
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
import pandas as pd
import numpy as np
import os
import glob
import re
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_PATH = r"E:\ModelCreation\Dataset\train"
STAGE2_EPOCHS = 350
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def norm_id(x):
    """Extract element ID from label"""
    x = str(x).strip().replace(".0", "")
    nums = re.findall(r'\d+', x)
    
    if len(nums) >= 2:
        return nums[1]  # second numeric block = element id
    if len(nums) == 1:
        return nums[0]
    return x

def getUniqueValues(all_dfs, ELEMENT_TYPES=("Wall", "Structural Column", "Structural Framing")):
    unique_family_per_entities = {entity: set() for entity in ELEMENT_TYPES}
    
    for name, df in all_dfs.items():
        if "FeatureMatrix" not in name:
            continue
        for entity in ELEMENT_TYPES:
            families = (
                df.loc[df["Element Type"] == entity, "Family"]
                  .dropna()
                  .astype(str)
                  .unique()
            )
            unique_family_per_entities[entity].update(families)
    
    return {entity: sorted(families) for entity, families in unique_family_per_entities.items()}

def map_family_names(family_dict):
    type_code = {
        "Wall": 1,
        "Structural Column": 2,
        "Structural Framing": 3
    }
    
    mapping = {}
    for element_type, names in family_dict.items():
        prefix = type_code[element_type]
        for idx, name in enumerate(names, start=1):
            mapped_id = int(f"{prefix}{idx}")
            mapping[(element_type, name)] = mapped_id
    
    return mapping

def add_family_id_column(all_dfs, family_mapping):
    updated_dfs = {}
    
    for name, df in all_dfs.items():
        if "FeatureMatrix" not in name:
            updated_dfs[name] = df
            continue
        
        df = df.copy()
        keys = list(zip(df["Element Type"], df["Family"].astype(str)))
        df["Family_ID"] = [family_mapping.get(key, None) for key in keys]
        updated_dfs[name] = df
    
    return updated_dfs

# ============================================================================
# MATERIAL MAPPING (built globally across all buildings)
# ============================================================================

MATERIAL_MAPPING = {
    "Metal - Steel 43-275":          0,
    "Metal - Scaffold Tube":         1,
    "Steel 50-355":                  2,
    "Steel 43-275":                  3,
    "Stainless Steel":               4,
    "Metal - Steel 43-275 - existing": 5,
    "Steel 43-275 EXISTING":         6,
    "Unknown":                       7,   # fallback
}
N_MATERIALS = len(MATERIAL_MAPPING)

def encode_material(val):
    v = str(val).strip() if pd.notna(val) else "Unknown"
    return MATERIAL_MAPPING.get(v, MATERIAL_MAPPING["Unknown"])

# ============================================================================
# STAGE 2: GRAPH LOADING
# ============================================================================

def load_stage2_graph(feature_csv, beam_wall_csv, beam_column_csv, beam_beam_csv=None):
    """Load Stage 2 graph with BeamBeamMatrix support"""
    
    df = pd.read_csv(feature_csv)
    bw = pd.read_csv(beam_wall_csv)
    bc = pd.read_csv(beam_column_csv)
    bb = pd.read_csv(beam_beam_csv) if beam_beam_csv else None
    
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Remove column nodes
    df = df[~df["Element Type"].str.contains("Column", case=False, na=False)]
    
    wall_df = df[df["Element Type"].str.contains("Wall", case=False, na=False)]
    beam_df = df[df["Element Type"].str.contains("Beam|Framing", case=False, na=False)]
    
    WALL_FEATS = ["Start X", "Start Y", "Start Z", "End X", "End Y", "End Z", "Length", "Area"]
    BEAM_FEATS = ["Start X", "Start Y", "Start Z", "End X", "End Y", "End Z",
                  "Length", "Volume", "Entity Start Level", "Entity End Level"]
    
    wall_x = torch.tensor(wall_df[WALL_FEATS].values, dtype=torch.float)
    beam_x = torch.tensor(beam_df[BEAM_FEATS].values, dtype=torch.float)
    
    # Normalized beam length
    bl = beam_df["Length"].values
    bl_norm = bl / (np.mean(bl) + 1e-6)
    bl_feat = torch.tensor(bl_norm, dtype=torch.float).unsqueeze(1)
    beam_x = torch.cat([beam_x, bl_feat], dim=1)

    # Structural Material as encoded integer feature (beam only)
    mat_ids = beam_df["Structural Material"].apply(encode_material).values
    mat_feat = torch.tensor(mat_ids, dtype=torch.float).unsqueeze(1)
    beam_x = torch.cat([beam_x, mat_feat], dim=1)

    # Pad dimensions
    max_dim = max(wall_x.shape[1], beam_x.shape[1])
    wall_x = F.pad(wall_x, (0, max_dim - wall_x.shape[1]))
    beam_x = F.pad(beam_x, (0, max_dim - beam_x.shape[1]))
    
    # Type flag
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
    
    for _, row in bw.iterrows():
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
    
    # Add beam-beam edges
    if bb is not None:
        for _, row in bb.iterrows():
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
    
    # Append support features
    x = torch.cat([
        x,
        support_degree.unsqueeze(1),
        endpoint_support.unsqueeze(1)
    ], dim=1)
    
    # Beam column labels
    beam_col_count = {norm_id(b): 0 for b in beam_ids}
    
    beam_id_col = bc.columns[0]
    
    for _, row in bc.iterrows():
        beam_id = norm_id(row[beam_id_col])
        
        if beam_id not in beam_col_count:
            continue
        
        vals = pd.to_numeric(row.iloc[1:], errors="coerce").fillna(0)
        beam_col_count[beam_id] = int((vals.values > 0).sum())
    
    y = torch.zeros(x.size(0), dtype=torch.long)
    mat_y = torch.full((x.size(0),), -1, dtype=torch.long)   # -1 = wall/unknown
    beam_mask = torch.zeros(x.size(0), dtype=torch.bool)
    
    for i, b in enumerate(beam_ids):
        idx = id_to_idx[b]
        y[idx] = min(beam_col_count[b], 2)
        beam_mask[idx] = True
        mat_y[idx] = int(mat_ids[i])
    
    # Train/val/test masks (beam only)
    beam_indices = torch.where(beam_mask)[0]
    perm = beam_indices[torch.randperm(len(beam_indices))]
    
    n = len(perm)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    train_mask = torch.zeros(x.size(0), dtype=torch.bool)
    val_mask = torch.zeros(x.size(0), dtype=torch.bool)
    test_mask = torch.zeros(x.size(0), dtype=torch.bool)
    
    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True
    
    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        mat_y=mat_y,
        beam_mask=beam_mask,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

def build_stage2_dataset(all_dfs, dataset_path="Dataset/"):
    """Build Stage 2 dataset with BeamBeamMatrix support"""
    
    feat = {}
    bw = {}
    bc = {}
    bb = {}
    
    for filename in all_dfs.keys():
        bid = filename.split("_")[0]
        
        if "FeatureMatrix" in filename:
            feat[bid] = os.path.join(dataset_path, filename)
        if "BeamWallMatrix" in filename:
            bw[bid] = os.path.join(dataset_path, filename)
        if "BeamColumnMatrix" in filename:
            bc[bid] = os.path.join(dataset_path, filename)
        if "BeamBeamMatrix" in filename:
            bb[bid] = os.path.join(dataset_path, filename)
    
    dataset = []
    
    for bid in feat:
        if bid in bw and bid in bc:
            beam_beam_file = bb.get(bid, None)
            try:
                data = load_stage2_graph(feat[bid], bw[bid], bc[bid], beam_beam_file)
                dataset.append(data)
                print(f"  ✅ Loaded {bid}")
            except Exception as e:
                print(f"  ❌ Error loading {bid}: {e}")
    
    return dataset

# ============================================================================
# STAGE 2: MODEL
# ============================================================================

class Stage2BeamModel(torch.nn.Module):
    def __init__(self, in_dim, hidden=32, classes=3):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.head     = torch.nn.Linear(hidden, classes)       # column count: 0/1/2
        self.mat_head = torch.nn.Linear(hidden, N_MATERIALS)   # structural material
    
    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, 0.35, self.training)
        x = F.relu(self.c2(x, edge_index))
        return self.head(x), self.mat_head(x)

def train_stage2(model, loader, class_weights, epochs=350):
    """Train Stage 2 model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.to(DEVICE)
    class_weights = class_weights.to(DEVICE)
    
    for epoch in range(epochs):
        model.train()
        total = 0
        
        for data in loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            logits, mat_logits = model(data.x, data.edge_index)
            
            mask = data.beam_mask
            y = data.y
            
            if mask.sum() == 0:
                continue
            
            loss_main = F.cross_entropy(
                logits[mask],
                y[mask],
                weight=class_weights,
                label_smoothing=0.03
            )
            
            need_col = (y > 0).long()
            
            logits_bin = torch.stack([
                logits[:, 0],
                logits[:, 1] + logits[:, 2]
            ], dim=1)
            
            loss_bin = F.cross_entropy(
                logits_bin[mask],
                need_col[mask]
            )

            # Material prediction loss (only on beams with valid material label)
            mat_mask = mask & (data.mat_y >= 0)
            loss_mat = (
                F.cross_entropy(mat_logits[mat_mask], data.mat_y[mat_mask])
                if mat_mask.sum() > 0 else torch.tensor(0.0, device=DEVICE)
            )
            
            loss = loss_main + 0.7 * loss_bin + 0.3 * loss_mat
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            
            total += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Training Loss: {total:.4f}")
    
    return model

def evaluate_stage2(model, dataset):
    """Evaluate Stage 2 model"""
    model.eval()
    
    tp = [0, 0, 0]
    tot = [0, 0, 0]
    
    all_preds = []
    all_labels = []
    mat_correct = 0
    mat_total = 0
    
    with torch.no_grad():
        for d in dataset:
            d = d.to(DEVICE)
            
            logits, mat_logits = model(d.x, d.edge_index)
            pred = logits.argmax(dim=1)
            mat_pred = mat_logits.argmax(dim=1)
            
            m = d.beam_mask
            y = d.y
            
            all_preds.extend(pred[m].cpu().numpy())
            all_labels.extend(y[m].cpu().numpy())
            
            for c in [0, 1, 2]:
                mc = (y == c) & m
                tot[c] += mc.sum().item()
                tp[c] += ((pred == c) & mc).sum().item()

            # Material accuracy
            mat_mask = m & (d.mat_y >= 0)
            if mat_mask.sum() > 0:
                mat_correct += (mat_pred[mat_mask] == d.mat_y[mat_mask]).sum().item()
                mat_total   += mat_mask.sum().item()
    
    return tp, tot, all_preds, all_labels, mat_correct, mat_total

def topk_recall(dataset, model, K=2):
    """Calculate top-K recall"""
    model.eval()
    hits = 0
    total = 0
    
    with torch.no_grad():
        for d in dataset:
            d = d.to(DEVICE)
            
            logits, _ = model(d.x, d.edge_index)
            
            mask = d.beam_mask
            y = d.y
            
            topk = logits.topk(K, dim=1).indices
            
            correct = (topk == y.unsqueeze(1)).any(dim=1)
            
            hits += (correct & mask).sum().item()
            total += mask.sum().item()
    
    return hits / total if total > 0 else 0

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def load_dataset_from_folder(dataset_path):
    """Load all CSV files from Dataset folder"""
    print(f"\n📂 Loading dataset from: {dataset_path}")
    
    all_dfs = {}
    csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
    
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        try:
            df = pd.read_csv(filepath)
            all_dfs[filename] = df
        except Exception as e:
            print(f"  ❌ Error loading {filename}: {e}")
    
    print(f"\n📊 Total files loaded: {len(all_dfs)}")
    
    # Count file types
    feature_count = sum(1 for f in all_dfs if 'FeatureMatrix' in f)
    beam_wall_count = sum(1 for f in all_dfs if 'BeamWallMatrix' in f)
    beam_column_count = sum(1 for f in all_dfs if 'BeamColumnMatrix' in f)
    beam_beam_count = sum(1 for f in all_dfs if 'BeamBeamMatrix' in f)
    
    print(f"  - FeatureMatrix: {feature_count}")
    print(f"  - BeamWallMatrix: {beam_wall_count}")
    print(f"  - BeamColumnMatrix: {beam_column_count}")
    print(f"  - BeamBeamMatrix: {beam_beam_count}")
    
    return all_dfs

def main():
    print("=" * 80)
    print("STAGE 2: PER-BEAM COLUMN CLASSIFICATION")
    print("=" * 80)
    
    # Load dataset
    all_dfs = load_dataset_from_folder(DATASET_PATH)
    
    # Extract unique families and map to IDs
    print("\n🔍 Extracting unique families...")
    unique_family_per_entities = getUniqueValues(all_dfs)
    family_id_mapping = map_family_names(unique_family_per_entities)
    print(f"  Mapped {len(family_id_mapping)} unique families")
    
    # Add Family_ID column
    print("\n🏷️  Adding Family_ID columns...")
    all_dfs = add_family_id_column(all_dfs, family_id_mapping)
    
    # Build Stage 2 dataset
    print("\n🔨 Building Stage 2 dataset...")
    dataset = build_stage2_dataset(all_dfs, DATASET_PATH)
    print(f"\n  ✅ Created {len(dataset)} graphs")
    
    if len(dataset) == 0:
        print("❌ No valid graphs created. Check your dataset.")
        return
    
    # Analyze class distribution
    print("\n📊 Analyzing class distribution...")
    cnt = Counter()
    for d in dataset:
        cnt.update(d["y"][d["beam_mask"]].tolist())
    
    print(f"  Class counts: {dict(cnt)}")
    
    counts = np.array([cnt[0], cnt[1], cnt[2]], dtype=float)
    weights = counts.max() / counts
    weights = np.clip(weights, 1.0, 6.0)
    
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"  Class weights: {class_weights.numpy()}")
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Initialize model
    in_dim = dataset[0].x.shape[1]
    print(f"\n🤖 Initializing Stage2BeamModel (input dim: {in_dim})")
    model = Stage2BeamModel(in_dim)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print(f"\n🚀 Training Stage 2 model for {STAGE2_EPOCHS} epochs...")
    print("-" * 80)
    model = train_stage2(model, loader, class_weights, epochs=STAGE2_EPOCHS)
    
    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    tp, tot, all_preds, all_labels, mat_correct, mat_total = evaluate_stage2(model, dataset)
    
    print("\n📊 Class-wise Recall:")
    for c in [0, 1, 2]:
        if tot[c] == 0:
            print(f"  Class {c}: N/A (no samples)")
        else:
            recall = tp[c] / tot[c]
            print(f"  Class {c}: {recall:.4f} ({tp[c]}/{tot[c]})")
    
    # Overall accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = np.mean(all_preds == all_labels)
    print(f"\n📈 Overall Accuracy: {accuracy:.4f} ({np.sum(all_preds == all_labels)}/{len(all_labels)})")
    
    # Material accuracy
    mat_acc = mat_correct / mat_total if mat_total > 0 else 0
    print(f"🔩 Material Accuracy: {mat_acc:.4f} ({mat_correct}/{mat_total})")
    
    # Top-K recall
    topk = topk_recall(dataset, model, K=2)
    print(f"📈 Top-2 Recall: {topk:.4f}")
    
    # Confusion matrix
    print("\n📊 Confusion Matrix:")
    print("     Pred:  0    1    2")
    for true_class in [0, 1, 2]:
        mask = all_labels == true_class
        counts = [np.sum((all_preds == pred_class) & mask) for pred_class in [0, 1, 2]]
        print(f"  True {true_class}: {counts[0]:4d} {counts[1]:4d} {counts[2]:4d}")
    
    # Save model
    model_path = "stage2_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'in_dim': in_dim,
        'accuracy': accuracy,
        'mat_accuracy': mat_acc,
        'material_mapping': MATERIAL_MAPPING,
        'class_weights': class_weights,
        'class_recalls': [tp[c] / tot[c] if tot[c] > 0 else 0 for c in [0, 1, 2]]
    }, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()
