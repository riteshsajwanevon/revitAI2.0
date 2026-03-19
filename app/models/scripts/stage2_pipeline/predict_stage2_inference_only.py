#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2 Inference Script (Batch Mode)

Runs inference for ALL buildings inside:
Dataset/test/

Does NOT use BeamColumnMatrix (ground truth).

Outputs:
- Individual prediction CSV per building
- Combined CSV for all buildings
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path

# ============================================================
# MATERIAL ENCODING
# ============================================================

MATERIAL_MAPPING = {
    "Metal - Steel 43-275": 0,
    "Metal - Scaffold Tube": 1,
    "Steel 50-355": 2,
    "Steel 43-275": 3,
    "Stainless Steel": 4,
    "Metal - Steel 43-275 - existing": 5,
    "Steel 43-275 EXISTING": 6,
    "Unknown": 7,
}

N_MATERIALS = len(MATERIAL_MAPPING)
MATERIAL_NAMES = {v: k for k, v in MATERIAL_MAPPING.items()}


def encode_material(val):
    v = str(val).strip() if pd.notna(val) else "Unknown"
    return MATERIAL_MAPPING.get(v, MATERIAL_MAPPING["Unknown"])


# ============================================================
# MODEL
# ============================================================

class Stage2BeamModel(torch.nn.Module):
    def __init__(self, in_dim, hidden=32, classes=3):
        super().__init__()

        self.c1 = SAGEConv(in_dim, hidden)
        self.c2 = SAGEConv(hidden, hidden)

        self.head = torch.nn.Linear(hidden, classes)
        self.mat_head = torch.nn.Linear(hidden, N_MATERIALS)

    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, 0.35, self.training)
        x = F.relu(self.c2(x, edge_index))

        return self.head(x), self.mat_head(x)


# ============================================================
# UTIL
# ============================================================

def norm_id(x):
    x = str(x).strip().replace(".0", "")
    nums = re.findall(r'\d+', x)

    if len(nums) >= 2:
        return nums[1]
    if len(nums) == 1:
        return nums[0]
    return x


# ============================================================
# GRAPH LOADING
# ============================================================

def load_graph_for_inference(feature_csv, beam_wall_csv, beam_beam_csv=None):

    df = pd.read_csv(feature_csv)
    bw = pd.read_csv(beam_wall_csv)
    bb = pd.read_csv(beam_beam_csv) if beam_beam_csv else None

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    df = df[~df["Element Type"].str.contains("Column", case=False, na=False)]

    wall_df = df[df["Element Type"].str.contains("Wall", case=False, na=False)]
    beam_df = df[df["Element Type"].str.contains("Beam|Framing", case=False, na=False)]

    WALL_FEATS = ["Start X","Start Y","Start Z","End X","End Y","End Z","Length","Area"]
    BEAM_FEATS = ["Start X","Start Y","Start Z","End X","End Y","End Z",
                  "Length","Volume","Entity Start Level","Entity End Level"]

    wall_x = torch.tensor(wall_df[WALL_FEATS].values, dtype=torch.float)
    beam_x = torch.tensor(beam_df[BEAM_FEATS].values, dtype=torch.float)

    # Beam length normalization
    bl = beam_df["Length"].values
    bl_norm = bl / (np.mean(bl) + 1e-6)
    bl_feat = torch.tensor(bl_norm, dtype=torch.float).unsqueeze(1)

    beam_x = torch.cat([beam_x, bl_feat], dim=1)

    # Material feature
    mat_ids = beam_df["Structural Material"].apply(encode_material).values
    mat_feat = torch.tensor(mat_ids, dtype=torch.float).unsqueeze(1)

    beam_x = torch.cat([beam_x, mat_feat], dim=1)

    max_dim = max(wall_x.shape[1], beam_x.shape[1])

    wall_x = F.pad(wall_x, (0, max_dim - wall_x.shape[1]))
    beam_x = F.pad(beam_x, (0, max_dim - beam_x.shape[1]))

    wall_x = torch.cat([wall_x, torch.ones((wall_x.size(0), 1))], dim=1)
    beam_x = torch.cat([beam_x, torch.zeros((beam_x.size(0), 1))], dim=1)

    x = torch.cat([wall_x, beam_x], dim=0)

    beam_ids = beam_df["Element ID"].astype(str).apply(norm_id).tolist()
    wall_ids = wall_df["Element ID"].astype(str).apply(norm_id).tolist()

    node_ids = wall_ids + beam_ids
    id_to_idx = {eid: i for i, eid in enumerate(node_ids)}

    edges = []
    support_degree = torch.zeros(len(node_ids))
    endpoint_support = torch.zeros(len(node_ids))

    # Beam-Wall edges
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

    # Beam-Beam edges
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

    x = torch.cat([
        x,
        support_degree.unsqueeze(1),
        endpoint_support.unsqueeze(1)
    ], dim=1)

    beam_mask = torch.zeros(x.size(0), dtype=torch.bool)

    for b in beam_ids:
        beam_mask[id_to_idx[b]] = True

    return Data(
        x=x,
        edge_index=edge_index,
        beam_mask=beam_mask
    ), beam_ids


# ============================================================
# MODEL LOAD
# ============================================================

def load_model(model_path):

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    model = Stage2BeamModel(checkpoint['in_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    print(f"Model loaded | Accuracy {checkpoint['accuracy']:.2%}")

    return model


# ============================================================
# PREDICT SINGLE BUILDING
# ============================================================

def predict_building(model, feature_csv, bw_csv, bb_csv=None):

    data, beam_ids = load_graph_for_inference(feature_csv, bw_csv, bb_csv)

    with torch.no_grad():

        logits, mat_logits = model(data.x, data.edge_index)

        probs = torch.softmax(logits, dim=1)
        mat_probs = torch.softmax(mat_logits, dim=1)

        pred = logits.argmax(dim=1)
        mat_pred = mat_logits.argmax(dim=1)

    beam_preds = pred[data.beam_mask]
    beam_probs = probs[data.beam_mask]

    beam_mat = mat_pred[data.beam_mask]
    beam_mat_probs = mat_probs[data.beam_mask]

    results = []

    for beam_id, cls, p, m, mp in zip(
        beam_ids, beam_preds, beam_probs, beam_mat, beam_mat_probs
    ):

        results.append({
            "beam_id": beam_id,
            "predicted_columns": int(cls.item()),
            "confidence": float(p.max().item()),
            "material": MATERIAL_NAMES[int(m.item())],
            "material_confidence": float(mp.max().item())
        })

    return results


# ============================================================
# DATASET DISCOVERY
# ============================================================

def find_buildings(dataset_path):

    feature_files = Path(dataset_path).glob("*_FeatureMatrix.csv")

    building_ids = []

    for f in feature_files:
        building_ids.append(f.name.split("_")[0])

    return sorted(set(building_ids))


# ============================================================
# MAIN
# ============================================================

def main():

    ROOT = Path(__file__).resolve().parent.parent
    DATASET = ROOT / "Dataset" / "test"

    MODEL_PATH = ROOT / "Models" / "stage2_model.pth"

    OUTPUT_DIR = ROOT / "result" / "predict_stage2_inference"
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\nScanning dataset:", DATASET)

    building_ids = find_buildings(DATASET)

    print(f"Found {len(building_ids)} buildings\n")

    model = load_model(MODEL_PATH)

    all_predictions = []

    for bid in building_ids:

        print(f"\nProcessing building {bid}")

        feature = DATASET / f"{bid}_FeatureMatrix.csv"
        bw = DATASET / f"{bid}_BeamWallMatrix.csv"
        bb = DATASET / f"{bid}_BeamBeamMatrix.csv"

        if not feature.exists() or not bw.exists():
            print("Missing required files — skipping")
            continue

        if not bb.exists():
            bb = None

        results = predict_building(model, feature, bw, bb)

        df = pd.DataFrame(results)

        df["building_id"] = bid

        df.to_csv(OUTPUT_DIR / f"{bid}_predictions.csv", index=False)

        all_predictions.append(df)

    if all_predictions:

        combined = pd.concat(all_predictions, ignore_index=True)

        combined.to_csv(
            OUTPUT_DIR / "all_building_predictions.csv",
            index=False
        )

        print("\nCombined predictions saved.")

    print("\nInference finished.")


if __name__ == "__main__":
    main()