#!/usr/bin/env python3
"""
Plot CNN prediction vs ground truth for sample beams from validation buildings.
Saves plots to cnn_pipeline/plots/
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import ast
import os
import matplotlib
matplotlib.use("Agg")  # headless - no display needed
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ── Config ──────────────────────────────────────────────────
MODEL_PATH   = "column_predictor_no_leakage.pth"
SIGNAL_DIR   = "Dataset/ml_signal"
OUTPUT_DIR   = "cnn_pipeline/plots"
BUILDINGS    = ["20250034", "20250041", "20250050", "20250079", "20250082"]
BEAMS_PER_BUILDING = 3   # plot first N beams per building
THRESHOLD    = 0.10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Model ────────────────────────────────────────────────────
class ImprovedColumnPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, 7, padding=3);  self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2); self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, 5, padding=2);self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 64, 5, padding=2); self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 1, 1)
        self.relu = nn.ReLU(); self.dropout = nn.Dropout(0.1); self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout(self.relu(self.bn4(self.conv4(x))))
        return self.sigmoid(self.conv5(x)).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedColumnPredictor().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded from {MODEL_PATH}")

# ── GT position extractor (plateau-aware) ───────────────────
def gt_positions(signal, threshold=0.5):
    above = np.where(signal >= threshold)[0]
    if len(above) == 0:
        return []
    positions = []
    region_start = above[0]
    prev = above[0]
    for idx in above[1:]:
        if idx - prev > 3:
            positions.append((region_start + prev) // 2)
            region_start = idx
        prev = idx
    positions.append((region_start + prev) // 2)
    return positions

# ── Plot loop ────────────────────────────────────────────────
total_plots = 0

for bid in BUILDINGS:
    signal_file = f"{SIGNAL_DIR}/{bid}_BeamSignal128.csv"
    if not os.path.exists(signal_file):
        print(f"  Skipping {bid} — signal file not found")
        continue

    df = pd.read_csv(signal_file)
    plotted = 0

    for i, (_, row) in enumerate(df.iterrows()):
        if plotted >= BEAMS_PER_BUILDING:
            break

        wall_sig   = np.array(ast.literal_eval(row["wall_signal"]))
        beam_sig   = np.array(ast.literal_eval(row["beam_signal"]))
        column_sig = np.array(ast.literal_eval(row["column_signal"]))

        X = torch.tensor(np.stack([wall_sig, beam_sig]), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_signal = model(X).cpu().numpy()[0]

        gt_pos   = gt_positions(column_sig)
        pred_pos, _ = find_peaks(pred_signal, height=THRESHOLD, distance=5)

        x = np.arange(128)

        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        fig.suptitle(f"Building {bid} — Beam {i}  |  GT columns: {len(gt_pos)}  Predicted: {len(pred_pos)}", fontsize=11)

        # Input channel 0: wall
        axes[0].fill_between(x, wall_sig, alpha=0.6, color="steelblue")
        axes[0].set_ylabel("Wall signal")
        axes[0].set_ylim(-0.1, 1.2)

        # Input channel 1: beam intersections
        axes[1].fill_between(x, beam_sig, alpha=0.6, color="orange")
        axes[1].set_ylabel("Beam signal")
        axes[1].set_ylim(-0.1, 1.2)

        # Prediction vs GT
        axes[2].plot(x, column_sig, color="green",  linewidth=1.5, label="Ground truth")
        axes[2].plot(x, pred_signal, color="red",   linewidth=1.5, label="CNN prediction", linestyle="--")
        for gp in gt_pos:
            axes[2].axvline(gp, color="green", alpha=0.4, linewidth=1)
        for pp in pred_pos:
            axes[2].axvline(pp, color="red", alpha=0.4, linewidth=1, linestyle=":")
        axes[2].set_ylabel("Column signal")
        axes[2].set_xlabel("Position along beam (0–127)")
        axes[2].set_ylim(-0.05, 1.1)
        axes[2].legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"{bid}_beam{i}.png")
        plt.savefig(out_path, dpi=120)
        plt.close()

        print(f"  Saved: {out_path}  (GT={len(gt_pos)}, Pred={len(pred_pos)})")
        plotted += 1
        total_plots += 1

print(f"\nDone. {total_plots} plots saved to {OUTPUT_DIR}/")
