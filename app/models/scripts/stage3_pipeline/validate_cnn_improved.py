#!/usr/bin/env python3
"""
CNN VALIDATION (NO DATA LEAKAGE)
Runs on all files inside validation folder
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "column_predictor_no_leakage.pth"

# Folder containing validation CSV files
VALIDATION_DIR = "Dataset/validationTest/ml_signal"

TOLERANCE = 5
PEAK_THRESHOLD = 0.10


# ============================================================
# CNN MODEL (same architecture as training)
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
# Load CNN
# ============================================================

print("Loading CNN model...")

model = ImprovedColumnPredictor().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded")


# ============================================================
# Utility functions
# ============================================================

def extract_peaks(signal, threshold=None):
    t = threshold if threshold is not None else PEAK_THRESHOLD
    peaks, _ = find_peaks(signal, height=t, distance=5)
    peaks = list(peaks)

    if signal[0] >= t and (len(signal) < 2 or signal[0] >= signal[1]):
        if 0 not in peaks:
            peaks = [0] + peaks

    if signal[-1] >= t and (len(signal) < 2 or signal[-1] >= signal[-2]):
        if (len(signal) - 1) not in peaks:
            peaks = peaks + [len(signal) - 1]

    return np.array(peaks)


def extract_gt_positions(signal, threshold=0.5):

    above = np.where(signal >= threshold)[0]

    if len(above) == 0:
        return np.array([])

    positions = []
    region_start = above[0]
    prev = above[0]

    for idx in above[1:]:

        if idx - prev > 3:
            positions.append((region_start + prev) // 2)
            region_start = idx

        prev = idx

    positions.append((region_start + prev) // 2)

    return np.array(positions)


def match_peaks(pred,true):

    matched = 0

    for t in true:

        for p in pred:

            if abs(t-p) <= TOLERANCE:
                matched += 1
                break

    return matched


# ============================================================
# Validation
# ============================================================

total_gt = 0
total_pred = 0
total_correct = 0

files = [f for f in os.listdir(VALIDATION_DIR) if f.endswith(".csv")]

print("\nValidation files found:",len(files))

for file in files:

    path = os.path.join(VALIDATION_DIR,file)

    print("\nProcessing:",file)

    df = pd.read_csv(path)

    for _,row in df.iterrows():

        wall_signal = np.array(eval(row["wall_signal"]))
        beam_signal = np.array(eval(row["beam_signal"]))
        column_signal = np.array(eval(row["column_signal"]))

        X = np.stack([wall_signal,beam_signal])

        X_tensor = torch.tensor(X,dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_signal = model(X_tensor).cpu().numpy()[0]

        pred_peaks = extract_peaks(pred_signal)
        true_peaks = extract_gt_positions(column_signal)

        correct = match_peaks(pred_peaks,true_peaks)

        total_correct += correct
        total_gt += len(true_peaks)
        total_pred += len(pred_peaks)


# ============================================================
# Metrics
# ============================================================

precision = total_correct / total_pred if total_pred>0 else 0
recall = total_correct / total_gt if total_gt>0 else 0

f1 = 2*(precision*recall)/(precision+recall+1e-6)

print("\n===============================")
print("VALIDATION RESULTS")
print("===============================")

print("GT Columns:",total_gt)
print("Predicted:",total_pred)
print("Correct:",total_correct)

print("Precision:",round(precision,3))
print("Recall:",round(recall,3))
print("F1:",round(f1,3))