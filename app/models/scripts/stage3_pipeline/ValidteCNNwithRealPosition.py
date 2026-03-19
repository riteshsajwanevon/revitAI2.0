#!/usr/bin/env python3

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import ast
from scipy.signal import find_peaks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "column_predictor_no_leakage.pth"

# Validation folder paths
VALIDATION_SIGNAL_DIR = "Dataset/validation/ml_signal"
FEATURE_DIR = "Dataset/validation"

PEAK_THRESHOLD = 0.35 #0.10


# ============================================================
# CNN MODEL
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
# Load Model
# ============================================================

print("Loading CNN model...")

model = ImprovedColumnPredictor().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded")


# ============================================================
# Utility Functions
# ============================================================

def extract_peaks(signal):

    peaks,_ = find_peaks(signal,height=PEAK_THRESHOLD,distance=5)

    # include edges
    if signal[0] >= PEAK_THRESHOLD:
        peaks = np.insert(peaks,0,0)

    if signal[-1] >= PEAK_THRESHOLD:
        peaks = np.append(peaks,len(signal)-1)

    return peaks


def index_to_xyz(idx, beam_row):

    sx, sy, sz = beam_row["Start X"], beam_row["Start Y"], beam_row["Start Z"]
    ex, ey, ez = beam_row["End X"], beam_row["End Y"], beam_row["End Z"]

    relative = idx / 127.0

    x = sx + relative * (ex - sx)
    y = sy + relative * (ey - sy)
    z = sz + relative * (ez - sz)

    return x,y,z


# ============================================================
# Convert Predictions to XYZ
# ============================================================

results = []

signal_files = [
    f for f in os.listdir(VALIDATION_SIGNAL_DIR)
    if f.endswith("_BeamSignal128.csv")
]

print("Found validation buildings:",len(signal_files))

for file in signal_files:

    bid = file.split("_")[0]

    print("\nProcessing building:",bid)

    signal_file = os.path.join(VALIDATION_SIGNAL_DIR,file)
    feature_file = f"{FEATURE_DIR}/{bid}_FeatureMatrix.csv"

    if not os.path.exists(feature_file):
        feature_file = f"{FEATURE_DIR}/validation/{bid}_FeatureMatrix.csv"

    if not os.path.exists(feature_file):
        print("FeatureMatrix missing")
        continue

    signal_df = pd.read_csv(signal_file)
    feature_df = pd.read_csv(feature_file)

    beam_df = feature_df[
        feature_df["Element Type"].str.contains("Beam|Framing", case=False)
    ]

    beam_map = {
        str(row["Element ID"]): row
        for _,row in beam_df.iterrows()
    }

    for _,row in signal_df.iterrows():

        beam_id = str(row["beam_id"]).split("_")[1] if "_" in str(row["beam_id"]) else str(row["beam_id"])

        if beam_id not in beam_map:
            continue

        beam_row = beam_map[beam_id]

        wall_signal = np.array(ast.literal_eval(row["wall_signal"]))
        beam_signal = np.array(ast.literal_eval(row["beam_signal"]))

        X = np.stack([wall_signal,beam_signal])

        X_tensor = torch.tensor(X,dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_signal = model(X_tensor).cpu().numpy()[0]

        peaks = extract_peaks(pred_signal)

        for idx in peaks:

            x,y,z = index_to_xyz(idx, beam_row)

            results.append({
                "building_id":bid,
                "beam_id":row["beam_id"],
                "signal_index":int(idx),
                "x":x,
                "y":y,
                "z":z
            })


# ============================================================
# Save Results
# ============================================================

output_file = "ColumnPredictionwithGap4_stage2pipline.csv"

pd.DataFrame(results).to_csv(output_file,index=False)

print("\nSaved predicted column coordinates to:",output_file)