#!/usr/bin/env python3
"""
BeamSegmenationUsingConnection.py
Updated version of BeamSegmentation.py

Changes vs original:
  - Reads connections from Dataset/Beam_Connections_TXT/{bid}_Connections.txt
    instead of loading BeamWallMatrix / BeamColumnMatrix / BeamBeamMatrix CSVs
  - Column positions still use actual XYZ coordinates from FeatureMatrix
    (projected onto beam axis) — same accurate method as before
  - Wall and beam-intersection positions use linspace (only IDs available in txt)
"""

import pandas as pd
import numpy as np
import os
import glob

# ── Parameters ───────────────────────────────────────────────
FEATURE_FOLDER     = "Dataset/"
CONNECTIONS_FOLDER = "Dataset/Beam_Connections_TXT/"
OUTPUT_FOLDER      = "Dataset/BeamSegmentation_v2/"
SEGMENT_SIZE       = 0.5

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ── Geometry helpers ─────────────────────────────────────────

def calc_beam_length(row):
    sx, sy, sz = row["Start X"], row["Start Y"], row["Start Z"]
    ex, ey, ez = row["End X"],   row["End Y"],   row["End Z"]
    return np.sqrt((ex-sx)**2 + (ey-sy)**2 + (ez-sz)**2)

def project_point_on_beam(point_xyz, beam_start, beam_end):
    """Return relative position (0–1) of point projected onto beam axis."""
    beam_vec = beam_end - beam_start
    length_sq = np.dot(beam_vec, beam_vec)
    if length_sq < 1e-6:
        return 0.0
    t = np.dot(point_xyz - beam_start, beam_vec) / length_sq
    return float(np.clip(t, 0.0, 1.0))

# ── Connection file parser ────────────────────────────────────

def parse_connections(txt_path):
    """
    Parse a Connections.txt file into three dicts:
      beam_beam   {beam_id: [connected_beam_ids]}
      beam_column {beam_id: [connected_column_ids]}
      beam_wall   {beam_id: [connected_wall_ids]}
    All IDs are plain integers (strings stripped).
    """
    beam_beam   = {}
    beam_column = {}
    beam_wall   = {}

    current_section = None

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line == "BEAM-BEAM":
                current_section = "bb"
            elif line == "BEAM-COLUMN":
                current_section = "bc"
            elif line == "BEAM-WALL":
                current_section = "bw"
            elif "->" in line:
                left, right = line.split("->", 1)
                src = left.strip()
                targets = [t.strip() for t in right.split(",") if t.strip()]

                if current_section == "bb":
                    beam_beam.setdefault(src, []).extend(targets)
                elif current_section == "bc":
                    beam_column.setdefault(src, []).extend(targets)
                elif current_section == "bw":
                    beam_wall.setdefault(src, []).extend(targets)

    return beam_beam, beam_column, beam_wall

# ── Main loop ────────────────────────────────────────────────

# Collect feature files from both main folder and validation subfolder
feature_files = (
    glob.glob(os.path.join(FEATURE_FOLDER, "*_FeatureMatrix.csv")) +
    glob.glob(os.path.join(FEATURE_FOLDER, "validation", "*_FeatureMatrix.csv"))
)
building_ids  = sorted(set(os.path.basename(f).split("_")[0] for f in feature_files))

print(f"Found {len(building_ids)} buildings")

processed = 0
skipped   = 0

for bid in building_ids:

    conn_file    = os.path.join(CONNECTIONS_FOLDER, f"{bid}_Connections.txt")
    # Check main folder first, then validation subfolder
    feature_file = os.path.join(FEATURE_FOLDER, f"{bid}_FeatureMatrix.csv")
    if not os.path.exists(feature_file):
        feature_file = os.path.join(FEATURE_FOLDER, "validation", f"{bid}_FeatureMatrix.csv")

    if not os.path.exists(conn_file):
        print(f"  [{bid}] ⚠️  No connection file — skipping")
        skipped += 1
        continue

    print(f"  [{bid}] Processing...")

    # Load feature matrix
    feature = pd.read_csv(feature_file)
    feature = feature.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Build element ID → row lookup (integer IDs)
    feature["_eid"] = feature["Element ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    eid_lookup = feature.set_index("_eid")

    # Parse connections
    beam_beam, beam_column, beam_wall = parse_connections(conn_file)

    # Get all beams from FeatureMatrix
    beams = feature[feature["Element Type"] == "Structural Framing"]

    rows = []

    for _, beam in beams.iterrows():
        beam_eid = str(int(beam["Element ID"]))   # plain integer string

        length     = calc_beam_length(beam)
        n_segments = max(1, int(np.ceil(length / SEGMENT_SIZE)))
        vector     = ["0"] * n_segments

        beam_start = np.array([beam["Start X"], beam["Start Y"], beam["Start Z"]])
        beam_end   = np.array([beam["End X"],   beam["End Y"],   beam["End Z"]])

        # ── WALL SUPPORT ──────────────────────────────────────
        walls = beam_wall.get(beam_eid, [])
        if walls:
            positions = np.linspace(0, n_segments - 1, len(walls))
            for p in positions:
                vector[int(p)] = "W"

        # ── COLUMN SUPPORT (actual XYZ projection) ────────────
        columns = beam_column.get(beam_eid, [])
        for col_eid in columns:
            if col_eid in eid_lookup.index:
                col_row = eid_lookup.loc[col_eid]
                # handle duplicate index (take first)
                if isinstance(col_row, pd.DataFrame):
                    col_row = col_row.iloc[0]
                col_xyz = np.array([col_row["Start X"], col_row["Start Y"], col_row["Start Z"]])
                rel_pos = project_point_on_beam(col_xyz, beam_start, beam_end)
                seg_idx = min(int(rel_pos * (n_segments - 1)), n_segments - 1)
                vector[seg_idx] = "C"

        # ── BEAM INTERSECTION ─────────────────────────────────
        intersecting = beam_beam.get(beam_eid, [])
        if intersecting:
            positions = np.linspace(0, n_segments - 1, len(intersecting))
            for p in positions:
                if vector[int(p)] != "C":   # don't overwrite columns
                    vector[int(p)] = "B"

        rows.append({
            "building_id":    bid,
            "beam_id":        beam_eid,
            "beam_length":    round(length, 4),
            "segments":       n_segments,
            "segment_vector": "[" + ",".join(vector) + "]"
        })

    result = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_FOLDER, f"{bid}_BeamSegmentVector.csv")
    result.to_csv(out_path, index=False)
    print(f"    → {len(rows)} beams saved to {out_path}")
    processed += 1

print(f"\nDone. Processed: {processed}  Skipped: {skipped}")
