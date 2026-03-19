import pandas as pd
import numpy as np
import os
import glob

# -----------------------------
# PARAMETERS (FEET)
# -----------------------------
INPUT_FOLDER = r"Dataset/validation"
OUTPUT_FOLDER = r"Dataset/validation/BeamSegmentation"
SEGMENT_SIZE = 1.5
NEIGHBOR_ZERO = 4

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# GEOMETRY FUNCTIONS
# -----------------------------
def beam_length(row):
    return np.sqrt(
        (row["End X"] - row["Start X"])**2 +
        (row["End Y"] - row["Start Y"])**2 +
        (row["End Z"] - row["Start Z"])**2
    )

def project_point_on_beam(point, beam_start, beam_end):
    beam_vec = beam_end - beam_start
    length_sq = np.dot(beam_vec, beam_vec)

    if length_sq < 1e-6:
        return 0.0

    proj = np.dot(point - beam_start, beam_vec) / length_sq
    return np.clip(proj, 0.0, 1.0)

def closest_point_between_segments(p1, p2, q1, q2):
    p1, p2, q1, q2 = map(np.array, (p1, p2, q1, q2))

    u = p2 - p1
    v = q2 - q1
    w0 = p1 - q1

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)

    denom = a*c - b*b

    if abs(denom) < 1e-6:
        return p1

    s = (b*e - c*d) / denom
    t = (a*e - b*d) / denom

    s = np.clip(s, 0, 1)
    t = np.clip(t, 0, 1)

    pt1 = p1 + s*u
    pt2 = q1 + t*v

    return (pt1 + pt2) / 2

# -----------------------------
# WALL PROPAGATION
# -----------------------------
def propagate_wall(vector, has_wall):
    if not has_wall:
        return vector

    if "W" not in vector:
        return vector

    return ["W" if v == "0" else v for v in vector]

# -----------------------------
# NEIGHBOR RULE (FINAL)
# -----------------------------
def apply_neighbor_zero_rule(vector, n):
    n_segments = len(vector)
    final = vector.copy()

    # Step 1: Strong blockers (B & C)
    blockers = [i for i, v in enumerate(vector) if v in ["B", "C"]]

    blocked = set()
    for idx in blockers:
        for j in range(idx - n, idx + n + 1):
            if 0 <= j < n_segments:
                blocked.add(j)

    for i in blocked:
        if final[i] not in ["B", "C"]:
            final[i] = "0"

    # Step 2: Space out W
    last_w = -np.inf
    for i in range(n_segments):
        if final[i] == "W":
            if i - last_w <= n:
                final[i] = "0"
            else:
                last_w = i

    return final

# -----------------------------
# WALL ENDPOINT ENFORCEMENT
# -----------------------------
def enforce_wall_endpoints(vector, has_wall):
    if not has_wall:
        return vector

    if "W" not in vector:
        return vector

    if vector.count("W") == 1:
        vector[-1] = "W"

    return vector

# -----------------------------
# MAIN PROCESS
# -----------------------------
feature_files = glob.glob(os.path.join(INPUT_FOLDER, "*_FeatureMatrix.csv"))
building_ids = [os.path.basename(f).split("_")[0] for f in feature_files]

for bid in building_ids:

    print("Processing:", bid)

    feature = pd.read_csv(f"{INPUT_FOLDER}/{bid}_FeatureMatrix.csv")

    beam_wall = pd.read_csv(f"{INPUT_FOLDER}/{bid}_BeamWallMatrix.csv").set_index("Unnamed: 0")
    beam_column = pd.read_csv(f"{INPUT_FOLDER}/{bid}_BeamColumnMatrix.csv").set_index("Unnamed: 0")
    beam_beam = pd.read_csv(f"{INPUT_FOLDER}/{bid}_BeamBeamMatrix.csv").set_index("Unnamed: 0")

    rows = []

    beams = feature[feature["Element Type"] == "Structural Framing"]

    for _, beam in beams.iterrows():

        beam_id = beam["Element ID"]
        full_beam_id = f"{bid}_{beam_id}_B"

        length = beam_length(beam)
        n_segments = max(2, int(np.ceil(length / SEGMENT_SIZE)))

        vector = ["0"] * n_segments

        beam_start = np.array([beam["Start X"], beam["Start Y"], beam["Start Z"]])
        beam_end = np.array([beam["End X"], beam["End Y"], beam["End Z"]])

        priority = {"0": 0, "W": 1, "B": 2, "C": 3}

        def assign(idx, label):
            idx = max(0, min(n_segments - 1, idx))
            if priority[label] > priority[vector[idx]]:
                vector[idx] = label

        # -------------------------
        # COLUMN
        # -------------------------
        if full_beam_id in beam_column.index:
            cols = beam_column.loc[full_beam_id]
            for col_full in cols[cols == 1].index:
                col_id = int(col_full.split("_")[1])
                col = feature[feature["Element ID"] == col_id]

                if col.empty:
                    continue

                col = col.iloc[0]
                point = np.array([col["Start X"], col["Start Y"], col["Start Z"]])

                rel = project_point_on_beam(point, beam_start, beam_end)
                assign(int(round(rel * (n_segments - 1))), "C")

        # -------------------------
        # BEAM-BEAM
        # -------------------------
        if full_beam_id in beam_beam.index:
            others = beam_beam.loc[full_beam_id]
            for other_full in others[others == 1].index:
                other_id = int(other_full.split("_")[1])
                other = feature[feature["Element ID"] == other_id]

                if other.empty:
                    continue

                other = other.iloc[0]

                q1 = np.array([other["Start X"], other["Start Y"], other["Start Z"]])
                q2 = np.array([other["End X"], other["End Y"], other["End Z"]])

                inter = closest_point_between_segments(beam_start, beam_end, q1, q2)
                rel = project_point_on_beam(inter, beam_start, beam_end)

                assign(int(round(rel * (n_segments - 1))), "B")

        # -------------------------
        # WALL
        # -------------------------
        if full_beam_id in beam_wall.index:
            walls = beam_wall.loc[full_beam_id]
            for wall_full in walls[walls == 1].index:
                wall_id = int(wall_full.split("_")[1])
                wall = feature[feature["Element ID"] == wall_id]

                if wall.empty:
                    continue

                wall = wall.iloc[0]

                w1 = np.array([wall["Start X"], wall["Start Y"], wall["Start Z"]])
                w2 = np.array([wall["End X"], wall["End Y"], wall["End Z"]])

                inter = closest_point_between_segments(beam_start, beam_end, w1, w2)
                rel = project_point_on_beam(inter, beam_start, beam_end)

                assign(int(round(rel * (n_segments - 1))), "W")

        # -------------------------
        # COUNT CORRECTION
        # -------------------------
        def enforce_count(label, expected):
            current = vector.count(label)
            for i in range(n_segments):
                if current >= expected:
                    break
                if vector[i] == "0":
                    vector[i] = label
                    current += 1

        if full_beam_id in beam_column.index:
            enforce_count("C", int(beam_column.loc[full_beam_id].sum()))

        if full_beam_id in beam_beam.index:
            enforce_count("B", int(beam_beam.loc[full_beam_id].sum()))

        if full_beam_id in beam_wall.index:
            enforce_count("W", int(beam_wall.loc[full_beam_id].sum()))

        # -------------------------
        # WALL PROPAGATION
        # -------------------------
        has_wall_connection = (
            full_beam_id in beam_wall.index and
            beam_wall.loc[full_beam_id].sum() > 0
        )

        vector = propagate_wall(vector, has_wall_connection)

        # -------------------------
        # APPLY NEIGHBOR RULE
        # -------------------------
        vector = apply_neighbor_zero_rule(vector, NEIGHBOR_ZERO)

        # -------------------------
        # ENFORCE WALL ENDPOINTS
        # -------------------------
        vector = enforce_wall_endpoints(vector, has_wall_connection)

        # -------------------------
        # SAVE
        # -------------------------
        rows.append({
            "building_id": bid,
            "beam_id": full_beam_id,
            "beam_length": length,
            "segments": n_segments,
            "segment_vector": "[" + ",".join(vector) + "]"
        })

    result = pd.DataFrame(rows)
    result.to_csv(f"{OUTPUT_FOLDER}/{bid}_BeamSegmentVector.csv", index=False)

    print("Saved:", bid)

print("Done.")