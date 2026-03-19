import pandas as pd
import numpy as np
import os
import glob

# -----------------------------
# PARAMETERS
# -----------------------------
INPUT_FOLDER = r"Dataset/validation"
OUTPUT_FOLDER = r"Dataset/validation/BeamSegmentation"
SEGMENT_SIZE = 0.5

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def beam_length(row):
    sx, sy, sz = row["Start X"], row["Start Y"], row["Start Z"]
    ex, ey, ez = row["End X"], row["End Y"], row["End Z"]

    return np.sqrt((ex-sx)**2 + (ey-sy)**2 + (ez-sz)**2)

def project_point_on_beam(point_xyz, beam_start, beam_end):
    """
    Project a point onto a beam and return relative position (0-1).
    
    Args:
        point_xyz: [x, y, z] coordinates of point (e.g., column)
        beam_start: [x, y, z] coordinates of beam start
        beam_end: [x, y, z] coordinates of beam end
    
    Returns:
        relative_position: 0-1 value (0=start, 1=end)
    """
    beam_vec = beam_end - beam_start
    beam_length_sq = np.dot(beam_vec, beam_vec)
    
    if beam_length_sq < 1e-6:  # Avoid division by zero
        return 0.0
    
    point_vec = point_xyz - beam_start
    projection = np.dot(point_vec, beam_vec) / beam_length_sq
    
    # Clamp to [0, 1] range
    return np.clip(projection, 0.0, 1.0)


feature_files = glob.glob(os.path.join(INPUT_FOLDER,"*_FeatureMatrix.csv"))
building_ids = [os.path.basename(f).split("_")[0] for f in feature_files]

print("Buildings:", building_ids)


for bid in building_ids:

    print("Processing:", bid)

    feature = pd.read_csv(f"{INPUT_FOLDER}/{bid}_FeatureMatrix.csv")
    beam_wall = pd.read_csv(f"{INPUT_FOLDER}/{bid}_BeamWallMatrix.csv")
    beam_column = pd.read_csv(f"{INPUT_FOLDER}/{bid}_BeamColumnMatrix.csv")
    beam_beam = pd.read_csv(f"{INPUT_FOLDER}/{bid}_BeamBeamMatrix.csv")

    beam_wall = beam_wall.set_index("Unnamed: 0")
    beam_column = beam_column.set_index("Unnamed: 0")
    beam_beam = beam_beam.set_index("Unnamed: 0")

    rows = []

    # --------------------------------------------------
    # Iterate beams
    # --------------------------------------------------

    beams = feature[feature["Element Type"]=="Structural Framing"]

    for _, beam in beams.iterrows():

        beam_id = beam["Element ID"]

        full_beam_id = f"{bid}_{beam_id}_B"

        length = beam_length(beam)

        n_segments = int(np.ceil(length / SEGMENT_SIZE))

        vector = ["0"] * n_segments

        # ---------------------------
        # WALL SUPPORT
        # ---------------------------

        if full_beam_id in beam_wall.index:

            relations = beam_wall.loc[full_beam_id]

            wall_count = int(relations.sum())

            if wall_count > 0:

                # distribute supports along beam
                positions = np.linspace(0, n_segments-1, wall_count)

                for p in positions:
                    vector[int(p)] = "W"

        # ---------------------------
        # COLUMN SUPPORT (FIXED: Use actual positions)
        # ---------------------------

        if full_beam_id in beam_column.index:

            relations = beam_column.loc[full_beam_id]
            
            # Get column IDs that connect to this beam
            connected_columns = relations[relations == 1].index.tolist()
            
            if len(connected_columns) > 0:
                
                # Get beam start and end coordinates
                beam_start = np.array([beam["Start X"], beam["Start Y"], beam["Start Z"]])
                beam_end = np.array([beam["End X"], beam["End Y"], beam["End Z"]])
                
                for col_full_id in connected_columns:
                    # Extract column element ID from full ID (format: building_elementID_C)
                    col_element_id = int(col_full_id.split("_")[1])
                    
                    # Find column in feature matrix
                    col_row = feature[feature["Element ID"] == col_element_id]
                    
                    if not col_row.empty:
                        col_row = col_row.iloc[0]
                        
                        # Use column start position (base of column)
                        col_xyz = np.array([col_row["Start X"], col_row["Start Y"], col_row["Start Z"]])
                        
                        # Project column onto beam to get relative position
                        relative_pos = project_point_on_beam(col_xyz, beam_start, beam_end)
                        
                        # Convert to segment index
                        segment_idx = int(relative_pos * (n_segments - 1))
                        segment_idx = min(segment_idx, n_segments - 1)  # Ensure within bounds
                        
                        # Mark segment
                        vector[segment_idx] = "C"

        # ---------------------------
        # BEAM INTERSECTION
        # ---------------------------

        if full_beam_id in beam_beam.index:

            relations = beam_beam.loc[full_beam_id]

            beam_count = int(relations.sum())

            if beam_count > 0:

                positions = np.linspace(0, n_segments-1, beam_count)

                for p in positions:
                    # Don't overwrite columns with beam markers
                    if vector[int(p)] != "C":
                        vector[int(p)] = "B"

        rows.append({
            "building_id": bid,
            "beam_id": full_beam_id,
            "beam_length": length,
            "segments": n_segments,
            "segment_vector": "[" + ",".join(vector) + "]"
        })


    result = pd.DataFrame(rows)

    result.to_csv(f"{OUTPUT_FOLDER}/{bid}_BeamSegmentVector.csv",index=False)

    print("Saved:", bid)

print("Done.")