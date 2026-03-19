import pandas as pd
import ast
import os
import glob

INPUT_FOLDER = "Dataset/validation/BeamSegmentation"   # updated: uses connection-based segmentation
OUTPUT_FOLDER = "Dataset/validation/MLDataSet"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

files = glob.glob(os.path.join(INPUT_FOLDER,"*_BeamSegmentVector.csv"))

for f in files:
    
    try:
        df = pd.read_csv(f)
        
        if df.empty:
            print(f"Skipping empty file: {os.path.basename(f)}")
            continue

        rows = []

        for _, row in df.iterrows():

            beam_id = row["beam_id"]
            building_id = row["building_id"]

            vector_str = row["segment_vector"]

            # remove brackets
            vector_str = vector_str.replace("[","").replace("]","")

            vector = vector_str.split(",")

            wall = []
            beam = []
            column = []

            for v in vector:

                v = v.strip()

                if v == "W":
                    wall.append(1)
                    beam.append(0)
                    column.append(0)

                elif v == "B":
                    wall.append(0)
                    beam.append(1)
                    column.append(0)

                elif v == "C":
                    wall.append(0)
                    beam.append(0)
                    column.append(1)

                else:
                    wall.append(0)
                    beam.append(0)
                    column.append(0)

            rows.append({
                "building_id": building_id,
                "beam_id": beam_id,
                "wall_channel": wall,
                "beam_channel": beam,
                "column_channel": column
            })

        result = pd.DataFrame(rows)

        building_id = os.path.basename(f).split("_")[0]

        result.to_csv(f"{OUTPUT_FOLDER}/{building_id}_BeamChannels.csv",index=False)

        print("Saved:", building_id)
        
    except Exception as e:
        print(f"Error processing {os.path.basename(f)}: {e}")
        continue

print("Done.")
