import pandas as pd
import numpy as np
import glob
import os
import ast

INPUT_FOLDER = r"Dataset/validation/MLDataSet"
OUTPUT_FOLDER = r"Dataset/validation/ml_signal"

TARGET_LENGTH = 128

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

files = glob.glob(os.path.join(INPUT_FOLDER,"*_BeamChannels.csv"))

for f in files:

    df = pd.read_csv(f)

    rows = []

    for _, row in df.iterrows():

        building_id = row["building_id"]
        beam_id = row["beam_id"]

        wall = np.array(ast.literal_eval(row["wall_channel"]))
        beam = np.array(ast.literal_eval(row["beam_channel"]))
        column = np.array(ast.literal_eval(row["column_channel"]))

        n = len(wall)

        x_old = np.linspace(0,1,n)
        x_new = np.linspace(0,1,TARGET_LENGTH)

        wall_resampled = np.interp(x_new,x_old,wall)
        beam_resampled = np.interp(x_new,x_old,beam)
        column_resampled = np.interp(x_new,x_old,column)

        rows.append({
            "building_id": building_id,
            "beam_id": beam_id,
            "wall_signal": wall_resampled.tolist(),
            "beam_signal": beam_resampled.tolist(),
            "column_signal": column_resampled.tolist()
        })

    result = pd.DataFrame(rows)

    bid = os.path.basename(f).split("_")[0]

    result.to_csv(f"{OUTPUT_FOLDER}/{bid}_BeamSignal128.csv",index=False)

    print("Saved:", bid)