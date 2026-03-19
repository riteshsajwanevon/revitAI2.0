#!/usr/bin/env python3
"""
Create Training Dataset from Beam Signal CSV files

This script:
1. Reads beam signal CSV files
2. Extracts wall, beam, and column signals
3. Builds ML training arrays
4. Saves dataset as .npz
5. Also creates a smoothed target dataset
"""

import os
import glob
import ast
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

INPUT_FOLDER = "Dataset/validation/ml_signal"

OUTPUT_DATASET = "Dataset/validation/beam_training_dataset.npz"
OUTPUT_DATASET_SMOOTHED = "Dataset/validation/beam_training_dataset_smoothed.npz"


# -----------------------------------------------------------------------------
# START PROCESS
# -----------------------------------------------------------------------------

print("=" * 80)
print("CREATING TRAINING DATASET")
print("=" * 80)

# Find all CSV files
csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*_BeamSignal128.csv"))

if len(csv_files) == 0:
    print("No CSV files found!")
    exit()


# -----------------------------------------------------------------------------
# STORAGE LISTS
# -----------------------------------------------------------------------------

X_samples = []
Y_samples = []
building_ids = []


# -----------------------------------------------------------------------------
# PROCESS EACH FILE
# -----------------------------------------------------------------------------

for file_path in csv_files:

    file_name = os.path.basename(file_path)

    # Extract building ID
    building_id = file_name.split("_")[0]

    print(f"Processing building: {building_id}")

    # Read CSV
    df = pd.read_csv(file_path)

    # Process each row
    for _, row in df.iterrows():

        # Convert string lists to numpy arrays
        wall_signal = np.array(ast.literal_eval(row["wall_signal"]))
        beam_signal = np.array(ast.literal_eval(row["beam_signal"]))
        column_signal = np.array(ast.literal_eval(row["column_signal"]))

        # Create input (2 channels)
        X = np.stack([wall_signal, beam_signal])

        # Target output
        Y = column_signal

        # Store sample
        X_samples.append(X)
        Y_samples.append(Y)
        building_ids.append(building_id)


# -----------------------------------------------------------------------------
# CONVERT TO NUMPY ARRAYS
# -----------------------------------------------------------------------------

X = np.array(X_samples)
Y = np.array(Y_samples)
building_ids = np.array(building_ids)


# -----------------------------------------------------------------------------
# DATASET STATISTICS
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("DATASET STATISTICS")
print("=" * 80)

print(f"Input shape (X): {X.shape}")
print(f"Target shape (Y): {Y.shape}")
print(f"Total samples: {len(X)}")
print(f"Unique buildings: {len(set(building_ids))}")


# -----------------------------------------------------------------------------
# SAVE DATASET
# -----------------------------------------------------------------------------

np.savez(
    OUTPUT_DATASET,
    X=X,
    Y=Y,
    building_ids=building_ids
)

print(f"\nDataset saved to: {OUTPUT_DATASET}")


# -----------------------------------------------------------------------------
# CREATE SMOOTHED TARGET DATASET
# -----------------------------------------------------------------------------

print("\nCreating smoothed target dataset...")

Y_smoothed = np.array([
    gaussian_filter1d(signal, sigma=3)
    for signal in Y
])


np.savez(
    OUTPUT_DATASET_SMOOTHED,
    X=X,
    Y=Y_smoothed,
    building_ids=building_ids
)

print(f"Smoothed dataset saved to: {OUTPUT_DATASET_SMOOTHED}")


# -----------------------------------------------------------------------------
# FINISHED
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("DATASET CREATION COMPLETE")
print("=" * 80)



# #!/usr/bin/env python3
# """
# Create Training Dataset WITHOUT Validation Buildings
# This ensures no data leakage
# """

# import pandas as pd
# import numpy as np
# import glob
# import os
# import ast

# INPUT_FOLDER = "Dataset/train/ml_signal"
# OUTPUT_FILE = "Dataset/train/beam_training_dataset_no_leakage.npz"
# OUTPUT_FILE_SMOOTHED = "Dataset/train/beam_training_dataset_smoothed_no_leakage.npz"

# # Validation buildings to EXCLUDE from training
# VALIDATION_BUILDINGS = [
#     "20250034", "20250041", "20250050", "20250053", "20250060",
#     "20250079", "20250082", "20250084", "20250665", "20260049"
# ]

# print("="*80)
# print("CREATING TRAINING DATASET WITHOUT VALIDATION BUILDINGS")
# print("="*80)

# files = glob.glob(os.path.join(INPUT_FOLDER, "*_BeamSignal128.csv"))

# X_list = []
# Y_list = []
# building_ids = []
# excluded_count = 0
# included_count = 0

# for f in files:
#     # Extract building ID from filename
#     building_id = os.path.basename(f).split('_')[0]
    
    
    
#     included_count += 1
    
#     df = pd.read_csv(f)
    
#     for _, row in df.iterrows():
#         wall = np.array(ast.literal_eval(row["wall_signal"]))
#         beam = np.array(ast.literal_eval(row["beam_signal"]))
#         column = np.array(ast.literal_eval(row["column_signal"]))
        
#         X = np.stack([wall, beam])
#         Y = column
        
#         X_list.append(X)
#         Y_list.append(Y)
#         building_ids.append(building_id)

# X = np.array(X_list)
# Y = np.array(Y_list)
# building_ids = np.array(building_ids)

# print(f"\n{'='*80}")
# print("DATASET STATISTICS")
# print(f"{'='*80}")
# print(f"\nBuildings excluded (validation): {excluded_count}")
# print(f"Buildings included (training): {included_count}")
# print(f"\nDataset shape:")
# print(f"  X: {X.shape}")
# print(f"  Y: {Y.shape}")
# print(f"  Total beams: {len(X)}")
# print(f"  Unique buildings: {len(set(building_ids))}")

# # Save dataset
# np.savez(OUTPUT_FILE, X=X, Y=Y, building_ids=building_ids)
# print(f"\n✅ Saved dataset: {OUTPUT_FILE}")

# # Also create smoothed version
# from scipy.ndimage import gaussian_filter1d

# Y_smoothed = np.array([gaussian_filter1d(y, sigma=3) for y in Y])

# np.savez(OUTPUT_FILE_SMOOTHED, X=X, Y=Y_smoothed, building_ids=building_ids)
# print(f"✅ Saved smoothed dataset: {OUTPUT_FILE_SMOOTHED}")

# print(f"\n{'='*80}")
# print("VERIFICATION")
# print(f"{'='*80}")

# # Verify no validation buildings in training data
# training_buildings = set(building_ids)
# validation_set = set(VALIDATION_BUILDINGS)
# overlap = training_buildings.intersection(validation_set)

# if len(overlap) == 0:
#     print(f"\n✅ NO DATA LEAKAGE CONFIRMED")
#     print(f"   Training buildings: {len(training_buildings)}")
#     print(f"   Validation buildings: {len(validation_set)}")
#     print(f"   Overlap: 0")
# else:
#     print(f"\n❌ ERROR: Still have overlap!")
#     print(f"   Overlap: {overlap}")

# print(f"\n{'='*80}")
