import numpy as np


# ---------------------------------
# Parameters
# ---------------------------------

INPUT_FILE = r"Dataset/validation/beam_training_dataset.npz"
OUTPUT_FILE = r"Dataset/validation/beam_training_dataset_smoothed.npz"

SIGMA = 3     # gaussian width


# ---------------------------------
# Gaussian smoothing function
# ---------------------------------

def gaussian_smooth(signal, sigma=3):

    signal = np.array(signal)

    length = len(signal)

    smoothed = np.zeros(length)

    column_positions = np.where(signal > 0)[0]

    for p in column_positions:

        for i in range(length):

            smoothed[i] += np.exp(-(i - p) ** 2 / (2 * sigma ** 2))

    if smoothed.max() > 0:
        smoothed = smoothed / smoothed.max()

    return smoothed


# ---------------------------------
# Load dataset
# ---------------------------------

data = np.load(INPUT_FILE)

X = data["X"]
Y = data["Y"]

print("Loaded dataset")
print("X shape:", X.shape)
print("Y shape:", Y.shape)


# ---------------------------------
# Smooth column signals
# ---------------------------------

Y_smooth = []

for signal in Y:

    smoothed = gaussian_smooth(signal, sigma=SIGMA)

    Y_smooth.append(smoothed)

Y_smooth = np.array(Y_smooth)


print("Smoothed Y shape:", Y_smooth.shape)


# ---------------------------------
# Save new dataset
# ---------------------------------

np.savez(
    OUTPUT_FILE,
    X=X,
    Y=Y_smooth
)

print("Saved new dataset:", OUTPUT_FILE)