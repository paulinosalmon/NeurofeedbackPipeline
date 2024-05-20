import numpy as np

# Load the data from .npy files
eeg_data_path = './EEG_epochs_sample.npy'
labels_data_path = './y_categories_sample.npy'

# Load the EEG data and labels
eeg_data = np.load(eeg_data_path, allow_pickle=True)  # Shape: (1200, 550, 32)
labels_data = np.load(labels_data_path)  # Shape: (1200,)

# Display shape of the loaded data
print("EEG data shape:", eeg_data.shape)
print("Labels data shape:", labels_data.shape)

# Display the contents of the labels data
print("\nLabels data (first 20 entries):", labels_data[:20])

# Check if the EEG data contains channel names or indices
if isinstance(eeg_data, np.ndarray) and eeg_data.dtype.names:
    channel_info = eeg_data.dtype.names
    print("\nChannel information (names):", channel_info)
else:
    print("\nChannel information not available in the EEG data array. Assuming numerical indices are used.")

# Optionally, inspect the first few trials to see data structure
print("\nFirst trial data (first 10 samples):")
print(eeg_data[0, :10, :])
