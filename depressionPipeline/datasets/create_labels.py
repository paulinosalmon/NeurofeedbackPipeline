import numpy as np
import os

# Create the output directory if it doesn't exist
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# Labels for Alzheimer's vs. Healthy
alzheimers_labels = np.concatenate([
    np.ones(36, dtype=int),  # Alzheimer's: label 1
    np.zeros(18, dtype=int)  # Healthy: label 0
])

# Labels for Dementia vs. Healthy
dementia_labels = np.concatenate([
    np.ones(23, dtype=int),  # Dementia: label 1
    np.zeros(11, dtype=int)  # Healthy: label 0
])

# Save the label arrays to .npy files
alzheimers_file_path = os.path.join(output_dir, 'y_alzheimers.npy')
dementia_file_path = os.path.join(output_dir, 'y_dementia.npy')

np.save(alzheimers_file_path, alzheimers_labels)
np.save(dementia_file_path, dementia_labels)

# Load and inspect the created label files to ensure correctness
loaded_alzheimers_labels = np.load(alzheimers_file_path)
loaded_dementia_labels = np.load(dementia_file_path)

# Print the shapes and the first few entries of the loaded labels for verification
print("Alzheimer's vs. Healthy Labels:")
print("Shape:", loaded_alzheimers_labels.shape)
print("First 10 Labels:", loaded_alzheimers_labels[:10])
print("Last 10 Labels:", loaded_alzheimers_labels[-10:])

print("\nDementia vs. Healthy Labels:")
print("Shape:", loaded_dementia_labels.shape)
print("First 10 Labels:", loaded_dementia_labels[:10])
print("Last 10 Labels:", loaded_dementia_labels[-10:])
