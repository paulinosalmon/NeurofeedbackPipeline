import numpy as np

# Paths to the EEG datasets
alzheimers_data_path = './output/EEG_alzheimers.npy'
dementia_data_path = './output/EEG_dementia.npy'
eeg_sample_data_path = './output/EEG_epochs_sample.npy'

# Load the EEG datasets
alzheimers_data = np.load(alzheimers_data_path)
dementia_data = np.load(dementia_data_path)
eeg_sample_data = np.load(eeg_sample_data_path)

# Inspect shapes
print("EEG Alzheimer's vs. Healthy data shape:", alzheimers_data.shape)
print("EEG Dementia vs. Healthy data shape:", dementia_data.shape)
print("EEG sample data shape:", eeg_sample_data.shape)

# Display a few samples to verify structure
print("\nFirst trial of Alzheimer's vs. Healthy EEG data (first 10 samples):")
print(alzheimers_data[0, :10, :])

print("\nFirst trial of Dementia vs. Healthy EEG data (first 10 samples):")
print(dementia_data[0, :10, :])

print("\nFirst trial of sample EEG data (first 10 samples):")
print(eeg_sample_data[0, :10, :])

# Verify the number of trials, samples, and channels
alz_trials, alz_samples, alz_channels = alzheimers_data.shape
dem_trials, dem_samples, dem_channels = dementia_data.shape
sample_trials, sample_samples, sample_channels = eeg_sample_data.shape

print(f"\nAlzheimer's vs. Healthy - Trials: {alz_trials}, Samples per trial: {alz_samples}, Channels: {alz_channels}")
print(f"Dementia vs. Healthy - Trials: {dem_trials}, Samples per trial: {dem_samples}, Channels: {dem_channels}")
print(f"Sample EEG Data - Trials: {sample_trials}, Samples per trial: {sample_samples}, Channels: {sample_channels}")
