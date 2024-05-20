import csv
import json
import mne
import numpy as np

# Function to load the TSV file without using pandas
def load_channels_info(file_path):
    channels_info = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            channels_info.append(row)
    return channels_info

# Function to load the JSON file
def load_eeg_metadata(file_path):
    with open(file_path, 'r') as f:
        eeg_metadata = json.load(f)
    return eeg_metadata

# Function to load the SET file using MNE
def load_eeg_data(file_path):
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    return raw

# Function to load labels from a CSV file (modify as needed for your dataset)
def load_labels(file_path):
    labels = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            labels.append(row[0])
    return labels

# Paths to the files
tsv_file_path = './sub-001_task-eyesclosed_channels.tsv'
json_file_path = './sub-001_task-eyesclosed_eeg.json'
set_file_path = './sub-001_task-eyesclosed_eeg.set'
labels_file_path = './labels.csv'  # Path to the labels file (adjust as necessary)

# Load the data
channels_info = load_channels_info(tsv_file_path)
print("Channels Info:")
for channel in channels_info:
    print(channel)

eeg_metadata = load_eeg_metadata(json_file_path)
print("\nEEG Metadata:")
print(eeg_metadata)

raw_eeg_data = load_eeg_data(set_file_path)
print("\nRaw EEG Data Info:")
print(raw_eeg_data.info)  # Correctly access the info attribute

# Load the labels
labels = load_labels(labels_file_path)
print("\nLabels:")
print(labels)

# Check for specific labels
has_normal = any('normal' in label.lower() for label in labels)
has_alzheimers = any('alzheimer' in label.lower() for label in labels)
has_dementia = any('dementia' in label.lower() for label in labels)

print("\nPresence of specific conditions in labels:")
print(f"Normal: {has_normal}")
print(f"Alzheimer's: {has_alzheimers}")
print(f"Dementia: {has_dementia}")

# Extract all data (in Volts)
data, times = raw_eeg_data[:, :]  # Extract data for all channels and all samples

# Calculate statistics
global_min_value = data.min()
global_max_value = data.max()
mean_value = data.mean()
std_value = data.std()
percentiles = np.percentile(data, [25, 50, 75])

# Print statistics
print("\nGlobal statistics for values across all channels:")
print(f"Global Minimum value: {global_min_value}")
print(f"Global Maximum value: {global_max_value}")
print(f"Mean value: {mean_value}")
print(f"Standard deviation: {std_value}")
print(f"25th percentile: {percentiles[0]}")
print(f"50th percentile (median): {percentiles[1]}")
print(f"75th percentile: {percentiles[2]}")
