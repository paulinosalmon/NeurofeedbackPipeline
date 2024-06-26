import numpy as np
from pylsl import StreamInlet, resolve_stream
import mne
from scipy.signal import detrend
from settings import (samplingRate, channelNames, 
                      rejectChannels, channelNamesExcluded,
                      training_trials, training_state)
import time

# Resolve the EEG stream
print("Looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])

def preprocess_eeg(data, channel_names):

    # Exclude specified channels if rejection is enabled
    if rejectChannels:
        exclude_indices = [channel_names.index(ch) for ch in channelNamesExcluded if ch in channel_names]
        data = np.delete(data, exclude_indices, axis=0)
        channel_names = [ch for ch in channel_names if ch not in channelNamesExcluded]

    # Linear detrending
    data_detrended = detrend(data, axis=1)

    # Low-pass filtering
    data_filtered = mne.filter.filter_data(data_detrended, sfreq=samplingRate, l_freq=None, h_freq=40, method='fir', phase='zero-double')

    # Downsampling
    data_downsampled = mne.filter.resample(data_filtered, down=samplingRate // 100)

    # Baseline correction
    baseline = np.mean(data_downsampled[:, :int(0.1 * 100)], axis=1)
    data_baseline_corrected = data_downsampled - baseline[:, None]

    # Z-scoring
    data_zscored = (data_baseline_corrected - np.mean(data_baseline_corrected)) / np.std(data_baseline_corrected)

    return data_zscored

def process_continuous_eeg(gui_queue, data_transfer_queue, preprocessing_done):
    epoch_duration = 0.9  # 900 ms
    buffer_size = int(epoch_duration * samplingRate)  # Number of samples per epoch
    eeg_buffer = np.empty((len(channelNames), 0))
    label_buffer = []  # Buffer to store labels
    trial_counter = 0  # Counter for the number of trials processed

    if training_state == 0:
        gui_queue.put(f"[Data Preprocessing] Dataset mode active.")
        dataset_X = np.load('../data/train/EEG_epochs_sample.npy')
        dataset_y = np.load('../data/train/y_categories_sample.npy')

        # Swap the 'samples' and 'channels' dimensions in dataset_X
        # New shape will be [1200 trials, 32 channels, 550 samples]
        dataset_X = np.transpose(dataset_X, (0, 2, 1))        

        # Create a permutation based on the number of entries
        permutation = np.random.permutation(dataset_X.shape[0])

        # Shuffle both dataset_X and dataset_y according to the same permutation
        dataset_X = dataset_X[permutation]
        dataset_y = dataset_y[permutation]

        # Iterate through each trial and corresponding label
        dataset_counter = 1
        for trial_data, current_label in zip(dataset_X, dataset_y):
            # Package the trial data and label together
            gui_queue.put(f"[Trial {dataset_counter}/{len(dataset_X)}]")
            processed_data = preprocess_eeg(trial_data, channelNames)

            gui_queue.put(f"[Reading Dataset...] Data shape: {processed_data.shape}")
            gui_queue.put(f"[Reading Dataset...] Label: {current_label}")

            data_transfer_queue.put((processed_data, current_label))  # Send both data and label
            time.sleep(0.5)
            dataset_counter += 1
            preprocessing_done.set()
            
        gui_queue.put(f"[Data Preprocessing] Full dataset sent.")

    while True:
        sample, timestamp = inlet.pull_sample()
        eeg_sample = np.array(sample[:-1]).reshape(len(channelNames), -1)  # Exclude the last element (label)
        label = sample[-1]  # Extract the label

        eeg_buffer = np.hstack((eeg_buffer, eeg_sample))
        label_buffer.append(label)  # Append the label to the label buffer

        # Checks until 450 sample points (900ms epoch, 500Hz Sampling Rate) have been recorded
        if eeg_buffer.shape[1] >= buffer_size:
            gui_queue.put(f"[Data Preprocessing] Full 900ms epoch recorded. Proceeding with preprocessing.")
            processed_data = preprocess_eeg(eeg_buffer[:, :buffer_size], channelNames)
            eeg_buffer = eeg_buffer[:, buffer_size:]  # Remove processed data from buffer

            # Use the first label of the epoch
            current_label = label_buffer[0]
            label_buffer = label_buffer[buffer_size:]  # Remove processed labels from buffer

            trial_counter += 1  # Increment the trial counter
            gui_queue.put(f"======== [Trial {trial_counter}] ========\nProcessed EEG Epoch.\nData shape: {processed_data.shape}\nLabel: {current_label}")

            data_transfer_queue.put((processed_data, current_label))  # Send both data and label
            gui_queue.put(f"[Data Preprocessing] Preprocessing done. Data pushed to queue for artifact rejection.")
            
            print("[Data Preprocessing] Preprocessing done. Signaling event.")
            preprocessing_done.set()
            print("[Data Preprocessing] Event signaled.")

            if trial_counter == training_trials:
                gui_queue.put(f"[Data Preprocessing] {training_trials} reached. Ending data preprocessing.")
                break

def run_data_preprocessing(gui_queue, data_transfer_queue, preprocessing_done):
    print("[Data Preprocessing] Starting preprocessing...")
    gui_queue.put("[Data Preprocessing] Starting preprocessing...")
    process_continuous_eeg(gui_queue, data_transfer_queue, preprocessing_done)
