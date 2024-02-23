import numpy as np
from pylsl import StreamInlet, resolve_stream
import mne
from scipy.signal import detrend
from settings import (samplingRate, channelNames, 
                      baselineTime, epochTime, 
                      rejectChannels, channelNamesExcluded)
# import keyboard

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

def process_continuous_eeg(gui_queue, data_transfer_queue):
    epoch_duration = 0.9  # 900 ms
    buffer_size = int(epoch_duration * samplingRate)  # Number of samples per epoch
    eeg_buffer = np.empty((len(channelNames), 0))
    label_buffer = []

    while True:
        sample, timestamp = inlet.pull_sample()
        eeg_sample = np.array(sample[:-1]).reshape(len(channelNames), -1)  # Exclude the label
        label = sample[-1]  # Last element is the label

        eeg_buffer = np.hstack((eeg_buffer, eeg_sample))
        label_buffer.append(label)
        message = f"[Data Preprocessing] Current buffer size: {eeg_buffer.shape[1]}"
        print(message) 
        gui_queue.put(message)  

        if eeg_buffer.shape[1] >= buffer_size:
            processed_data = preprocess_eeg(eeg_buffer[:, :buffer_size], channelNames)
            eeg_buffer = eeg_buffer[:, buffer_size:]  # Remove processed data from buffer
            current_labels = label_buffer[:buffer_size]
            label_buffer = label_buffer[buffer_size:]  # Remove processed labels from buffer

            print(f"[Data Preprocessing] Processed an EEG epoch. Data shape: {processed_data.shape}")
            gui_queue.put(f"[Data Preprocessing] Processed an EEG epoch. Data shape: {processed_data.shape}")
            print(f"[Data Preprocessing] Labels for epoch: {current_labels}")
            gui_queue.put(f"[Data Preprocessing] Labels for epoch: {current_labels}")

def run_data_preprocessing(gui_queue, data_transfer_queue):
    print("[Data Preprocessing] Starting preprocessing...")
    gui_queue.put("[Data Preprocessing] Starting preprocessing...")
    process_continuous_eeg(gui_queue, data_transfer_queue)
