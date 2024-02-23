import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock
import time
from settings import samplingRate, channelNames

# Simulate EEG data for testing
def simulate_eeg_data():
    eeg_data = np.random.randn(len(channelNames)) * 0.0001
    # Introduce occasional spikes
    if np.random.rand() < 0.1:  # 10% chance of a spike
        spike_channel = np.random.choice(len(channelNames))
        eeg_data[spike_channel] += np.random.choice([-5, 5]) * 0.001
    timestamp = time.time()
    label = np.random.choice([0, 1])  # Randomly choose between two states (0 for faces, 1 for scenes)
    return eeg_data, label, timestamp

# Set up LSL stream for EEG data
def create_lsl_stream():
    # Define LSL stream parameters with an additional channel for the label
    info = StreamInfo(name='EmotivEPOCX', type='EEG', channel_count=len(channelNames) + 1,
                      channel_format='float32', source_id='epocx1234', nominal_srate=samplingRate)
    # Add channel names to LSL info
    channels = info.desc().append_child("channels")
    for ch in channelNames:
        channels.append_child("channel").append_child_value("label", ch)
    # Add a channel for the label
    channels.append_child("channel").append_child_value("label", "label")
    # Create LSL outlet
    outlet = StreamOutlet(info)
    return outlet

def stream_eeg_data(outlet):
    while True:
        simulated_data, label, timestamp = simulate_eeg_data()
        print(f"[Data Acquisition] Streaming data: {simulated_data} with label: {label} at timestamp: {timestamp}")
        outlet.push_sample(np.append(simulated_data, label), timestamp)  # Append label to data
        # time.sleep(0.05)
        time.sleep(1 / samplingRate)

def run_data_acquisition():
    outlet = create_lsl_stream()
    print("[Data Acquisition] Streaming simulated EEG data...")
    stream_eeg_data(outlet)

if __name__ == '__main__':
    run_data_acquisition()