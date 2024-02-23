import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock
import time
from settings import samplingRate, channelNames

def simulate_eeg_data():
    eeg_data = np.random.randn(len(channelNames)) * 0.0001
    label = np.random.choice([0, 1])

    # Introduce patterns based on the label
    if label == 1:
        # Pattern more common when label is 1
        eeg_data += (np.random.rand(len(channelNames)) - 0.5) * 0.001
    else:
        # Pattern more common when label is 0
        eeg_data -= (np.random.rand(len(channelNames)) - 0.5) * 0.001

    timestamp = time.time()
    return eeg_data, label, timestamp

def create_lsl_stream():
    info = StreamInfo(name='Enobio32', type='EEG', channel_count=len(channelNames) + 1,
                      channel_format='float32', source_id='enobio32_1234', nominal_srate=samplingRate)
    channels = info.desc().append_child("channels")
    for ch in channelNames:
        channels.append_child("channel").append_child_value("label", ch)
    channels.append_child("channel").append_child_value("label", "label")
    outlet = StreamOutlet(info)
    return outlet

def stream_eeg_data(outlet):
    while True:
        simulated_data, label, timestamp = simulate_eeg_data()
        print(f"[Data Acquisition] Streaming data: {simulated_data} with label: {label} at timestamp: {timestamp}")
        outlet.push_sample(np.append(simulated_data, label), timestamp) 
        time.sleep(1 / samplingRate)

def run_data_acquisition():
    outlet = create_lsl_stream()
    print("[Data Acquisition] Streaming simulated EEG data...")
    stream_eeg_data(outlet)

if __name__ == '__main__':
    run_data_acquisition()
