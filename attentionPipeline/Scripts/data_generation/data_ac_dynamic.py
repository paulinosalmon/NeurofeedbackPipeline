import numpy as np
import time

# Constants
TRIALS = 1200
SAMPLES = 550
CHANNELS = 32
SAMPLING_RATE = 500  # 500 Hz

def generate_synthetic_eeg_data(trials, samples, channels):
    """
    Generate a synthetic EEG dataset with dimensions [trials, samples, channels].
    """
    # Generate random EEG data
    synthetic_eeg_data = np.random.randn(trials, samples, channels) * 1e-6  # Scale to typical EEG microvolt range

    # Introduce variability (e.g., some random spikes in the data)
    for i in range(trials):
        if np.random.rand() < 0.05:  # 5% chance of a random spike
            spike_time = np.random.randint(0, samples)
            spike_channel = np.random.randint(0, channels)
            synthetic_eeg_data[i, spike_time, spike_channel] += np.random.uniform(-50, 50) * 1e-6  # Add a random spike within the typical EEG microvolt range

    return synthetic_eeg_data

def stream_synthetic_eeg_data():
    while True:
        # Generate synthetic data
        data = generate_synthetic_eeg_data(TRIALS, SAMPLES, CHANNELS)
        
        # You can add code here to stream this data to a destination, such as LSL
        # For now, we just print the shape of the data to confirm it's correct
        print(f"Generated synthetic EEG data with shape: {data.shape}")
        print(f"Trial: {data[0]}")
        print(f"Samples: {data[1]}")
        print(f"Channels: {data[2]}")

        # Log output (this is where you would have your logging or streaming)
        print("Streaming synthetic EEG data...")

        # Wait for a bit before generating the next batch to simulate continuous streaming
        time.sleep(1/SAMPLING_RATE)

if __name__ == "__main__":
    stream_synthetic_eeg_data()