import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pylsl import StreamInfo, StreamOutlet
from settings import samplingRate, channelNames
import threading
import time

# Set the parameters for the signal generation
start_time = -0.1  # Start at -0.1 seconds
end_time = 1
time_array = np.arange(start_time, end_time, 1 / samplingRate)
num_channels = len(channelNames)

# Initialize the figure
fig, ax = plt.subplots(figsize=(12, 24))
lines = []
for i, channel in enumerate(channelNames):
    line, = ax.plot(time_array, np.zeros_like(time_array), label=channel)
    lines.append(line)

# Graph Settings
ax.set_xlim(start_time, end_time)
ax.set_xticks(np.arange(start_time, end_time + 0.1, 0.10))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Channel')
ax.set_yticks(range(0, 400 * num_channels, 400))
ax.set_yticklabels(channelNames)
ax.set_title('Synthetic EEG Signals - All Channels')
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.4)
ax.set_ylim(-300, 400 * num_channels - 100)

# Add vertical lines and color borders
ax.axvline(x=0.0, color='black', linestyle='--')  # Add a vertical line at x = 0.0
ax.axvspan(start_time, 0.0, facecolor='red', alpha=0.3)  # Color the region between -0.1 and 0.0
ax.axvspan(0.0, 0.8, facecolor='green', alpha=0.3)  # Color the region between 0.0 and 0.8

# Create an LSL stream
info = StreamInfo(name='SyntheticEEG', type='EEG', channel_count=num_channels + 1, nominal_srate=samplingRate, channel_format='float32', source_id='random1234')
outlet = StreamOutlet(info)

# Define the duration of each epoch and calculate the number of samples per epoch
epoch_duration = 0.9  # 900 ms
samples_per_epoch = int(samplingRate * epoch_duration)

# Function to generate and send data at a specific rate
def transmit_data():
    label = 0  # Start with label 0 for the first epoch
    while True:
        # Generate the entire epoch at once
        eeg_epoch = np.zeros((samples_per_epoch, len(channelNames)))

        for i in range(len(channelNames)):
            # Add a sinusoidal component to simulate oscillatory activity
            freq = np.random.choice([8, 10, 12])  # Randomly choose a frequency for the oscillation
            amplitude = np.random.uniform(50, 100)  # Randomly choose an amplitude
            phase = np.random.uniform(0, 2 * np.pi)  # Randomly choose a phase
            sinusoid = amplitude * np.sin(2 * np.pi * freq * np.linspace(start_time, end_time, samples_per_epoch) + phase)

            # Add random noise
            noise = np.random.normal(0, 20, samples_per_epoch)

            # Combine the sinusoidal component and noise
            eeg_epoch[:, i] = sinusoid + noise

        # Normalize the data to be within the range [-200, 200]
        eeg_epoch = 200 * (eeg_epoch - np.min(eeg_epoch)) / (np.max(eeg_epoch) - np.min(eeg_epoch)) - 200

        # Append the label as a new column to the epoch
        eeg_epoch_with_label = np.hstack((eeg_epoch, np.full((samples_per_epoch, 1), label)))

        # Send the entire epoch with the label
        for sample in eeg_epoch_with_label:
            outlet.push_sample(sample)

        label = 1 - label  # Toggle the label after each epoch
        time.sleep(epoch_duration)  # Wait before starting the next epoch


# Start the data transmission in a separate thread
thread = threading.Thread(target=transmit_data)
thread.daemon = True
thread.start()

def update(frame):
    for i, channel in enumerate(channelNames):
        eeg_signal = np.random.uniform(-200, 200, size=len(time_array))
        lines[i].set_ydata(eeg_signal + 400 * i)
    return lines

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(100), blit=True, interval=2000)

# Show the plot
plt.show()
