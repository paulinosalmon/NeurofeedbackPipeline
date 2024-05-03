import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation

# Define the channel names
channel_names = ['P7', 'P4', 'Cz', 'Pz', 'P3', 'P8', 'O1', 'O2', 'T8', 'F8', 'C4', 'F4', 'Fp2', 'Fz', 'C3', 'F3', 'Fp1', 'T7', 'F7', 'Oz', 'PO3', 'AF3', 'FC5', 'FC1', 'CP5', 'CP1', 'CP2', 'CP6', 'AF4', 'FC2', 'FC6', 'PO4']

# Set the parameters for the signal generation
sampling_rate = 1000  # Hz
duration = 1  # seconds
time = np.arange(0, duration, 1/sampling_rate)
num_channels = len(channel_names)

# Select 5 random channels
selected_channels = random.sample(channel_names, 5)

# Initialize the figure with increased height to accommodate the channels
fig, ax = plt.subplots(figsize=(12, 12))  # Increased the height here
lines = []
for i in range(len(selected_channels)):
    line, = ax.plot(time, np.zeros_like(time), label=selected_channels[i])
    lines.append(line)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Channel')
ax.set_yticks(range(0, 400 * len(selected_channels), 400))
ax.set_yticklabels(selected_channels)
ax.set_title('Synthetic EEG Signals - 5 Random Channels')

# Adjust the spacing of the subplots
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.4)

# Set the y-axis limits with some padding
ax.set_ylim(-300, 400 * len(selected_channels) - 100)

# ax.legend(loc='upper right')

# Function to update the plot
def update(frame):
    for i, channel in enumerate(selected_channels):
        eeg_signal = np.random.uniform(-200, 200, size=len(time))
        lines[i].set_ydata(eeg_signal + 400 * i)
    return lines

# Create the animation
ani = FuncAnimation(fig, update, frames=np.arange(100), blit=True, interval=1000)

# Show the plot
plt.show()
