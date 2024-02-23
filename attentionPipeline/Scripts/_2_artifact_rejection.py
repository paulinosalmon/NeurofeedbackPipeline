import numpy as np
from numpy.linalg import svd
from pylsl import StreamInlet, resolve_stream
from settings import channelNames, samplingRate
from datetime import datetime

def generate_synthetic_training_data(num_channels, num_samples, random_state=42):
    np.random.seed(random_state)
    return np.random.randn(num_channels, num_samples) * 1e-6

def perform_svd_and_select_components(X_training, variance_threshold=0.1):
    U, S, Vt = svd(X_training, full_matrices=False)
    variance_explained = (S**2) / np.sum(S**2)
    significant_components = U[:, variance_explained >= variance_threshold]
    return significant_components

def denoise_data(X_noisy_test, significant_components, queue_gui):
    if significant_components.size == 0:
        message = "[{datetime.now()}] [Artifact Rejection] No significant components found. Skipping denoising."
        print(message)
        queue_gui.put(message)
        return X_noisy_test
    return X_noisy_test - X_noisy_test.dot(significant_components).dot(significant_components.T)

def process_artifact_rejection(X_training, queue_artifact_rejection, queue_gui):
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])

    buffer_size = int(0.9 * samplingRate)
    X_noisy_test = np.empty((len(channelNames), 0))
    label_buffer = []

    significant_components = perform_svd_and_select_components(X_training)
    message = f"[{datetime.now()}] [Artifact Rejection] Significant components shape: {significant_components.shape}"
    print(message)
    queue_gui.put(message)

    while True:
        sample, timestamp = inlet.pull_sample()
        eeg_sample = np.array(sample[:-1]).reshape(len(channelNames), -1)
        label = sample[-1]  # Last element is the label

        X_noisy_test = np.hstack((X_noisy_test, eeg_sample))
        label_buffer.append(label)

        if X_noisy_test.shape[1] >= buffer_size:
            X_clean_test = denoise_data(X_noisy_test, significant_components, queue_gui)
            current_labels = label_buffer[:buffer_size]

            # Push the processed data and labels to the queue
            queue_artifact_rejection.put((X_clean_test, current_labels))
            message = f"[{datetime.now()}] [Artifact Rejection] Data pushed to queue. Shape: {X_clean_test.shape}"
            print(message)
            queue_gui.put(message)
            queue_gui.put(f"===================================")

            # Reset for next batch
            X_noisy_test = np.empty((len(channelNames), 0))
            label_buffer = label_buffer[buffer_size:]

def run_artifact_rejection(queue_artifact_rejection, queue_gui):
    num_channels = len(channelNames)
    num_samples = 450
    X_training = generate_synthetic_training_data(num_channels, num_samples)
    message = f"[{datetime.now()}] [Artifact Rejection] Starting artifact rejection..."
    print(message)
    queue_gui.put(message)
    process_artifact_rejection(X_training, queue_artifact_rejection, queue_gui)