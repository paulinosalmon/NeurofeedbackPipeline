import numpy as np
from numpy.linalg import svd
from pylsl import StreamInlet, resolve_stream
from datetime import datetime
import settings
import os

# Placeholder for offline testing
def generate_synthetic_feedback_signal(num_channels=len(settings.channelNames) - len(settings.channelNamesExcluded), 
                                       num_samples=90):
    """
    [For Offline Testing, If live EEG cap is active, x_noisy_test should get data from the feedback state]
    Generates a synthetic feedback signal for testing purposes along with labels.

    Parameters:
    - num_channels: Number of EEG channels.
    - num_samples: Number of samples per channel.

    Returns:
    - A tuple containing:
        - A NumPy array of shape (num_channels, num_samples) containing the synthetic feedback signal.
        - A label (0 for faces, 1 for scenes).
    """
    label = np.random.choice([0, 1])  # Randomly choose between faces (0) and scenes (1)
    if label == 0:  # Faces
        amplitude_range = (-100, 100)  # Define a specific amplitude range or pattern for faces
    else:  # Scenes
        amplitude_range = (-200, 200)  # Define a different amplitude range or pattern for scenes

    synthetic_signal = np.random.uniform(amplitude_range[0], amplitude_range[1], size=(num_channels, num_samples))
    return synthetic_signal, label

# Only active on training state/blocks
def create_significant_components(X_training, variance_threshold=0.10):
    U, S, Vt = svd(X_training, full_matrices=False)
    variance_explained = (S**2) / np.sum(S**2)
    num_components = np.sum(variance_explained >= variance_threshold)
    U_reduced = U[:, :num_components]
    Vt_reduced = Vt[:num_components, :]
    X_denoised = U_reduced @ np.diag(S[:num_components]) @ Vt_reduced
    return X_denoised, U_reduced, num_components

# Only active on feedback state/blocks
def denoise_feedback_data(X_noisy_test, significant_components, queue_gui):
    if significant_components.size == 0:
        message = f"[{datetime.now()}] [Artifact Rejection] No significant components found. Skipping denoising."
        queue_gui.put(message)
        return X_noisy_test
    
    # Calculate the projection of the noisy data onto the space spanned by the significant components
    projection = significant_components.dot(significant_components.T.dot(X_noisy_test))

    # Subtract the projection from the original noisy data to get the denoised data
    X_clean_test = X_noisy_test - projection

    return X_clean_test

# Main Function
def run_artifact_rejection(queue_gui, queue_preprocessed_data, queue_artifact_rejection, preprocessing_done, artifact_done):

    queue_gui.put(f"[{datetime.now()}] [Artifact Rejection] Beginning artifact rejection...")
    preprocessing_done.wait()  # Wait for preprocessing to complete
    trial_counter = 0  # Counter for the number of trials processed

    # Create the directory for significant components if it doesn't exist
    significant_components_dir = f"{settings.subject_path_init()}/{settings.subjID}"
    if not os.path.exists(significant_components_dir):
        os.makedirs(significant_components_dir)

    ####################### If training day, perform denoising on X_training only #######################
    if settings.training_state == 1:  # If training day
        queue_gui.put("[Artifact Rejection] Training state initiated.")

        # Path for significant components CSV file
        significant_components_path = f"{significant_components_dir}/significantComponents_{settings.subjID}_day_{settings.expDay}.csv"
        k_value_path = f"{significant_components_dir}/k_value_{settings.subjID}_day_{settings.expDay}.txt"

        # Continuously pull from the queue and update significant components
        while True:
            trial_counter += 1  # Increment the trial counter
            queue_gui.put(f"======== [Trial {trial_counter}] (TRAINING STATE) ========")
            X_training, label = queue_preprocessed_data.get()
            queue_gui.put(f"[{datetime.now()}] [Artifact Rejection] Data received from pre-processing stage.")
            print(f"[{datetime.now()}] [Artifact Rejection] Data shape received: {X_training.shape}")
            print(f"[{datetime.now()}] [Artifact Rejection] Label received: {label}")

            # Perform denoising and create significant components
            X_denoised, significant_components, k_value = create_significant_components(X_training)
            message = f"[{datetime.now()}] [Artifact Rejection] Denoised training data. Shape: {X_denoised.shape}"
            print(message)
            # queue_gui.put(message)

            # Update significant components in the CSV file
            np.savetxt(significant_components_path, significant_components, delimiter=",")
            with open(k_value_path, 'w') as f:
                f.write(str(k_value))

            # queue_gui.put(f"[{datetime.now()}] [Artifact Rejection] Significant components generated. Shape: {significant_components.shape}")

            # Send denoised X_training directly to the queue_artifact_rejection
            queue_artifact_rejection.put((X_denoised, label))
            queue_gui.put(f"[{datetime.now()}] [Artifact Rejection] Training state is on. Denoised X_training sent to queue. Shape: {X_denoised.shape}")
            artifact_done.set()  
            print(f"[{datetime.now()}] [Artifact Rejection] Event signaled.")

    ######################## If neurofeedback day, perform SVD on X_training and take into account X_noisy_test #######################
    else:
        # Load the significant components and K value from the saved files
        significant_components_path = f"{settings.subject_path_init()}/{settings.subjID}/significantComponents_{settings.subjID}_day_{settings.expDay}.csv"
        k_value_path = f"{significant_components_dir}/k_value_{settings.subjID}_day_{settings.expDay}.txt"

        # Assuming K value is saved correctly, load it
        with open(k_value_path, 'r') as f:
            k_value = int(f.read().strip())

        # Load significant components and assume it has the correct shape
        significant_components = np.loadtxt(significant_components_path, delimiter=",")
        significant_components = significant_components.reshape(-1, k_value)

        while True:
            trial_counter += 1  # Increment the trial counter
            queue_gui.put(f"======== [Trial {trial_counter}] (FEEDBACK STATE) ========")
            X_noisy_test, label = queue_preprocessed_data.get()
            queue_gui.put(f"[{datetime.now()}] [Artifact Rejection] Data received from pre-processing stage.")
            print(f"[{datetime.now()}] [Artifact Rejection] Data shape received: {X_noisy_test.shape}")

            # Denoise the synthetic noisy test data using the loaded significant components
            X_clean_test = denoise_feedback_data(X_noisy_test, significant_components, queue_gui)

            queue_artifact_rejection.put((X_clean_test, label))
            queue_gui.put(f"[{datetime.now()}] [Artifact Rejection] Clean data and label pushed to queue. Shape: {X_clean_test.shape}, Label: {label}")

            # Signal that artifact rejection is done for this batch
            print(f"[{datetime.now()}] [Artifact Rejection] Denoising done. Signaling event.")
            artifact_done.set()
            print(f"[{datetime.now()}] [Artifact Rejection] Event signaled.")


