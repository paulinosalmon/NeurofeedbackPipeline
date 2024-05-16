# classifier.py
import numpy as np
import time
import joblib
import os

from datetime import datetime
from settings import (model_path_init, subject_path_init, 
                      subjID, training_state, training_trials)

from _0_model import create_eeg_model, scheduler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

# ================== New Eval Functions ================== #

def train_and_evaluate_eeg_model(X, y, input_shape=(23, 110, 1), num_classes=2, normalize=False):
    # Stratified split to ensure similar distribution of the target variable in training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Convert labels to binary (if necessary)
    y_train = y_train.reshape((-1, 1)).astype(float)
    y_test = y_test.reshape((-1, 1)).astype(float)
    
    if normalize:
        # Normalize the data
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_train_reshaped = scaler.fit_transform(X_train_reshaped)
        X_test_reshaped = scaler.transform(X_test_reshaped)
        X_train = X_train_reshaped.reshape(X_train.shape)
        X_test = X_test_reshaped.reshape(X_test.shape)

    # Create the model
    model = create_eeg_model(input_shape=input_shape, num_classes=num_classes)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Setup the checkpoint path
    model_path = os.path.join(model_path_init(), "best_model.keras")
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(filepath=model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler_callback = LearningRateScheduler(scheduler)

    # Train the model with the callbacks
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), 
                        callbacks=[checkpoint_callback, early_stopping_callback, lr_scheduler_callback])

    # Load the best model saved
    best_model = load_model(model_path)

    # Evaluate the best model
    test_loss, test_acc = best_model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')
    print(f'CER: {1 - test_acc}')

    return best_model, test_acc
 
# ================== Fine tuning after classifier output ==================== #

def calculate_bias_offset(model_path, X, y, cv=3, limit=0.125):
    """
    Calculate the bias offset based on cross-validation predicted probabilities.
    """
    # Load the CNN model from the .keras file
    model = load_model(model_path)

    # Predict the probabilities for the positive class
    probabilities = model.predict(X)
    
    # Assuming binary classification, the positive class probability is the second value
    if probabilities.shape[1] == 2:
        positive_class_probs = probabilities[:, 1]
    else:
        # If the model only outputs one probability per sample, it represents the positive class
        positive_class_probs = probabilities[:, 0]

    # Calculate the mean probability of the positive class across all samples
    mean_proba = np.mean(positive_class_probs)
    # Calculate the bias as the deviation from the chance level (0.5)
    bias = mean_proba - 0.5
    # Limit the bias to within the specified range
    bias_offset = np.clip(bias, -limit, limit)
    return bias_offset

def calculate_wma(epochs):
    """
    Calculates the weighted moving average of the given list of epochs.
    Each epoch e_i is weighted by an exponentially decreasing factor according to the formula:
    e'_i = Σ (e_j / 2^(i-j+1)) for j=1 to i, plus (e_1 / 2^i)
    
    :param epochs: List of epoch values e_i where i is the trial number.
    :return: The weighted moving average of the epochs.
    """
    wma = []
    for i in range(len(epochs)):
        weighted_sum = 0
        for j in range(i + 1):
            weighted_sum += epochs[j] / (2 ** (i - j + 1))
        weighted_sum += epochs[0] / (2 ** i)  # Add e_1 / 2^i separately
        wma.append(weighted_sum)
    return wma

def calculate_classifier_output(probabilities, task_relevant_index):
    """
    Calculate the classifier output as the difference between
    the probability of the task-relevant category and the task-irrelevant category.
    """
    task_irrelevant_index = 1 - task_relevant_index
    classifier_output = probabilities[:, task_relevant_index] - probabilities[:, task_irrelevant_index]
    return classifier_output

def asymmetric_sigmoid_transfer(output, inflection_point=0.6, lower_bound=0.17, upper_bound=0.98):
    """
    Custom sigmoid function that maps the output to a specific range with an inflection point.
    
    Parameters:
    - output: Input value or array
    - inflection_point: The point at which the sigmoid curve has its midpoint
    - lower_bound: The lower bound of the output range
    - upper_bound: The upper bound of the output range
    
    Returns:
    - Sigmoid output mapped between lower_bound and upper_bound
    """
    sigmoid_range = upper_bound - lower_bound

    # Shift the input to adjust the inflection point
    adjusted_output = output - inflection_point

    # Apply the sigmoid function to the adjusted input
    sigmoid_output = 1 / (1 + np.exp(-adjusted_output))

    # Scale the output of the sigmoid function to the desired range
    scaled_output = lower_bound + sigmoid_range * sigmoid_output

    return scaled_output


# ==================== Main Function ========================== #

def run_classifier(queue_gui, queue_artifact_rejection, queue_classifier, artifact_done, classifier_done):
    queue_gui.put(f"[{datetime.now()}] [Classifier] Initializing classifier...")
    
    artifact_done.wait()  # Wait for preprocessing to complete

    # Initialize lists to store epochs and labels and WMA recordings
    all_epochs = []
    all_labels = []
    wma_list = []
    
    # The directory where the trained classifier and score will be saved
    model_directory = model_path_init()

    # The path where the trained classifier and score will be saved
    score_path = f"{model_directory}/{subjID}_classifier_score.txt"

    # Check if the directory exists, if not, create it
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
        queue_gui.put(f"[{datetime.now()}] [Classifier] Created directory: {model_directory}")

    while True:
        queue_gui.put(f"[{datetime.now()}] [Classifier] Waiting for data...")

        # If training model first, only accept data as labelled X_training to identify resting state of patient
        if training_state == 0 or training_state == 1: # dataset training or generative training
            X_processed, label = queue_artifact_rejection.get(block=True)
            all_epochs.append(X_processed)
            all_labels.append(label)
            queue_gui.put(f"[{datetime.now()}] [Classifier] {len(all_epochs)}/{training_trials} epochs collected.")

            # Send out 50-50 blend alpha
            alpha = 0.5
            print(f"[{datetime.now()}] [Classifier] Alpha fixed to 0.5 for training blocks.")
            placeholder_classifier_output = 0
            queue_classifier.put((alpha, placeholder_classifier_output))
            classifier_done.set()

            # Check if we have collected enough epochs for training
            # Only in classifier.py, we check if all trials are present (1200 in sample data) then train them in one go
            if len(all_epochs) == training_trials:
                # Reshape the data to 2D for the classifier
                # Uncomment if not a CNN or if the input data does not need to be flattened
                # X_reshaped = np.array(all_epochs).reshape(len(all_epochs), -1)
                X_reshaped = np.array(all_epochs).reshape(-1, 23, 110, 1)
                y_train = np.array(all_labels)
                queue_gui.put(f"[{datetime.now()}] [Classifier] All samples collected. Training model...")
                best_model, accuracy = train_and_evaluate_eeg_model(X_reshaped, y_train)
                error_rate = round(100 - round(accuracy * 100, 2), 2)
                queue_gui.put(f"[{datetime.now()}] [Classifier] CER: {error_rate}%")

                # Define the directory where to save the model and data
                save_directory = os.path.join(subject_path_init(), subjID)
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)
                
                training_data_path = os.path.join(save_directory, 'X_train.npy')
                training_labels_path = os.path.join(save_directory, 'y_labels.npy')

                # Define the paths for score
                score_path = os.path.join(model_directory, f"{subjID}_classifier_score.txt")

                # Save the model and score to respective subjectID folder
                with open(score_path, 'w') as f:
                    f.write(f"Model error rate: {error_rate}\n")
                queue_gui.put(f"[{datetime.now()}] [Classifier] Model and score saved.")

                # Save the training data and labels
                np.save(training_data_path, X_reshaped)
                np.save(training_labels_path, y_train)
                queue_gui.put(f"[{datetime.now()}] [Classifier] Training data and labels saved.")

                # Break out of the loop after training
                break
                    
        # If feedback blocks are toggled  (settings.training_state == 2)       
        else: 
            
            # Get the cleaned test data and labels from the queue
            X_clean_test, labels = queue_artifact_rejection.get(block=True)
            queue_gui.put(f"[{datetime.now()}] [Classifier] Feedback state is on. Loading pre-trained classifier.")

            # Calculate the current WMA and append it to the list
            current_wma = calculate_wma(wma_list + [X_clean_test])
            wma_list.append(current_wma[-1])  # Append only the WMA for the current epoch to the list

            # Flatten the WMA of the current epoch to match the shape expected by the classifier
            # X_wma_flattened = np.array(current_wma[-1]).reshape(1, -1)  # Reshape to (1, N); Use only if not DL model
            X_wma_flattened = np.array(current_wma[-1]).reshape(1, 23, 110, 1)
            
            # Load the pre-trained classifier
            model_path = os.path.join(model_path_init(), f"{subjID}_best_cnn.keras")
            model = load_model(model_path)

            # Load X_training data since first feedback block will have bias calculated based on this
            load_directory = os.path.join(subject_path_init(), subjID)
            training_data_path = os.path.join(load_directory, 'X_train.npy')
            training_labels_path = os.path.join(load_directory, 'y_labels.npy')
            X_train = np.load(training_data_path)
            y_train = np.load(training_labels_path)
            
            # Predict the probabilities using the loaded classifier
            # probabilities = model.predict_proba(X_wma_flattened) # Use for non DL models
            positive_proba = model.predict(X_wma_flattened)
            negative_proba = 1 - positive_proba
            probabilities = np.hstack([negative_proba, positive_proba])

            # Calculate the classifier output
            task_relevant_index = 1  # Assuming class 1 is task-relevant
            classifier_output = calculate_classifier_output(probabilities, task_relevant_index)

            # Calculate bias offset using the training data and labels
            bias_offset = calculate_bias_offset(model_path, X_train, y_train, cv=3, limit=0.125)

            # Adjust the classifier output for bias
            adjusted_classifier_output = classifier_output - bias_offset

            # Apply the transfer function to the adjusted classifier output
            alpha = asymmetric_sigmoid_transfer(adjusted_classifier_output)

            queue_gui.put(f"[{datetime.now()}] [Classifier] Alpha computed for this trial: {alpha}")
            queue_classifier.put((alpha, adjusted_classifier_output))

            # Signal that classifier processing is done for this batch
            classifier_done.set()
            print(f"[{datetime.now()}] [Classifier] Classifier processing done. Signaling event.")