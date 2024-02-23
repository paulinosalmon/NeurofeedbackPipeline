# classifier.py
import numpy as np
import time
import joblib
import os

from settings import classifier, config, config_score, channelNames
from datetime import datetime
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.utils import shuffle

# ================== Model Evaluation Functions ================== #

def calculate_mean_error_rate_cv(test_X, test_y, n_trials=50, n_folds=10, model_path="../model/trained_classifier.pkl", report_path="../reports/classifier_report.txt"):
    """
    Calculate the mean classifier error rate over N trials using cross-validation.
    """
    mean_trial_error_rates = []  # List to store mean error rate for each trial

    kf = StratifiedKFold(n_splits=n_folds)

    for trial in range(n_trials):
        trial_error_rates = []  # List to store error rates for each fold in this trial
        randomized_X, randomized_y = shuffle(test_X, test_y, random_state=trial)

        for train_index, test_index in kf.split(randomized_X, randomized_y):
            X_train, X_test = randomized_X[train_index], randomized_X[test_index]
            y_train, y_test = randomized_y[train_index], randomized_y[test_index]

            # Load the saved model
            model = joblib.load(model_path)

            # Train and predict
            model.fit(X_train, y_train)
            predicted_y = model.predict(X_test)

            # Calculate error rate for the fold and append to the trial list
            error_rate = 1 - accuracy_score(y_test, predicted_y)
            trial_error_rates.append(error_rate)

        # Calculate the mean error rate for this trial
        mean_trial_error_rate = np.mean(trial_error_rates)
        mean_trial_error_rates.append(mean_trial_error_rate)

        print(f"Trial {trial + 1} Error Rate: {mean_trial_error_rate * 100}%")

    # Calculate the overall mean error rate 
    mean_error_rate = np.mean(mean_trial_error_rates)

    # Write the error rates to a file
    with open(report_path, "w") as file:
        file.write(f"Individual Trial Error Rates:\n")
        for i, error_rate in enumerate(mean_trial_error_rates, 1):
            file.write(f"Trial {i}: {error_rate * 100}%\n")
        file.write(f"\nOverall Mean Classifier Error Rate (over {n_trials} trials with {n_folds}-fold CV): {mean_error_rate * 100}%\n")

    return mean_error_rate

def save_data(X, y, directory="../data/", prefix=""):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    x_filename = os.path.join(directory, "X_test.npy")
    y_filename = os.path.join(directory, "y_test.npy")

    np.save(x_filename, X)
    np.save(y_filename, y)
    print(f"Saved X to {x_filename}")
    print(f"Saved y to {y_filename}")

# =============== Current iteration evaluation ================== #

def evaluate_model(classifier, X, y, cv=3):
    """
    Evaluate the classifier using cross-validation and return the average score.
    """
    scores = cross_val_score(clone(classifier), X, y, cv=cv)
    return np.mean(scores)

def evaluate_saved_model(test_X, test_y):
    """
    Evaluate the saved classifier model and return the mean classifier decoding error rate.

    Parameters:
    - test_X: Test features
    - test_y: True labels for the test data

    Returns:
    - Mean classifier decoding error rate
    """
    # Load the saved model
    model = joblib.load("../model/trained_classifier.pkl")

    # Predict using the loaded model
    predicted_y = model.predict(test_X)

    # Calculate accuracy
    accuracy = accuracy_score(test_y, predicted_y)

    # Calculate and return error rate
    error_rate = 1 - accuracy
    return error_rate

# ================== Calculations after classifier output ==================== #

def calculate_bias_offset(classifier, X, y, cv=3, limit=0.125):
    """
    Calculate the bias offset based on cross-validation predicted probabilities.
    """
    # Perform cross-validation and get the probabilities for the positive class
    cross_val_probs = cross_val_predict(clone(classifier), X, y, cv=cv, method='predict_proba')
    # Calculate the mean probability across folds for the positive class
    mean_proba = np.mean(cross_val_probs[:, 1])
    # Calculate the bias as the deviation from the chance level (0.5)
    bias = mean_proba - 0.5
    # Limit the bias to within the specified range
    bias_offset = np.clip(bias, -limit, limit)
    return bias_offset

def calculate_weighted_moving_average(data, alpha=0.5):
    """Calculate the weighted moving average of the data with an exponential decay."""
    wma = np.zeros_like(data)
    wma[0] = data[0]
    for i in range(1, len(data)):
        wma[i] = alpha * data[i] + (1 - alpha) * wma[i - 1]
    return wma

def calculate_classifier_output(probabilities, task_relevant_index):
    """
    Calculate the classifier output as the difference between
    the probability of the task-relevant category and the task-irrelevant category.
    """
    task_irrelevant_index = 1 - task_relevant_index
    classifier_output = probabilities[:, task_relevant_index] - probabilities[:, task_irrelevant_index]
    return classifier_output

# ==================== Main Function ========================== #

def run_classifier(queue_gui, queue_artifact_rejection, queue_classifier):
    global is_model_generated
    queue_gui.put(f"[{datetime.now()}] [Classifier] Initializing classifier...")
    model_path = "../model/trained_classifier.pkl"
    past_probabilities = []
    feedback_bias_offsets = []

    while True:
        try:
            queue_gui.put(f"[{datetime.now()}] [Classifier] Waiting for data...")
            X_processed, y_processed = queue_artifact_rejection.get(block=True)

            # Convert list to NumPy array if necessary and transpose
            X_processed = np.array(X_processed).T if isinstance(X_processed, list) else X_processed.T
            y_processed = np.array(y_processed).flatten() if isinstance(y_processed, list) else y_processed.flatten()

            # Check dimensions and ensure they match
            if X_processed.shape[0] != y_processed.shape[0]:
                raise ValueError("Number of samples in X and y do not match")

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.2, random_state=42
            )

            # Save test data for evaluation after pipeline termination
            save_data(X_test, y_test)
            # Train the classifier
            queue_gui.put(f"[{datetime.now()}] [Classifier] Current model is: {classifier}")
            # Train the classifier
            classifier.fit(X_train, y_train)
        
            # Evaluate the model
            current_score = evaluate_model(classifier, X_test, y_test)
            queue_gui.put(f"[{datetime.now()}] [Classifier] Current model score: {current_score}")

            # Save the model if it performs better
            if current_score > config_score["best_score"]:
                config_score["best_score"] = current_score
                joblib.dump(classifier, model_path)
                queue_gui.put(f"[{datetime.now()}] [Classifier] New best model saved.")

            # ============= Perform probability computations (LR) + bias offset on classifier output ====================== #

            queue_gui.put(f"[{datetime.now()}] [Classifier] Classifier trained.")

            # Estimate prediction probabilities
            probabilities = classifier.predict_proba(X_train)
            past_probabilities.append(probabilities)

            # Calculate bias offset
            if len(past_probabilities) == 1:
                # For the first feedback run
                bias_offset = calculate_bias_offset(classifier, X_train, y_train)
            else:
                # For subsequent feedback runs, use the bias of the four most recent feedback blocks
                recent_probabilities = past_probabilities[-4:]
                recent_avg_proba = np.mean([probs[:, 1] for probs in recent_probabilities], axis=0)
                recent_bias = recent_avg_proba - 0.5
                bias_offset = np.clip(recent_bias, -0.125, 0.125)

            # Ensure bias_offset is a scalar before printing
            if not np.isscalar(bias_offset):
                bias_offset = np.mean(bias_offset)  # or bias_offset[0] if you want the first element

            queue_gui.put(f"[{datetime.now()}] [Classifier] Bias offset calculated: {bias_offset}")

            # ============ Perform WMA and apply transfer function to probabilities ================ #

            # If more than one set of probabilities, calculate WMA
            if len(past_probabilities) > 1:
                proba_array = np.array(past_probabilities)
                wma_probabilities = np.apply_along_axis(calculate_weighted_moving_average, 0, proba_array)
                classifier_output = calculate_classifier_output(wma_probabilities[-1], task_relevant_index=1)
            else:
                classifier_output = calculate_classifier_output(probabilities, task_relevant_index=1)

            # Apply bias offset to the classifier output
            classifier_output -= bias_offset

            # ============== Print and send visibility scores to queue for feedback generation ===================== #

            average_output_probability = np.mean(classifier_output)
            queue_gui.put(f"[{datetime.now()}] [Classifier] Classifier output: {average_output_probability:.3f}")

            queue_gui.put(f"[{datetime.now()}] [Classifier] Data processing completed.")

            # Push the processed data and labels to the queue
            queue_classifier.put(average_output_probability)
            message = f"[{datetime.now()}] [Classifier] Data pushed to queue: {average_output_probability}"
            print(message)
            queue_gui.put(message)
            queue_gui.put(f"===================================")


        except Exception as e:
            queue_gui.put(f"[{datetime.now()}] [Classifier] Error encountered: {e}")
            break

    
