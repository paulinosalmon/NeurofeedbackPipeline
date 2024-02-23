import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Import settings from settings.py (assuming it's in the same directory)
from settings import classifier

# Load the EEG data and labels
X = np.load('../data/train/EEG_epochs_sample.npy')
y = np.load('../data/train/y_categories_sample.npy')

# Since EEG data is 3D and classifier expects 2D data, we need to reshape
# This flattens the samples and channels into one long feature vector for each trial
X_flattened = X.reshape(X.shape[0], -1)

# Split the data into training and test sets (if needed)
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.2, random_state=42)

# Save the test data
np.save('../data/train/X_test.npy', X_test)
np.save('../data/train/y_test.npy', y_test)

# Define a preprocessing pipeline with StandardScaler and the classifier from settings
pipeline = make_pipeline(StandardScaler(), classifier)

# Train the classifier
pipeline.fit(X_train, y_train)

# Save the trained classifier to a file
joblib.dump(pipeline, '../model/trained_classifier.pkl')
print('Classifier has been pretrained and saved to trained_classifier.pkl')
