# Separate script for the deep learning model for easier modularity

from tensorflow.keras.models import Sequential
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten, Reshape, SimpleRNN, GRU
from settings import channelNames, channelNamesExcluded

# Define the function to create the RNN model
def create_eeg_model(input_shape=(23, 110, 1), num_classes=2):
    model = Sequential([
        # Reshape input to match the format required for GRU
        Reshape((input_shape[0] * input_shape[1], input_shape[2]), input_shape=input_shape),  # Shape: (None, 2530, 1)
        
        # GRU layer
        GRU(128, return_sequences=True),  # Shape: (None, 2530, 128)
        
        # Flatten layer
        Flatten(),  # Shape: (None, 323840)
        
        # Dense layer
        Dense(64, activation='relu', kernel_regularizer='l2'),  # Shape: (None, 64)
        Dropout(0.5),
        
        # Output layer for binary classification
        Dense(1, activation='sigmoid')  # Shape: (None, 1)
    ])
    return model