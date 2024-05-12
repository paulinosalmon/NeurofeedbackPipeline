# Separate script for the deep learning model for easier modularity

from tensorflow.keras.models import Sequential
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten, Reshape
from settings import channelNames, channelNamesExcluded

# Define the function to create the RNN model
def create_rnn_model(input_shape=(23, 110, 1), num_classes=2):
    model = Sequential([
        # Reshape input to 2D time series
        Reshape((input_shape[0], input_shape[1] * input_shape[2]), input_shape=input_shape),

        # LSTM layers
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),

        # Flatten and fully connected layers
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer='l2'),  # L2 regularization
        Dropout(0.5),

        # Ensure single output neuron for binary classification
        Dense(1, activation='sigmoid')
    ])
    return model