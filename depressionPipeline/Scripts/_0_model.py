# Separate script for the deep learning model for easier modularity

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from settings import channelNames, channelNamesExcluded
from tensorflow.keras.regularizers import l2

# Define the function to create the CNN model
def create_cnn_model(input_shape=(23, 90, 1), num_classes=2):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),  # Additional Conv layer
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),  # L2 regularization
        Dropout(0.5),  # Dropout
        Dense(1, activation='sigmoid')
    ])
    return model