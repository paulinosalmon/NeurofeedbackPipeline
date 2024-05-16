import math
import tensorflow as tf
from tensorflow.keras.layers import (Dense, Flatten, Dropout, Reshape, GRU, Input, 
                                    LayerNormalization, MultiHeadAttention, Add, Activation)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam                

def create_eeg_model(input_shape=(23, 110, 1), num_classes=2):
    inputs = Input(shape=input_shape)
    
    # Reshape input to match the format required for GRU
    x = Reshape((input_shape[0] * input_shape[1], input_shape[2]))(inputs)  # Shape: (None, 2530, 1)
    
    # GRU layer
    x = GRU(128, return_sequences=True)(x)  # Shape: (None, 2530, 128)
    
    # Flatten layer
    x = Flatten()(x)  # Shape: (None, 323840)
    
    # Dense layer
    x = Dense(64, activation='relu', kernel_regularizer='l2')(x)  # Shape: (None, 64)
    x = Dropout(0.5)(x)
    
    # Output layer for binary classification
    outputs = Dense(1, activation='sigmoid')(x)  # Shape: (None, 1)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model
    
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * math.exp(-0.1)