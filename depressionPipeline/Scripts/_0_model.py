import math
import tensorflow as tf
from tensorflow.keras.layers import (Dense, Flatten, Dropout, Reshape, GRU, Input, 
                                    LayerNormalization, Add, Activation, Attention, GlobalAveragePooling1D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam                

def transformer_block(inputs, ff_dim=128, dropout=0.25):
    # Simple Self-Attention
    attn_output = Attention()([inputs, inputs])
    attn_output = Dropout(dropout)(attn_output)
    res = Add()([attn_output, inputs])
    
    # Feed Forward Network
    ff_output = Dense(ff_dim, activation='relu')(res)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    res = Add()([ff_output, res])
    
    return res

def create_eeg_model(input_shape=(23, 110, 1), 
                     num_classes=2, 
                     ff_dim=128, 
                     dropout=0.25,
                     num_transformer_blocks=3):
    
    inputs = Input(shape=input_shape)
    
    x = Reshape((input_shape[0] * input_shape[1], input_shape[2]))(inputs)
    x = GRU(128, return_sequences=True)(x)
    
    # Apply multiple Transformer Blocks with Simple Attention
    for _ in range(num_transformer_blocks):
        x = transformer_block(x, ff_dim, dropout)
    
    x = Flatten()(x)
    x = Dense(64, activation='relu', kernel_regularizer='l2')(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model
    
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * math.exp(-0.1)