# src/models.py

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

def build_cnn_model(input_shape, num_classes, conv_config, dense_config):
    """
    Builds a configurable 1D CNN model.

    Args:
        input_shape (tuple): Shape of the input data (e.g., (1000, 1)).
        num_classes (int): Number of output classes.
        conv_config (list of dicts): Configuration for convolutional layers.
        dense_config (list of dicts): Configuration for dense layers.
    
    Returns:
        A compiled Keras model.
    """
    model = Sequential([Input(shape=input_shape)])
    
    # Build Convolutional Base
    for layer_params in conv_config:
        model.add(Conv1D(**layer_params))
        model.add(BatchNormalization()) # Add BN after each conv layer as per your design
        
    model.add(Flatten())
    
    # Build Classifier Head
    for layer_params in dense_config:
        model.add(Dense(units=layer_params['units'], activation=layer_params['activation']))
        if 'dropout' in layer_params:
            model.add(Dropout(rate=layer_params['dropout']))
            
    # Final Output Layer
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

# Example of how to define the architecture for your 'model_3' [cite: 1135, 1146]
MODEL_3_CONFIG = {
    "conv_config": [
        {"filters": 40, "kernel_size": 10, "strides": 2, "padding": "same", "activation": "relu"},
        {"filters": 80, "kernel_size": 10, "strides": 2, "padding": "same", "activation": "relu"},
        {"filters": 160, "kernel_size": 10, "strides": 2, "padding": "same", "activation": "relu"},
        {"filters": 320, "kernel_size": 10, "strides": 2, "padding": "same", "activation": "relu"},
        {"filters": 640, "kernel_size": 10, "strides": 1, "padding": "same", "activation": "relu"}
    ],
    "dense_config": [
        {"units": 100, "activation": "relu", "dropout": 0.3}
    ]
}
