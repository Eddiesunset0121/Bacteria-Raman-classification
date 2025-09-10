# src/data_processing.py

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_data(data_path="data/"):
    """Loads all numpy arrays from the specified path."""
    data = {}
    data['X_reference'] = np.load(f"{data_path}X_reference.npy")
    data['y_reference'] = np.load(f"{data_path}y_reference.npy").astype(np.int64)
    data['X_test'] = np.load(f"{data_path}X_test.npy")
    data['y_test'] = np.load(f"{data_path}y_test.npy").astype(np.int64)
    data['X_finetune'] = np.load(f"{data_path}X_finetune.npy")
    data['y_finetune'] = np.load(f"{data_path}y_finetune.npy").astype(np.int64)
    data['X_2018clinical'] = np.load(f"{data_path}X_2018clinical.npy")
    data['y_2018clinical'] = np.load(f"{data_path}y_2018clinical.npy").astype(np.int64)
    # ... load other files as needed ...
    print("Data loaded successfully.")
    return data

def prepare_datasets(data):
    """Splits and preprocesses the raw data into final ML-ready datasets."""
    # 1. General Training Split [cite: 253, 254]
    X_train, X_val, y_train, y_val = train_test_split(
        data['X_reference'], data['y_reference'], test_size=0.05, random_state=42
    )

    # 2. Reshape features for Conv1D (add channel dimension) [cite: 558, 559]
    X_train = tf.expand_dims(X_train, axis=-1)
    X_val = tf.expand_dims(X_val, axis=-1)
    X_test = tf.expand_dims(data['X_test'], axis=-1)

    # 3. One-hot encode labels [cite: 653, 677, 680, 681]
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_val = encoder.transform(y_val.reshape(-1, 1))
    y_test_encoded = encoder.transform(data['y_test'].reshape(-1, 1))

    datasets = {
        'train': (X_train, y_train),
        'validation': (X_val, y_val),
        'test': (X_test, y_test_encoded),
        'original_y_test': data['y_test'] # Keep original for accuracy score
    }
    return datasets, encoder # Return encoder to use later
