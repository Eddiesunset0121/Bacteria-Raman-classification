# src/data_processing.py

import os
import json
import numpy as np
import tensorflow as tf
from google.colab import userdata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def download_kaggle_dataset(dataset_name, target_dir="."):
    """
    Sets up Kaggle API credentials and downloads/unzips a dataset.
    """
    print("--- Setting up Kaggle API ---")
    # Get the API key from Colab Secrets
    kaggle_json = userdata.get('KAGGLE_JSON')
    
    # Define the path for the kaggle.json file
    kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_file_path = os.path.join(kaggle_dir, 'kaggle.json')
    
    # Write the secret to the file
    with open(kaggle_file_path, 'w') as f:
        f.write(kaggle_json)
    
    # Set permissions for the API key
    os.chmod(kaggle_file_path, 600)
    
    print(f"--- Downloading dataset: {dataset_name} ---")
    # Using os.system to run shell commands
    os.system(f"kaggle datasets download -d {dataset_name} -p {target_dir}")
    zip_file = f"{dataset_name.split('/')[-1]}.zip"
    os.system(f"unzip -o {os.path.join(target_dir, zip_file)} -d {target_dir}") # -o overwrites without asking
    print("--- Dataset downloaded and unzipped ---")


# Modify the load_data function to look in the current directory
def load_data(data_path="."): # Changed default path
    """Loads all numpy arrays from the specified path."""
    data = {}
    data['X_reference'] = np.load(os.path.join(data_path, "X_reference.npy"))
    data['y_reference'] = np.load(os.path.join(data_path, "y_reference.npy")).astype(np.int64)
    # ... continue loading all other .npy files in the same way ...
    data['X_test'] = np.load(os.path.join(data_path, "X_test.npy"))
    data['y_test'] = np.load(os.path.join(data_path, "y_test.npy")).astype(np.int64)
    
    print("Data loaded into memory successfully.")
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
