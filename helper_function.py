# --- IMPORTS ---
# BEST PRACTICE: All imports are at the top, sorted by standard, then third-party.
import os
import random
import zipfile
import itertools
import datetime as dt
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# --- CONSTANTS ---
# BEST PRACTICE: Define "magic numbers" as constants.
RESCALE_FACTOR = 1./255

# --- 1. Data Handling Functions ---

def download_and_unzip(url: str, save_path: str = "data.zip"):
    """Downloads and unzips a file from a URL, then removes the zip."""
    print(f"Downloading file from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"File saved to {save_path}")

    with zipfile.ZipFile(save_path, "r") as zip_ref:
        zip_ref.extractall()
    print("Unzipping complete.")

    os.remove(save_path)
    print(f"Removed zip file: {save_path}")

def walk_through_dir(dir_path: str):
    """Walks through a directory and prints its contents."""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def create_data_pipelines(train_dir: str, test_dir: str, batch_size: int = 32, image_size: tuple[int, int] = (224, 224)):
    """
    This function load the train and test data from directories: train_dir and test_dir
    and perform data argumentation on training and test sets.
    Argumentations include:
        image_resize = (224,224),
        image_rescale = 1/255.,
        define batch, 
        one-hot encode labels based on num_claases (binary or multi-class),
        random: flip, rotation, zoom
    Returns: 
        argumanted training and test sets
    """
    # 1. Detect number of classes and set the label_mode
    num_classes = len(os.listdir(train_dir))
    if num_classes > 2:
        label_mode = "categorical"
        print(f"Found {num_classes} classes. Using 'categorical' label_mode for multi-class classification.")
    else:
        label_mode = "int"
        print(f"Found 2 classes. Using 'int' label_mode for binary classification.")

    # 2. Create the initial datasets
    print("\nCreating data pipelines...")
    train_data = tf.keras.utils.image_dataset_from_directory(
        directory=train_dir,
        label_mode=label_mode, # Use the dynamically set mode
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True)

    test_data = tf.keras.utils.image_dataset_from_directory(
        directory=test_dir,
        label_mode=label_mode, # Use the same mode for test data
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False)

    # 3. Create a data augmentation layer
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ], name="data_augmentation")

    # 4. Build the performant pipelines
    # For training data, apply rescaling and augmentation
    # When you create a dataset using tf.keras.utils.image_dataset_from_directory, 
    # it structures your data into a sequence of pairs. Each pair is a tuple: (images, labels) = train_data.
    # (images, labels).map(function:iterable) 
    # function, iterable = lambda x, y: (data_augmentation(x, training=True), y
    # x = images, y = labels
    # map x, y --> (data_augmentation(x, training=True), y
    # lambda -> arguments : expressions
    train_data = train_data.map(lambda x, y: (tf.keras.layers.Rescaling(1./255)(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_data = train_data.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_data = train_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # For test data, only apply rescaling
    test_data = test_data.map(lambda x, y: (tf.keras.layers.Rescaling(1./255)(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_data = test_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    print("\nData pipelines created successfully.")
    return train_data, test_data

def load_and_prep_image(filename: str, img_shape: int = 224) -> tf.Tensor:
    """Reads an image from filename, turns it into a tensor, and reshapes it."""
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img * RESCALE_FACTOR
    return img

def view_random_image(target_dir: str, target_class: str):
    """Views a random image from a target class in a target directory."""
    target_folder = os.path.join(target_dir, target_class)
    random_image_path = os.path.join(target_folder, random.choice(os.listdir(target_folder)))
    
    img = mpimg.imread(random_image_path)
    plt.imshow(img)
    plt.title(f"{target_class} (shape: {img.shape})")
    plt.axis("off")
    plt.show()

# --- 2. Modeling Functions ---

def create_generalized_model(input_shape: Tuple[int, int, int], num_classes: int, base_model_fn) -> tf.keras.Model:
    """Creates a model using a specified Keras application as a base."""
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    base_model = base_model_fn(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    
    activation = 'softmax' if num_classes > 2 else 'sigmoid'
    output_unit = num_classes if num_classes > 2 else 1
    
    outputs = tf.keras.layers.Dense(output_unit, activation=activation, name="output_layer")(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def create_tensorboard_callback(dir_name: str, experiment_name: str) -> tf.keras.callbacks.Callback:
    """Creates a TensorBoard callback with a timestamped log directory."""
    log_dir = os.path.join(dir_name, experiment_name, dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback

# --- 3. Evaluation & Inference Functions ---

def plot_loss_curves(history: tf.keras.callbacks.History):
    """Plots training & validation accuracy and loss curves from a History object."""
    df = pd.DataFrame(history.history)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['accuracy'], label="Training Accuracy")
    plt.plot(df['val_accuracy'], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs") # FIX: Completed this line.
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(df['loss'], label="Training Loss")
    plt.plot(df['val_loss'], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epochs") # FIX: Completed this line.
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def calculate_results(y_true, y_pred) -> Dict[str, float]:
    """
    Calculates model accuracy, precision, recall and f1 score of a binary classification model.

    Args:
        y_true: True labels in the form of a 1D array.
        y_pred: Predicted labels in the form of a 1D array.

    Returns:
        A dictionary of accuracy, precision, recall, f1-score.
    """
    model_accuracy = accuracy_score(y_true, y_pred)
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results

def make_confusion_matrix(y_true, y_pred, classes: List[str] = None, figsize: Tuple[int, int] = (10, 10), text_size: int = 15):
    """Makes a labelled confusion matrix comparing predictions and ground truth labels."""
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    labels = classes if classes else np.arange(cm.shape[0])
    
    ax.set(title="Confusion Matrix",
           xlabel="Predicted Label",
           ylabel="True Label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)
    
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    plt.xticks(rotation=70, fontsize=text_size) # REFINED: Rotated labels for better visibility
    plt.yticks(fontsize=text_size)

    threshold = (cm.max() + cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=text_size)
    plt.show() # REFINED: Added plt.show() to display the plot reliably

def pred_and_plot(model: tf.keras.Model, filename: str, class_names: List[str]):
    """Imports an image, makes a prediction, and plots the result."""
    img = load_and_prep_image(filename)
    pred_probs = model.predict(tf.expand_dims(img, axis=0))
    pred_class = class_names[tf.argmax(pred_probs[0])]
    
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis("off")
    plt.show()
