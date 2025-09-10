# src/train.py

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score

def train_model(model, datasets, learning_rate, epochs, batch_size, model_path):
    """Compiles and trains the model."""
    
    # Define callbacks [cite: 1149]
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max')
    
    # Compile the model [cite: 1151, 1152]
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    
    print("--- Starting Model Training ---")
    history = model.fit(
        datasets['train'][0], datasets['train'][1],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=datasets['validation'],
        callbacks=[checkpoint],
        shuffle=True
    )
    print("--- Model Training Complete ---")
    return history

def evaluate_model(model_path, datasets):
    """Loads the best model and evaluates it on the test set."""
    print("--- Evaluating Model on Test Set ---")
    best_model = tf.keras.models.load_model(model_path)
    
    # Make predictions [cite: 1308]
    y_pred_probs = best_model.predict(datasets['test'][0])
    y_pred_labels = tf.argmax(y_pred_probs, axis=1)
    
    # Calculate accuracy [cite: 1311]
    accuracy = accuracy_score(datasets['original_y_test'], y_pred_labels)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy, y_pred_labels
