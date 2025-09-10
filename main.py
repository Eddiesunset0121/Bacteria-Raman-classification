# main.py

import argparse
from src.data_processing import load_data, prepare_datasets
from src.models import build_cnn_model, MODEL_3_CONFIG
from src.train import train_model, evaluate_model
from src.utils import plot_loss_curves # and other utils

def main(args):
    # 1. Load Data
    raw_data = load_data(args.data_path)
    
    # 2. Prepare Datasets for Machine Learning
    datasets, encoder = prepare_datasets(raw_data)
    
    # 3. Build Model from Config
    input_shape = (1000, 1) # From your data's shape
    num_classes = 30 # Number of bacteria species
    
    print("--- Building Model ---")
    model = build_cnn_model(
        input_shape=input_shape,
        num_classes=num_classes,
        **MODEL_3_CONFIG
    )
    model.summary()
    
    # 4. Train Model
    history = train_model(
        model=model,
        datasets=datasets,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_path=args.model_path
    )
    
    # 5. Evaluate the best saved model
    accuracy, y_preds = evaluate_model(args.model_path, datasets)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bacteria Raman Classification Pipeline")
    parser.add_argument('--data_path', type=str, default='data/', help='Path to data files')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Optimizer learning rate')
    parser.add_argument('--model_path', type=str, default='best_model.keras', help='Path to save the best model')
    
    args = parser.parse_args()
    main(args)
