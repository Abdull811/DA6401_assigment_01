"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.data_loader import load_data
from src.ann.neural_network import NeuralNetwork

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    
    parser.add_argument("-m", "--model_path", required=True,
                        help="Relative path to saved model (.npy)")
    parser.add_argument("-d", "--dataset", required=True,
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-nhl", "--num_layers", type=int, required=True)
    parser.add_argument("-sz", "--hidden_size", nargs="+", type=int, required=True)
    parser.add_argument("-a", "--activation", required=True,
                        choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-l", "--loss", required=True,
                        choices=["cross_entropy", "mse"])
    parser.add_argument("-wi", "--weight_init", required=True,
                        choices=["random", "xavier"])
    parser.add_argument("-wd", "--weight_decay", type=float, required=True)
    parser.add_argument("-lr", "--learning_rate", type=float, required=True)
    
    return parser.parse_args()

def load_model(args):
    """
    Load trained model from disk.
    """
    model = NeuralNetwork(args)
    
    # Load weights
    layers_data = np.load(args.model_path, allow_pickle=True)

    weights = np.load(args.model_path, allow_pickle=True).item()
    model.set_weights(weights)

    return model   

def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    # forward pass 
    logits = model.forward(X_test)
    # convert logits to predicted labels
    y_pred = np.argmax(logits, axis=1)
    # compute loss
    loss = model.loss_fn.forward(y_test, logits)
    
    # Accuracy, Precision, F1-score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return { "logits": logits, "loss": loss, "accuracy": accuracy, 
              "precision": precision, "recall": recall, "f1": f1}

def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    # Load dataset
    _, _, _, _, x_test, y_test = load_data(args.dataset)

    # Load trained model
    model = load_model(args)

    # Evaluation metrices
    results = evaluate_model(model, x_test, y_test)
    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-score: {results['f1']:.4f}")
    print("Evaluation complete!")

    return results

if __name__ == '__main__':
    main()