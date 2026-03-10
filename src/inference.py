"""
Inference Script
Evaluate trained models on test sets
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.data_loader import load_data
from src.ann.neural_network import NeuralNetwork

# Parse command line arguments for model evaluation
def parse_arguments():
    
    parser = argparse.ArgumentParser(description='Run inference on test set')
    
    parser.add_argument("-m", "--model_path", default="src/best_model.npy")
    parser.add_argument("-d", "--dataset", default="mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-nhl", "--num_layers", type=int, default=2)
    parser.add_argument("-sz", "--hidden_size", nargs="+", type=int, default=[128, 64])
    parser.add_argument("-a", "--activation", default="relu",
                        choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-l", "--loss", default="cross_entropy",
                        choices=["cross_entropy", "mse"])
    parser.add_argument("-wi", "--weight_init", default="xavier",
                        choices=["random", "xavier"])
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0001)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)

    return parser.parse_args()

def resolve_model_path(model_path):
    candidates = [model_path, "src/best_model.npy", "best_model.npy"]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("Could not find model checkpoint")

def build_args_from_weights(args, weights):
    layer_indices = sorted(
        int(key[1:]) for key in weights.keys()
        if key.startswith("w") and key[1:].isdigit())

    if layer_indices:
        args.hidden_size = [weights[f"w{i}"].shape[1] for i in layer_indices[:-1]]
        args.num_layers = len(args.hidden_size)

    return args

# Resolve model path, load weights, build config and initialize model
def load_model(args):
    args.model_path = resolve_model_path(args.model_path)
    weights = np.load(args.model_path, allow_pickle=True).item()
    args = build_args_from_weights(args, weights)

    model = NeuralNetwork(args)
    model.set_weights(weights)

    return model   

# Run inference on the test set and compute evaluation metrics
def evaluate_model(model, X_test, y_test): 
    logits = model.forward(X_test)
    y_pred = np.argmax(logits, axis=1)
    loss = model.loss_fn.forward(y_test, logits)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return { "logits": logits, "loss": loss, "accuracy": accuracy, 
              "precision": precision, "recall": recall, "f1": f1}

# Main function to run inference and print results 
def main():
    wandb.init(project="da6401_Assigment_01_weight_bias", mode="disabled")
    args = parse_arguments()
    _, _, _, _, x_test, y_test = load_data(args.dataset)

    model = load_model(args)
    results = evaluate_model(model, x_test, y_test)

    print(f"Loss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"F1-score: {results['f1']:.4f}")
    print("Evaluation complete!")

    return results

if __name__ == '__main__':
    main()
