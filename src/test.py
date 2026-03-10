import argparse
import numpy as np
import os
import sys
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ann.neural_network import NeuralNetwork
from src.utils.data_loader import load_data


def resolve_model_path():
    candidates = [
        "src/best_model.npy",
        "best_model.npy",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("Could not find best_model.npy in expected locations")


def build_config_from_weights(weights):
    layer_indices = sorted(
        int(key[1:]) for key in weights.keys()
        if key.startswith("w") and key[1:].isdigit()
    )

    if not layer_indices:
        raise ValueError("No layer weights found in checkpoint")

    hidden_size = [weights[f"w{i}"].shape[1] for i in layer_indices[:-1]]

    return argparse.Namespace(
        dataset="mnist",
        epochs=2,
        batch_size=64,
        loss="cross_entropy",
        optimizer="sgd",
        weight_decay=0.0,
        learning_rate=0.01,
        num_layers=len(hidden_size),
        hidden_size=hidden_size,
        activation="relu",
        weight_init="xavier",
    )


def main():
    model_path = resolve_model_path()
    weights = np.load(model_path, allow_pickle=True).item()
    config = build_config_from_weights(weights)

    model = NeuralNetwork(config)
    model.set_weights(weights)

    _, _, _, _, x_test, y_test = load_data(config.dataset)
    logits = model.forward(x_test)
    y_pred_labels = np.argmax(logits, axis=1)

    macro_f1 = f1_score(y_test, y_pred_labels, average="macro")
    weighted_f1 = f1_score(y_test, y_pred_labels, average="weighted")

    print("F1 Score:", weighted_f1)
    print("Macro F1 Score:", macro_f1)
    print("Weighted F1 Score:", weighted_f1)

if __name__ == "__main__":
    main()
