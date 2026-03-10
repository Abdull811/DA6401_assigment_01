import argparse
import numpy as np
import os
import sys
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ann.neural_network import NeuralNetwork
from src.utils.data_loader import load_data


best_config = argparse.Namespace(
    dataset="mnist",
    epochs=2,
    batch_size=64,
    loss="cross_entropy",
    optimizer="sgd",
    weight_decay=0.0,
    learning_rate=0.01,
    num_layers=2,
    hidden_size=[64, 64],
    activation="relu",
    weight_init="xavier",
)


def main():
    model = NeuralNetwork(best_config)
    weights = np.load("src/best_model.npy", allow_pickle=True).item()
    model.set_weights(weights)

    _, _, _, _, x_test, y_test = load_data(best_config.dataset)
    logits = model.forward(x_test)
    y_pred_labels = np.argmax(logits, axis=1)

    macro_f1 = f1_score(y_test, y_pred_labels, average="macro")
    weighted_f1 = f1_score(y_test, y_pred_labels, average="weighted")

    print("Macro F1 Score:", macro_f1)
    print("Weighted F1 Score:", weighted_f1)

if __name__ == "__main__":
    main()
