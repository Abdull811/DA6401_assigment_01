import numpy as np, argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ann.neural_network import NeuralNetwork

best_config = argparse.Namespace(
    dataset="mnist", epochs=2, batch_size=64,
    loss="cross_entropy", optimizer="sgd",
    weight_decay=0.0, learning_rate=0.01,
    num_layers=2, hidden_size=[64,64],
    activation="relu", weight_init="xavier"
)

model = NeuralNetwork(best_config)
weights = np.load("src/best_model.npy", allow_pickle=True).item()
model.set_weights(weights)
