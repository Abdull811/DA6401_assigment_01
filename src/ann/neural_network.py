"""
Main Neural Network Model class
Handles forward and backward propagation
"""

import numpy as np
import wandb

from src.ann.neural_layer import NeuralLayer
from src.ann.activations import ReLu, Sigmoid, Tanh
from src.ann.objective_functions import CrossEntropyLoss, MSELoss


class NeuralNetwork:

    def __init__(self, cli_args):
        np.random.seed(42)
        self.layers = []
        self.activations = []

        input_dim = 784
        hidden_sizes = cli_args.hidden_size
        activation_name = cli_args.activation
        weight_init = cli_args.weight_init
        loss_name = cli_args.loss

        self.weight_decay = cli_args.weight_decay

        layer_dims = [input_dim] + hidden_sizes + [10]

        # Create layers
        for i in range(len(layer_dims) - 1):

            self.layers.append(
                NeuralLayer(layer_dims[i], layer_dims[i + 1], weight_init))

            if i < len(layer_dims) - 2:

                if activation_name == "relu":
                    self.activations.append(ReLu())

                elif activation_name == "sigmoid":
                    self.activations.append(Sigmoid())

                elif activation_name == "tanh":
                    self.activations.append(Tanh())

        # Loss
        if loss_name == "cross_entropy":
            self.loss_fn = CrossEntropyLoss()
        else:
            self.loss_fn = MSELoss()

    # Forward pass
    def forward(self, X):
        out = X

        for i in range(len(self.activations)):

            z = self.layers[i].forward(out)
            out = self.activations[i].forward(z)

            if wandb.run is not None:
                wandb.log({"layer1_activation_mean": np.mean(out),
                    "layer1_activation_fraction": np.mean(out == 0)})

        logits = self.layers[-1].forward(out)

        return logits

    # Backpropagation

    def backward(self, y_true, logits):

        self.loss_fn.forward(y_true, logits)
        dz = self.loss_fn.backward()

        # Output layer
        da = self.layers[-1].backward(dz, self.weight_decay)

        # Hidden layers
        for i in reversed(range(len(self.activations))):
            dz = self.activations[i].backward(da)
            da = self.layers[i].backward(dz, self.weight_decay)

        # Collect gradients
        grad_W = []
        grad_b = []

        for layer in self.layers:
            grad_W.append(layer.grad_w)
            grad_b.append(layer.grad_b)

        self.grad_W = np.array(grad_W, dtype=object)
        self.grad_b = np.array(grad_b, dtype=object)

        return self.grad_W, self.grad_b

    # Evaluation

    def evaluate(self, X, y):
        logits = self.forward(X)
        preds = np.argmax(logits, axis=1)
        accuracy = np.mean(preds == y)

        return accuracy

    # Save weights
    def get_weights(self):
        weights = {}
        for i, layer in enumerate(self.layers):
            weights[f"w{i}"] = layer.w.copy()
            weights[f"b{i}"] = layer.b.copy()

        return weights

    # Load weights
    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            layer.w = weight_dict[f"w{i}"].copy()
            layer.b = weight_dict[f"b{i}"].copy()
