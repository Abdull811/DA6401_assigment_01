import numpy as np
import wandb
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

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

        for i in range(len(layer_dims) - 1):

            layer = NeuralLayer(layer_dims[i], layer_dims[i+1], weight_init)

            self.layers.append(layer)

            # expose weights for autograder
            setattr(self, f"w{i}", layer.w)
            setattr(self, f"b{i}", layer.b)

            if i < len(layer_dims) - 2:

                if activation_name == "relu":
                    self.activations.append(ReLu())

                elif activation_name == "sigmoid":
                    self.activations.append(Sigmoid())

                elif activation_name == "tanh":
                    self.activations.append(Tanh())

                else:
                    raise ValueError("Unknown activation")

        if loss_name == "cross_entropy":
            self.loss_fn = CrossEntropyLoss()

        elif loss_name == "mse":
            self.loss_fn = MSELoss()

        else:
            raise ValueError("Unknown loss")

    # Forward pass
    def forward(self, X):

        out = X

        for i in range(len(self.activations)):

            z = self.layers[i].forward(out)
            out = self.activations[i].forward(z)

        logits = self.layers[-1].forward(out)

        return logits

    # Backward pass
    def backward(self, y_true, logits):

        self.loss_fn.forward(y_true, logits)

        dz = self.loss_fn.backward(y_true, logits)

        da = self.layers[-1].backward(dz, self.weight_decay)

        for i in reversed(range(len(self.activations))):

            dz = self.activations[i].backward(da)
            da = self.layers[i].backward(dz, self.weight_decay)

        self.grad_W = [layer.grad_w for layer in self.layers]
        self.grad_b = [layer.grad_b for layer in self.layers]

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

            if f"w{i}" not in weight_dict:
                raise KeyError(f"Missing weight w{i}")

            layer.w = weight_dict[f"w{i}"].copy()
            layer.b = weight_dict[f"b{i}"].copy()

            setattr(self, f"w{i}", layer.w)
            setattr(self, f"b{i}", layer.b)