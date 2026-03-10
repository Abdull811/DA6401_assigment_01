import numpy as np
import sys
import os

np.random.seed(42)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.ann.neural_layer import NeuralLayer
from src.ann.activations import ReLu, Sigmoid, Tanh
from src.ann.objective_functions import CrossEntropyLoss, MSELoss

class NeuralNetwork:

    def __init__(self, cli_args):

        self.layers = []
        self.activations = []

        input_dim = 784
        hidden_sizes = cli_args.hidden_size
        activation_name = cli_args.activation
        weight_init = cli_args.weight_init
        loss_name = cli_args.loss

        self.weight_decay = cli_args.weight_decay

        layer_dims = [input_dim] + hidden_sizes + [10]

        # build layers
        for i in range(len(layer_dims) - 1):

            layer = NeuralLayer(layer_dims[i], layer_dims[i + 1], weight_init)
            self.layers.append(layer)

            self._set_param_alias(i, layer)

            if i < len(layer_dims) - 2:

                if activation_name == "relu":
                    self.activations.append(ReLu())

                elif activation_name == "sigmoid":
                    self.activations.append(Sigmoid())

                elif activation_name == "tanh":
                    self.activations.append(Tanh())

                else:
                    raise ValueError("Unknown activation")

        # loss
        if loss_name == "cross_entropy":
            self.loss_fn = CrossEntropyLoss()

        elif loss_name == "mse":
            self.loss_fn = MSELoss()

        else:
            raise ValueError("Unknown loss")

    def forward(self, X):
        self._sync_layers_from_attrs()

        out = X

        for i in range(len(self.layers) - 1):
            z = self.layers[i].forward(out)
            out = self.activations[i].forward(z)

        logits = self.layers[-1].forward(out)

        return logits

    def backward(self, y_true, logits):
        self._sync_layers_from_attrs()

        dz = self.loss_fn.backward(y_true, logits)

        dz, _, _ = self.layers[-1].backward(dz, self.weight_decay)

        for i in reversed(range(len(self.layers) - 1)):
            dz = self.activations[i].backward(dz)
            dz, _, _ = self.layers[i].backward(dz, self.weight_decay)

        self.grad_W = []
        self.grad_b = []

        for i, layer in enumerate(self.layers):
            self.grad_W.append(layer.grad_w)
            self.grad_b.append(layer.grad_b)
            self._set_param_alias(i, layer)

        return self.grad_W, self.grad_b

    def evaluate(self, X, y):

        logits = self.forward(X)
        preds = np.argmax(logits, axis=1)

        accuracy = np.mean(preds == y)

        return accuracy

    def get_weights(self):
        self._sync_layers_from_attrs()

        weights = {}

        for i, layer in enumerate(self.layers):
            weights[f"w{i}"] = layer.w.copy()
            weights[f"b{i}"] = layer.b.copy()

        return weights

    def set_weights(self, weight_dict):

        for i, layer in enumerate(self.layers):

            w_key = f"w{i}"
            b_key = f"b{i}"
            W_key = f"W{i}"
            B_key = f"B{i}"

            if w_key in weight_dict:
                layer.w = weight_dict[w_key]
            elif W_key in weight_dict:
                layer.w = weight_dict[W_key]

            if b_key in weight_dict:
                layer.b = weight_dict[b_key]
            elif B_key in weight_dict:
                layer.b = weight_dict[B_key]
            self._set_param_alias(i, layer)

    def _set_param_alias(self, index, layer):
        layer._sync_aliases()
        setattr(self, f"w{index}", layer.w)
        setattr(self, f"b{index}", layer.b)
        setattr(self, f"W{index}", layer.w)
        setattr(self, f"B{index}", layer.b)

    def _sync_layers_from_attrs(self):
        for i, layer in enumerate(self.layers):
            w_attr = getattr(self, f"w{i}", layer.w)
            b_attr = getattr(self, f"b{i}", layer.b)
            W_attr = getattr(self, f"W{i}", layer.w)
            B_attr = getattr(self, f"B{i}", layer.b)

            if w_attr is not layer.w:
                layer.w = w_attr
            elif W_attr is not layer.w:
                layer.w = W_attr

            if b_attr is not layer.b:
                layer.b = b_attr
            elif B_attr is not layer.b:
                layer.b = B_attr

            self._set_param_alias(i, layer)
