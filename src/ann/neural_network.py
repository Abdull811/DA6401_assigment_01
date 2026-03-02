"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from src.ann.neural_layer import NeuralLayer
from src.ann.activations import Relu, Sigmoid, Tanh
from src.ann.objective_functions import CrossEntropyLoss, MSELoss

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        self.layers = []
        self.activations = []

        input_dim = cli_args.input_dim
        hidden_sizes = cli_args.hidden_sizes
        activation_name = cli_args.activation
        weight_init = cli_args.weight_init
        loss_name = cli_args.loss
        self.loss_name = cli_args.loss

        new_dim = [input_dim] + hidden_sizes + [10]
        # Layers
        for i in range(len(new_dim)-1):
            self.layers.append(NeuralLayer(new_dim[i], new_dim[i + 1], weight_init))

            # Activation function for hidden layers
            if i < len(new_dim) - 2:
               if activation_name == "relu" :
                   self.activations.append(Relu())
               elif activation_name == "sigmoid":
                   self.activations.append(Sigmoid())
               elif activation_name == "tanh" :
                   self.activations.append(Tanh())

        # Objective function
        if loss_name == "cross_entropy":
           self.loss_fn = CrossEntropyLoss()
        else:
            self.loss_fn = MSELoss()

        self.weight_decay = cli_args.weight_decay
        self.learning_rate = cli_args.learning_rate                           

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        x_v = X
        # Hidden layers
        for i in range(len(self.activations)):
            z = self.layers[i].forward(x_v)
            x_v = self.activations[i].forward(z)

        # Last layer 
        y_pred = self.layers[-1].forward(x_v)
        return y_pred   

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        self.loss_fn.forward(y_true, y_pred)
        dz = self.loss_fn.backward()

        grad_W_list = []
        grad_b_list = []

        # Backprop through layers in reverse; collect grads so that index 0 = last layer
        # LAst layer backpropagation
        da = self.layers[-1].backward(dz, self.weight_decay)
        grad_W_list.append(self.layers[-1].grad_w)
        grad_b_list.append(self.layers[-1].grad_b)
        
        # Hidden layer backpropagation
        for i in reversed(range(len(self.activations))):
            dz = self.activations[i].backward(da)
            da = self.layers[i].backward(dz, self.weight_decay)

            grad_W_list.append(self.layers[i].grad_w)
            grad_b_list.append(self.layers[i].grad_b)

        # create explicit object arrays to avoid numpy trying to broadcast shapes
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
        print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
        return self.grad_W, self.grad_b
    
    # weight update
    def update_weights(self):
        for i, j in enumerate(self.layers):
            j.W -= self.learning_rate * j.grad_w
            j.b -= self.learning_rate * j.grad_b
    
    # Training
    def train(self, X_train, y_train, epochs=1, batch_size=32):
        n = X_train.shape[0]
        for epoch in range(epochs):
            for i in range(0, n, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                y_pred = self.forward(X_batch)
                self.backward(y_batch, y_pred)
                self.update_weights()

    def evaluate(self, X, y):
        y_pred = self.forward(X)
        if self.loss_name == "cross_entropy":
            predictions = np.argmax(y_pred, axis=1)
            true_labels = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == true_labels)
            return accuracy
        else:
            mse = self.loss_fn.forward(y, y_pred)
            return mse

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()