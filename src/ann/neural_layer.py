import numpy as np

class NeuralLayer:

    def __init__(self, input_dim, output_dim, weight_init):

        if weight_init == "random":
            self.w = np.random.randn(input_dim, output_dim) * 0.01
            np.random.seed(42)
        elif weight_init == "xavier":
            std = np.sqrt(2.0 / (input_dim + output_dim))
            rng = np.random.RandomState(42)
            self.w = rng.randn(input_dim, output_dim) * std
        else:
            raise ValueError("weight_init must be 'random' or 'xavier'")

        self.b = np.zeros((1, output_dim))

        # gradients
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.w + self.b

    def backward(self, dz, weight_decay):

        # gradient of weights
        self.grad_w = self.x.T @ dz
        # gradient of bias
        self.grad_b = np.sum(dz, axis=0, keepdims=True)
        # gradient wrt input
        dx = dz @ self.w.T

        return dx