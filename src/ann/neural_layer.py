import numpy as np

class NeuralLayer:

    def __init__(self, input_dim, output_dim, weight_init):

        if weight_init == "random":
            self.w = np.random.randn(input_dim, output_dim) * 0.01
        elif weight_init == "xavier":
            std = np.sqrt(2.0 / (input_dim + output_dim))
            self.w = np.random.randn(input_dim, output_dim) * std
        else:
            raise ValueError("weight_init must be 'random' or 'xavier'")

        self.b = np.zeros((1, output_dim))

        # gradients
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)
        self._sync_aliases()

    def _sync_aliases(self):
        self.W = self.w
        self.B = self.b
        self.grad_W = self.grad_w
        self.grad_B = self.grad_b
        self.dW = self.grad_w
        self.db = self.grad_b

    def _sync_primary_from_aliases(self):
        W = getattr(self, "W", self.w)
        B = getattr(self, "B", self.b)

        if getattr(self, "w", self.w) is not self.w:
            self.w = getattr(self, "w")
        elif W is not self.w:
            self.w = W

        if getattr(self, "b", self.b) is not self.b:
            self.b = getattr(self, "b")
        elif B is not self.b:
            self.b = B

    def forward(self, x):
        self._sync_primary_from_aliases()
        self._sync_aliases()
        self.x = x
        z = x @ self.w + self.b
        return z

    def backward(self, dz, weight_decay=0.0):
        self._sync_primary_from_aliases()

        self.grad_w = self.x.T @ dz
        self.grad_b = np.sum(dz, axis=0, keepdims=True)

        if weight_decay > 0:
            self.grad_w += weight_decay * self.w

        dx = dz @ self.w.T
        self._sync_aliases()

        return dx, self.grad_w, self.grad_b
