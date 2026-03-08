import numpy as np

# ReLU
class ReLu:
    def forward(self, z):
        self.z = z
        return np.maximum(0, z)

    def backward(self, da):
        dz = da * (self.z > 0)
        return dz


# Sigmoid
class Sigmoid:
    def forward(self, z):
        z = np.clip(z, -500, 500)
        self.a = 1 / (1 + np.exp(-z))
        return self.a

    def backward(self, da):
        return da * self.a * (1 - self.a)


# Tanh
class Tanh:
    def forward(self, z):
        self.a = np.tanh(z)
        return self.a

    def backward(self, da):
        return da * (1 - self.a**2)


# Softmax
class Softmax:
    def forward(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        self.a = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.a

    def backward(self, da):
        return da