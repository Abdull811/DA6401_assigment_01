import numpy as np
# ReLu
class ReLu:

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dz):
        return dz * (self.x > 0)

# Sigmoid
class Sigmoid:

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dz):
        return dz * self.out * (1 - self.out)

# Tanh
class Tanh:

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dz):
        return dz * (1 - self.out ** 2)