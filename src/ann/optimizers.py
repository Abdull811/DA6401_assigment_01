"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp.
"""
import numpy as np

# Stochastic Gradient Descent (SGD)
class SGD:
    def __init__(self, lr, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, layers):
        for i in layers:
            # Gradient of weight and bias
            grad_w = i.grad_w + self.weight_decay * i.w
            grad_b = i.grad_b

            # Update parameters
            i.w = i.w - self.lr * grad_w
            i.b = i.b - self.lr * grad_b

# Momentum
class Momentum:
    def __init__(self, lr, beta, weight_decay):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.v_w = []
        self.v_b = []

    def update(self, layers):

        # Initialize velocity 
        if len(self.v_w) == 0:
            for j in layers:
                self.v_w.append(np.zeros_like(j.w))
                self.v_b.append(np.zeros_like(j.b))

        for i, j in enumerate(layers):

            grad_w = j.grad_w + self.weight_decay * j.w
            grad_b = j.grad_b

            # Velocity update
            self.v_w[i] = self.beta * self.v_w[i] + grad_w
            self.v_b[i] = self.beta * self.v_b[i] + grad_b

            # Parameter update
            j.w -= self.lr * self.v_w[i]
            j.b -= self.lr * self.v_b[i]

# Nesterov Accelerated Gradient (NAG)            
class NAG:
    def __init__(self, lr, beta, weight_decay):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.v_w = []
        self.v_b = []

    def update(self, layers):

        # Initialize velocity if first call
        if len(self.v_w) == 0:
            for j in layers:
                self.v_w.append(np.zeros_like(j.w))
                self.v_b.append(np.zeros_like(j.b))

        for i, j in enumerate(layers):

            grad_w = j.grad_w + self.weight_decay * j.w
            grad_b = j.grad_b

            # Store previous velocity
            v_prev_w = self.v_w[i]
            v_prev_b = self.v_b[i]

            # Update velocity
            self.v_w[i] = self.beta * self.v_w[i] + grad_w
            self.v_b[i] = self.beta * self.v_b[i] + grad_b

            # Nesterov update 
            j.w -= self.lr * (self.beta * v_prev_w + grad_w)
            j.b -= self.lr * (self.beta * v_prev_b + grad_b)

# RMSProp
class RMSProp:
    def __init__(self, lr, beta, eps, weight_decay):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay
        self.s_w = []
        self.s_b = []

    def update(self, layers):

        # Initialize running average 
        if len(self.s_w) == 0:
            for j in layers:
                self.s_w.append(np.zeros_like(j.w))
                self.s_b.append(np.zeros_like(j.b))

        for i, j in enumerate(layers):

            grad_w = j.grad_w + self.weight_decay * j.w
            grad_b = j.grad_b

            # Update squared gradient average
            self.s_w[i] = ( self.beta * self.s_w[i]
                + (1 - self.beta) * (grad_w ** 2))

            self.s_b[i] = ( self.beta * self.s_b[i]
                + (1 - self.beta) * (grad_b ** 2))

            # Parameter update
            j.w -= self.lr * grad_w / (np.sqrt(self.s_w[i]) + self.eps)
            j.b -= self.lr * grad_b / (np.sqrt(self.s_b[i]) + self.eps)