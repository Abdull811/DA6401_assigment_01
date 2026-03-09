"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp.
"""
import numpy as np

# Stochastic Gradient Descent (SGD)
class SGD:
    def __init__(self, lr=0.01, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay
    def update(self, layers):
        for layer in layers:
            layer.w -= self.lr * (layer.grad_w + self.weight_decay * layer.w)
            layer.b -= self.lr * layer.grad_b

# Momentum
# v = beta * v + grad
# w = w - lr * v
class Momentum:
    def __init__(self, lr, beta=0.9, weight_decay=0.0):
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

            # Velocity update
            self.v_w[i] = self.beta * self.v_w[i] + (j.grad_w + self.weight_decay * j.w)
            self.v_b[i] = self.beta * self.v_b[i] + j.grad_b

            # Parameter update
            j.w -= self.lr * self.v_w[i]
            j.b -= self.lr * self.v_b[i]

# Nesterov Accelerated Gradient (NAG)   
# w_look = w - beta * v
# v = beta * v + grad
# w = w - lr * v
         
class NAG:
    def __init__(self, lr, beta=0.9, weight_decay=0.0):

        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay

        self.v_w = []
        self.v_b = []

    def initialize(self, layers):
        # Initialize velocity 
        if len(self.v_w) == 0:
            for j in layers:
                self.v_w.append(np.zeros_like(j.w))
                self.v_b.append(np.zeros_like(j.b))
    
    def lookahead(self, layers):
        self.initialize(layers)
        for i, j in enumerate(layers):
            # Store previous velocity
            j.w -= self.beta * self.v_w[i]
            j.b -= self.beta * self.v_b[i]
    
    def update(self, layers):
            # Initialize velocity if first step
            if len(self.v_w) == 0:
                for j in layers:
                    self.v_w.append(np.zeros_like(j.w))
                    self.v_b.append(np.zeros_like(j.b))
                    
            for i, j in enumerate(layers):
                # Restore parameters
                j.w += self.beta * self.v_w[i]
                j.b += self.beta * self.v_b[i] 
                # update velocity

                self.v_w[i] = self.beta * self.v_w[i] + (j.grad_w + self.weight_decay * j.w)
                self.v_b[i] = self.beta * self.v_b[i] + j.grad_b

                # Final Nesterov update 
                j.w -= self.lr * self.v_w[i]
                j.b -= self.lr * self.v_b[i]

# RMSProp
# s = beta * s + (1-beta)(grad ** 2)
# w = w - lr * grad / (sqrt(s) + eps)

class RMSProp:
    def __init__(self, lr, beta=0.9, eps=1e-8, weight_decay=0.0):
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

            # Update squared gradient average
            self.s_w[i] = ( self.beta * self.s_w[i]
                + (1 - self.beta) * (j.grad_w ** 2))

            self.s_b[i] = ( self.beta * self.s_b[i]
                + (1 - self.beta) * (j.grad_b ** 2))

            # Parameter update
            j.w -= self.lr * (j.grad_w + self.weight_decay * j.w) / (np.sqrt(self.s_w[i]) + self.eps)
            j.b -= self.lr * j.grad_b / (np.sqrt(self.s_b[i]) + self.eps)
