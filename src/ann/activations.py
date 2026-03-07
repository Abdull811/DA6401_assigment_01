"""Activation Functions and TIt's Derivatives
Implements: ReLU, Sigmoid, Tanh
"""

import numpy as np
# ReLu function
class ReLu:
     def forward(self, z):
          self.z = z
          return np.maximum(0,z)
     
     def backward(self, da):
         return da * (self.z > 0)

# Sigmoid function forward and backward    
class Sigmoid:    
     def forward(self, z):
         z = np.clip(z, -500, 500)
         self.a = 1 / ( 1 + np.exp(-z))
         return self.a
     
     def backward(self, da):
         dz = da * self.a *(1 - self.a)
         return dz
        
# Tanh activation function forward and backward
class Tanh:
      def forward(self, z):
          self.a = np.tanh(z)
          return self.a

      def backward(self, da):
          dz = da * (1 - self.a ** 2) 
          return dz

class Softmax(x):
     def softmax(x):
         x = x - np.max(x, axis=1, keepdims=True)
         exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
