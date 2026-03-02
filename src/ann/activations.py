"""Activation Functions and TIt's Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np
# ReLu function
class Relu:
     def forward(self, z):
          self.z = z
          return np.maximum(0,z)
     
     def backward(self, da):
         return da * (self.z > 0)

# Sigmoid function forward and backward    
class Sigmoid:    
     def forward(self, z):
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
                