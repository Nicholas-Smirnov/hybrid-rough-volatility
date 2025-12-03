import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import norm
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class SmoothHeston:
    """Classical Heston with square-root diffusion"""
    
    def __init__(self, kappa=2.0, theta=0.04, xi=0.5, v0=0.04, rho=-0.7):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.v0 = v0
        self.rho = rho
        
    def simulate(self, T=1.0, N=1000, M=5000, S0=100.0):
        """Simulate classical Heston paths"""
        dt = T / N
        t = np.linspace(0, T, N + 1)
        
        S = np.zeros((M, N + 1))
        V = np.zeros((M, N + 1))
        S[:, 0], V[:, 0] = S0, self.v0
        
        dW_S = np.random.randn(M, N) * np.sqrt(dt)
        dW_V_indep = np.random.randn(M, N) * np.sqrt(dt)
        dW_V = self.rho * dW_S + np.sqrt(1 - self.rho**2) * dW_V_indep
        
        for i in range(N):
            V[:, i + 1] = V[:, i] + self.kappa * (self.theta - V[:, i]) * dt + \
                         self.xi * np.sqrt(np.maximum(V[:, i], 0)) * dW_V[:, i]
            V[:, i + 1] = np.maximum(V[:, i + 1], 1e-8)
            
            S[:, i + 1] = S[:, i] * np.exp(
                -0.5 * V[:, i] * dt + np.sqrt(V[:, i]) * dW_S[:, i]
            )
        
        return t, S, V

