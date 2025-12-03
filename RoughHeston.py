import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import norm
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class RoughHeston:
    """Rough Heston with fractional kernel (H < 0.5)"""
    
    def __init__(self, hurst=0.10, nu=0.5, v0=0.04, rho=-0.7):
        self.H = hurst
        self.nu = nu
        self.v0 = v0
        self.rho = rho
        
    def simulate(self, T=1.0, N=1000, M=5000, S0=100.0):
        """Simulate rough Heston paths"""
        dt = T / N
        t = np.linspace(0, T, N + 1)
        
        # Initialize
        S = np.zeros((M, N + 1))
        V = np.zeros((M, N + 1))
        S[:, 0], V[:, 0] = S0, self.v0
        
        # Fractional kernel weights
        kernel = lambda k: (k * dt)**(self.H - 0.5) / gamma(self.H + 0.5)
        weights = np.array([kernel(k) for k in range(1, N + 1)])[::-1]
        
        # Correlated Brownian increments
        dW_S = np.random.randn(M, N) * np.sqrt(dt)
        dW_V_indep = np.random.randn(M, N) * np.sqrt(dt)
        dW_V = self.rho * dW_S + np.sqrt(1 - self.rho**2) * dW_V_indep
        
        # Simulate
        for i in range(N):
            if i == 0:
                volterra = 0
            else:
                past_innov = self.nu * np.sqrt(np.maximum(V[:, :i+1], 0)) * dW_V[:, :i+1]
                volterra = np.sum(past_innov * weights[-(i+1):].reshape(1, -1), axis=1)
            
            V[:, i + 1] = np.maximum(self.v0 + volterra, 1e-8)
            S[:, i + 1] = S[:, i] * np.exp(
                -0.5 * V[:, i] * dt + np.sqrt(V[:, i]) * dW_S[:, i]
            )
        
        return t, S, V
