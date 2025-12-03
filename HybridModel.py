import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import norm
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class HybridModel:
    """Hybrid: V(t) = V_rough(t) + V_smooth(t)"""
    
    def __init__(self, hurst=0.10, nu_r=0.3, kappa_s=1.0, theta_s=0.03, 
                 xi_s=0.2, v_r0=0.02, v_s0=0.02, rho=-0.7):
        self.H = hurst
        self.nu_r = nu_r
        self.kappa_s = kappa_s
        self.theta_s = theta_s
        self.xi_s = xi_s
        self.v_r0 = v_r0
        self.v_s0 = v_s0
        self.rho = rho
        
    def simulate(self, T=1.0, N=1000, M=5000, S0=100.0):
        """Simulate hybrid paths"""
        dt = T / N
        t = np.linspace(0, T, N + 1)
        
        S = np.zeros((M, N + 1))
        V_r = np.zeros((M, N + 1))
        V_s = np.zeros((M, N + 1))
        
        S[:, 0] = S0
        V_r[:, 0] = self.v_r0
        V_s[:, 0] = self.v_s0
        
        # Fractional kernel for rough component
        kernel = lambda k: (k * dt)**(self.H - 0.5) / gamma(self.H + 0.5)
        weights = np.array([kernel(k) for k in range(1, N + 1)])[::-1]
        
        # Brownian increments
        dW_S = np.random.randn(M, N) * np.sqrt(dt)
        dW_r = np.random.randn(M, N) * np.sqrt(dt)
        dW_s = np.random.randn(M, N) * np.sqrt(dt)
        
        for i in range(N):
            # Rough component (fractional)
            if i == 0:
                volterra = 0
            else:
                past_innov = self.nu_r * dW_r[:, :i+1]
                volterra = np.sum(past_innov * weights[-(i+1):].reshape(1, -1), axis=1)
            
            V_r[:, i + 1] = np.maximum(self.v_r0 + volterra, 1e-8)
            
            # Smooth component (CIR)
            V_s[:, i + 1] = V_s[:, i] + self.kappa_s * (self.theta_s - V_s[:, i]) * dt + \
                           self.xi_s * np.sqrt(np.maximum(V_s[:, i], 0)) * dW_s[:, i]
            V_s[:, i + 1] = np.maximum(V_s[:, i + 1], 1e-8)
            
            # Total variance
            V_total = V_r[:, i] + V_s[:, i]
            
            # Stock price
            S[:, i + 1] = S[:, i] * np.exp(
                -0.5 * V_total * dt + np.sqrt(V_total) * dW_S[:, i]
            )
        
        return t, S, V_r + V_s, V_r, V_s

