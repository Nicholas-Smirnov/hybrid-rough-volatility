import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import norm
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from EstimateHurst import estimate_hurst


def plot_hybrid_decomposition(model, T=1.0, N=1000, M=1000):
    """Visualize rough + smooth decomposition in hybrid model"""
    t, S, V_total, V_r, V_s = model.simulate(T=T, N=N, M=M)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hybrid Model: Two-Scale Variance Decomposition', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Components
    ax = axes[0, 0]
    ax.plot(t, V_r[0], label='Rough Component', linewidth=2, color='#e74c3c')
    ax.plot(t, V_s[0], label='Smooth Component', linewidth=2, color='#3498db')
    ax.plot(t, V_total[0], label='Total Variance', linewidth=2.5, 
            color='#2ecc71', linestyle='--')
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Variance', fontsize=11)
    ax.set_title('Variance Decomposition: V = V_rough + V_smooth', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Contribution over time
    ax = axes[0, 1]
    contribution_r = np.mean(V_r, axis=0) / np.mean(V_total, axis=0)
    contribution_s = np.mean(V_s, axis=0) / np.mean(V_total, axis=0)
    
    ax.fill_between(t, 0, contribution_r, label='Rough', alpha=0.6, color='#e74c3c')
    ax.fill_between(t, contribution_r, 1, label='Smooth', alpha=0.6, color='#3498db')
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Proportion', fontsize=11)
    ax.set_title('Relative Contribution to Total Variance', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 3: Hurst for each component
    ax = axes[1, 0]
    H_r, log_scales_r, log_vars_r = estimate_hurst(V_r[0], t[1])
    H_s, log_scales_s, log_vars_s = estimate_hurst(V_s[0], t[1])
    
    if log_scales_r is not None:
        ax.scatter(log_scales_r, log_vars_r, label=f'Rough (H≈{H_r:.2f})', 
                  s=80, color='#e74c3c')
        coeffs = np.polyfit(log_scales_r, log_vars_r, 1)
        ax.plot(log_scales_r, coeffs[0]*log_scales_r + coeffs[1], '--', 
               color='#e74c3c', alpha=0.6)
    
    if log_scales_s is not None:
        ax.scatter(log_scales_s, log_vars_s, label=f'Smooth (H≈{H_s:.2f})', 
                  s=80, color='#3498db')
        coeffs = np.polyfit(log_scales_s, log_vars_s, 1)
        ax.plot(log_scales_s, coeffs[0]*log_scales_s + coeffs[1], '--', 
               color='#3498db', alpha=0.6)
    
    ax.set_xlabel('log(Scale)', fontsize=11)
    ax.set_ylabel('log(Variance)', fontsize=11)
    ax.set_title('Hurst Estimation for Each Component', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Stock price paths
    ax = axes[1, 1]
    for i in range(min(100, M)):
        ax.plot(t, S[i], alpha=0.1, color='gray', linewidth=0.5)
    ax.plot(t, np.mean(S, axis=0), linewidth=3, color='black', 
            label='Mean Path', linestyle='--')
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Stock Price', fontsize=11)
    ax.set_title('Stock Price Paths', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

