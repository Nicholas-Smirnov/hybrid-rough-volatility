import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import norm
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from EstimateHurst import estimate_hurst

def plot_variance_paths(models_dict, T=1.0, N=1000, M=5000):
    """Plot variance paths for multiple models"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Rough vs Smooth vs Hybrid Volatility Dynamics', 
                 fontsize=16, fontweight='bold')
    
    results = {}
    for name, model in models_dict.items():
        t, S, V = model.simulate(T=T, N=N, M=M)[:3]
        results[name] = (t, S, V)
    
    # Plot 1: Sample paths
    ax = axes[0, 0]
    for name, (t, S, V) in results.items():
        ax.plot(t, V[0], label=name, linewidth=2, alpha=0.8)
    ax.set_xlabel('Time (years)', fontsize=11)
    ax.set_ylabel('Variance', fontsize=11)
    ax.set_title('Sample Variance Paths', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Hurst estimation
    ax = axes[0, 1]
    for name, (t, S, V) in results.items():
        H_est, log_scales, log_vars = estimate_hurst(V[0], t[1])
        if log_scales is not None:
            ax.scatter(log_scales, log_vars, label=f'{name} (H≈{H_est:.2f})', s=60)
            # Fit line
            coeffs = np.polyfit(log_scales, log_vars, 1)
            ax.plot(log_scales, coeffs[0] * log_scales + coeffs[1], '--', alpha=0.6)
    
    ax.set_xlabel('log(Scale)', fontsize=11)
    ax.set_ylabel('log(Variance)', fontsize=11)
    ax.set_title('Hurst Estimation (Variance Method)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Distribution
    ax = axes[1, 0]
    for name, (t, S, V) in results.items():
        ax.hist(V[:, -1], bins=50, alpha=0.5, label=name, density=True)
    ax.set_xlabel('Terminal Variance', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Terminal Variance Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Autocorrelation
    ax = axes[1, 1]
    max_lag = 100
    for name, (t, S, V) in results.items():
        acf = [np.corrcoef(V[0, :-lag], V[0, lag:])[0, 1] 
               for lag in range(1, min(max_lag, len(V[0])//2))]
        ax.plot(acf, label=name, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Lag', fontsize=11)
    ax.set_ylabel('Autocorrelation', fontsize=11)
    ax.set_title('Variance Autocorrelation', fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
