import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import norm
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from EstimateHurst import estimate_hurst

def plot_atm_skew_term_structure():
    """Plot ATM skew decay: key signature of rough volatility"""
    maturities = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3])
    
    # Theoretical skew decay
    rough_skew = -0.7 * maturities**(-0.4)  # T^(H-0.5) with H≈0.1
    smooth_skew = -0.3 * maturities**(-0.05)  # Nearly flat
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(maturities, rough_skew, 'o-', linewidth=3, markersize=10,
            label='Rough Vol (H=0.1): ∼T^(-0.4)', color='#e74c3c')
    ax.plot(maturities, smooth_skew, 's-', linewidth=3, markersize=10,
            label='Smooth (Heston): ∼T^(-0.05)', color='#3498db')
    
    ax.set_xlabel('Maturity (years)', fontsize=13, fontweight='bold')
    ax.set_ylabel('ATM Skew', fontsize=13, fontweight='bold')
    ax.set_title('ATM Skew Term Structure\n(Gatheral et al. 2018 stylized fact)', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    # Add text annotation
    ax.text(0.5, -0.5, 'Empirical SPX data shows T^(-0.4) decay\n→ Evidence for H ≈ 0.1',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig
