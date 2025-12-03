import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import norm
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

def estimate_hurst(data, dt):
    scales = [2, 4, 8, 16, 32]
    variances = []
    
    for scale in scales:
        if scale < len(data) // 5:
            increments = data[scale:] - data[:-scale]
            variances.append(np.var(increments))
    
    if len(variances) < 3:
        return 0.5, None, None
    
    log_scales = np.log(scales[:len(variances)])
    log_vars = np.log(variances)
    
    slope = np.polyfit(log_scales, log_vars, 1)[0]
    H_est = slope / 2
    
    return H_est, log_scales, log_vars
