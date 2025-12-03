# Hybrid Rough & Smooth Volatility Modeling Toolkit

A small, research-oriented Python toolkit for simulating and visualizing **rough**, **smooth**, and **hybrid** stochastic volatility models.

> This project is **educational / research-focused**, not a trading system.

---

## 1. Overview

This repository implements:

- A **rough Heston** model with a fractional Volterra kernel (Hurst $H < 0.5$)
- A **classical smooth Heston** model (CIR variance)
- A **hybrid model** with total variance  
  $V(t) = V_{\text{rough}}(t) + V_{\text{smooth}}(t)$
- Basic analytics such as **Hurst exponent estimation**, **variance autocorrelation**, and **ATM skew term structure**
- High-level visualizations that reproduce key stylized facts from the rough volatility literature

The main script simulates these models, generates diagnostic plots, and saves them as PNG files.

---

## 2. Theoretical Background

The implementation and diagnostics are inspired by:

- **Gatheral, Jaisson & Rosenbaum (2018)** – “Volatility is rough”
- **Bayer, Friz & Gatheral (2016)** – “Pricing under rough volatility”
- **El Euch & Rosenbaum (2019)** – “The characteristic function of rough Heston”

Key ideas:

- Equity index volatility appears **“rough”**, with Hurst exponent $H \approx 0.1$
- Rough volatility models naturally reproduce the **$T^{-0.4}$** decay of ATM skew
- Hybrid models combining rough and smooth factors can capture **multi-scale variance dynamics**

---

## 3. Implemented Models

### 3.1 `RoughHeston`

A simplified rough Heston model with fractional kernel:
- Parameters: `hurst`, `nu`, `v0`, `rho`
- Uses a discrete Volterra convolution with kernel  
  $K(t) \propto t^{H - 1/2}$
- Simulates stock and variance paths via correlated Brownian motions

### 3.2 `SmoothHeston`

A classical Heston model with square-root variance diffusion:
- Parameters: `kappa`, `theta`, `xi`, `v0`, `rho`
- Euler–Maruyama discretization for the variance process
- Standard log-Euler step for the stock price

### 3.3 `HybridModel`

A two-factor hybrid model:
- Rough component: fractional-kernel variance `V_r`
- Smooth component: CIR/Heston-like variance `V_s`
- Total variance: `V_total = V_r + V_s`
- Allows visualization of **two-scale variance decomposition** and relative contributions over time

---

## 4. Analysis & Visualization

The script includes several analysis/plotting utilities:

### 4.1 Hurst Estimation

`estimate_hurst(data, dt)`

- Uses the **variance of increments at multiple scales** to estimate a Hurst exponent
- Applied separately to rough, smooth, and hybrid variance paths
- Produces log–log plots with fitted regression lines

### 4.2 `plot_variance_paths(models_dict, ...)`

Generates a 2×2 figure:

1. **Sample variance paths** (rough vs smooth vs hybrid)  
2. **Hurst estimation plot** (log variance vs log scale)  
3. **Terminal variance distribution** (histograms)  
4. **Variance autocorrelation** (ACF over lags)

Saved as: `variance_comparison.png`

---

### 4.3 `plot_atm_skew_term_structure()`

Produces an illustrative **ATM skew term structure**:

- Rough model: approximately decays like $T^{-0.4}$ (as in SPX data)
- Smooth Heston: almost flat in maturity

Saved as: `atm_skew.png`

This reproduces the canonical stylized fact from Gatheral–Jaisson–Rosenbaum (2018).

---

### 4.4 `plot_hybrid_decomposition(model, ...)`

Visualizes the hybrid model’s **two-scale variance structure**:

1. Rough vs smooth vs total variance over time  
2. Relative contribution of each component to total variance  
3. Hurst estimates for rough and smooth components  
4. Stock price paths with mean path overlay

Saved as: `hybrid_decomposition.png`

---

## 5. Running the Demo

### 5.1 Requirements

- Python 3.9+ (recommended)
- `numpy`
- `matplotlib`
- `scipy`

You can install dependencies with:

```bash
pip install numpy matplotlib scipy
```

## Known Limitations

- **Performance**: Volterra simulation is O(N²). For production use, 
  implement FFT-based convolution or Markovian approximation.
- **No pricing engine**: Currently simulation-only. Characteristic 
  function pricing (El Euch & Rosenbaum 2019) would enable fast option pricing.
- **No calibration**: Parameters are manually chosen. See [references] 
  for calibration methods.