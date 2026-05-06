# Monte Carlo Option Pricing

Prices European options under a risk-neutral GBM model using Monte Carlo simulation.
Compares standard MC against antithetic variates and benchmarks both against the
Black-Scholes closed-form solution.

Historical SPY volatility is downloaded via `yfinance` and used to calibrate sigma,
connecting the theoretical model to real market data.

## Project Structure

```
monte_carlo_project/
├── main.py                        # single-run demo
├── src/
│   ├── gbm.py                     # risk-neutral GBM simulation
│   ├── option_pricing.py          # payoff functions and CI estimation
│   ├── black_scholes.py           # analytical call/put formulas
│   └── convergence.py             # convergence study and plots
├── analysis/
│   └── convergence_analysis.ipynb # full analysis notebook
├── data/                          # cached yfinance CSV (generated on first run)
├── requirements.txt
└── theory.md                      # mathematical background
```

## Methods

**GBM simulation** — Euler-Maruyama discretization under the risk-neutral measure:

$$S_{t+\Delta t} = S_t \exp\left[\left(r - \tfrac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\, Z\right], \quad Z \sim \mathcal{N}(0,1)$$

**Variance reduction** — antithetic variates: for each standard normal Z, also use -Z.
This halves variance when payoff and Z are negatively correlated.

**Black-Scholes benchmark** — closed-form call price used to measure MC error.

## Results

Single run (N=10,000, antithetic, seed=42):

```
S0=100.0, K=100.0, r=0.03, sigma=0.20, T=1.0, steps=252, N=10000
Monte Carlo Price: 9.3093
Std. Error:       0.1412
95% CI:           [9.0326, 9.5860]
Black-Scholes:    9.4134
Abs Error:        0.1041
```

Convergence study (standard MC vs antithetic, seed=42):

```
Black-Scholes Call Price: 9.4134

    N Sims |   MC Price     SE   AbsErr | Anti Price     SE   AbsErr
------------------------------------------------------------------------
       500 |     9.4835  0.6176   0.0701 |     9.2030  0.6386   0.2104
      1000 |     9.2841  0.4298   0.1293 |     9.2024  0.4397   0.2111
      2500 |     9.1141  0.2716   0.2993 |     9.2089  0.2741   0.2045
      5000 |     9.0211  0.1935   0.3923 |     9.3775  0.1996   0.0359
     10000 |     9.1809  0.1386   0.2325 |     9.3093  0.1412   0.1041
     25000 |     9.2708  0.0884   0.1426 |     9.2947  0.0889   0.1187
     50000 |     9.3089  0.0627   0.1045 |     9.3184  0.0629   0.0950
```

Standard error decays at approximately 1/sqrt(N) as expected from the CLT.

## Usage

```bash
pip install -r requirements.txt

# quick demo
python main.py

# convergence study with plots
python -m src.convergence

# full notebook
jupyter notebook analysis/convergence_analysis.ipynb
```

## Dependencies

```
numpy, scipy, matplotlib, pandas, yfinance, jupyter
```
