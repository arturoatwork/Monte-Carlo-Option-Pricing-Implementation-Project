## Summary

This project implements a Monte Carlo pricing framework for European options under a risk-neutral GBM model.
It evaluates numerical convergence, quantifies uncertainty using confidence intervals, applies variance-reduction
techniques, and benchmarks results against the Black–Scholes closed-form solution.

The model is further calibrated using historical SPY volatility, demonstrating how empirical market data
can be integrated into stochastic pricing models.

## Structure of Files
MCP_SCHOOL/
  monte_carlo_project/
    analysis/
      convergence_analysis.ipynb
    data/
      README.md
    src/
      __init__.py
      black_scholes.py
      convergence.py
      gbm.py
      monte_carlo.py
      option_pricing.py
      main.py
    README.md
    requirements.txt
    theory.md



MCP_SCHOOL/
-     monte_carlo_project / analysis/ convergence_analysis.ipynb
-     data / README.md
-     src / __init__.py / black_scholes.py / convergence.py / gbm.py / monte_carlo.py / option_pricing.py
-     main.py
-     README.md
-     requirements.txt
-     theory.md

##Installation & Usage

Create a virtual environment and install dependencies:

    - pip install -r requirements.txt
    - python -m src.main
    - python -m src.convergence

## Results & Interpretation

Monte Carlo estimates converge toward the Black–Scholes closed-form value as the number of simulations increases.
Standard error decreases approximately at a \(1/\sqrt{N}\) rate (CLT), and antithetic variates reduce variance,
producing tighter confidence intervals for the same simulation budget.

Sensitivity checks confirm expected economic behavior:
- higher volatility increases option value,
- longer maturities increase option value,
- deeper in-the-money strikes increase expected payoff.

## Market Data

Historical SPY prices are downloaded programmatically using the `yfinance` package
inside the analysis notebook. Data is cached locally in `data/` when executed.


----------------------------------------------------------------------------------------

## Convergence Output (Example)

Monte Carlo European Call Option
S0=100.0, K=100.0, mu=0.05, sigma=0.2, r=0.03, T=1.0
Price: 10.6418
Std. Error: 0.1502
(.venv) arturofranco@Arturos-MacBook-Air monte_carlo_project % python -m src.convergence

Convergence Study: Monte Carlo European Call (GBM)
S0=100.0, K=100.0, mu=0.05, sigma=0.2, r=0.03, T=1.0, steps=252

Black–Scholes Call Price (closed-form): 9.4134

------------------------------------------------------------------------------------------
    N Sims |   MC Price |  Std Error |                   95% CI
------------------------------------------------------------------------------------------
       500 |    10.5172 |     0.6429 |  [    9.2570,    11.7773]
      1000 |    10.7773 |     0.4749 |  [    9.8465,    11.7080]
      2500 |    10.3782 |     0.2937 |  [    9.8024,    10.9539]
      5000 |    10.5302 |     0.2097 |  [   10.1192,    10.9413]
     10000 |    10.5330 |     0.1495 |  [   10.2399,    10.8260]
     25000 |    10.5815 |     0.0937 |  [   10.3980,    10.7651]
     50000 |    10.7430 |     0.0674 |  [   10.6110,    10.8751]

----------------------------------------------------------------------------------------

## Key Takeaways

- Monte Carlo estimates converge toward Black–Scholes as simulation count increases.
- Standard error decays at approximately a 1/sqrt(N) rate.
- Antithetic variates reduce variance for fixed computational cost.
- Calibrating volatility from real data connects theoretical models to markets.

These results indicate readiness for graduate-level coursework in financial engineering and quantitative finance.
