# Theory: Monte Carlo Pricing of European Options under GBM

This document summarizes the mathematical foundations behind the simulation, pricing, convergence analysis,
and variance reduction techniques used in this project.

---

## 1. Geometric Brownian Motion (GBM)

We model the stock price process \( S_t \) as:

\[
dS_t = \mu S_t dt + \sigma S_t dW_t
\]

where:
- \( \mu \) is the drift,
- \( \sigma \) is volatility,
- \( W_t \) is a standard Brownian motion.

---

## 2. Risk-Neutral Measure

For derivative pricing, we work under the **risk-neutral probability measure**.
In this world, all assets grow on average at the risk-free rate \( r \):

\[
dS_t = r S_t dt + \sigma S_t dW_t
\]

Monte Carlo simulation is therefore carried out with drift \( r \), not \( \mu \).

---

## 3. Discretization Scheme

Using Euler–Maruyama discretization in log-space:

\[
S_{t+\Delta t} = S_t \exp\left((r - \tfrac12\sigma^2)\Delta t + \sigma \sqrt{\Delta t} Z\right)
\]

where \( Z \sim \mathcal{N}(0,1) \).

---

## 4. European Option Payoff

For a European call option:

\[
\text{Payoff} = \max(S_T - K, 0)
\]

The discounted Monte Carlo estimator is:

\[
\hat C = e^{-rT} \frac{1}{N}\sum_{i=1}^N \max(S_T^{(i)} - K, 0)
\]

---

## 5. Monte Carlo Error & Central Limit Theorem

By the Central Limit Theorem:

\[
\sqrt{N}(\hat C - C) \xrightarrow{d} \mathcal{N}(0, \sigma^2)
\]

The standard error is estimated as:

\[
SE = \frac{s}{\sqrt{N}}
\]

where \( s \) is the sample standard deviation of discounted payoffs.

A 95% confidence interval is:

\[
\hat C \pm 1.96 \cdot SE
\]

---

## 6. Convergence & Law of Large Numbers

As \( N \to \infty \):

\[
\hat C \to C
\]

Empirically, this is observed through:
- stabilizing price estimates
- shrinking confidence intervals

---

## 7. Black–Scholes Benchmark

The analytical Black–Scholes call price is:

\[
C_{BS} = S_0 \Phi(d_1) - K e^{-rT}\Phi(d_2)
\]

with:

\[
d_1 = \frac{\ln(S_0/K) + (r + \frac12\sigma^2)T}{\sigma\sqrt{T}}, \quad
d_2 = d_1 - \sigma\sqrt{T}
\]

Monte Carlo estimates are compared against this closed-form solution.

---

## 8. Variance Reduction: Antithetic Variates

To improve efficiency, for each random vector \( Z \) we also simulate \( -Z \).
Averaging the two paths reduces variance without increasing the number of random draws.

Empirically, this results in lower standard error for the same \( N \).

---

## 9. Model Limitations

- GBM assumes constant volatility and lognormal returns.
- No transaction costs or jumps are modeled.
- Real markets often violate these assumptions.

Extensions could include stochastic volatility or jump–diffusion models.
