from src.gbm import simulate_gbm_terminal
from src.option_pricing import discounted_payoff_call, estimate_mean_se_ci
from src.black_scholes import black_scholes_call


def run_demo():
    S0 = 100.0
    K = 100.0
    r = 0.03
    sigma = 0.2
    T = 1.0
    steps = 252
    n_simulations = 10_000

    ST = simulate_gbm_terminal(S0, r, sigma, T, steps, n_simulations, seed=42, antithetic=True)
    disc = discounted_payoff_call(ST, K, r, T)
    price, se, ci = estimate_mean_se_ci(disc)

    bs = black_scholes_call(S0, K, r, sigma, T)

    print("European Call Option (Risk-Neutral GBM Monte Carlo)")
    print(f"S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}, steps={steps}, N={n_simulations}")
    print(f"Monte Carlo Price: {price:.4f}")
    print(f"Std. Error:       {se:.4f}")
    print(f"95% CI:           [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"Black–Scholes:    {bs:.4f}")
    print(f"Abs Error:        {abs(price - bs):.4f}")


if __name__ == "__main__":
    run_demo()
