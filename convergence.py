import numpy as np
import matplotlib.pyplot as plt

from src.gbm import simulate_gbm
from src.option_pricing import european_call_price
from src.black_scholes import black_scholes_call


def run_convergence_study(
    S0: float = 100.0,
    K: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    r: float = 0.03,
    T: float = 1.0,
    steps: int = 252,
    simulation_counts=(500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000),
    seed: int = 42,
):
    """
    Runs Monte Carlo pricing for increasing simulation counts to show convergence,
    and compares estimates to the Black–Scholes closed-form call price.

    Prints a table and plots:
      1) Monte Carlo price vs number of simulations (log scale)
      2) Standard error vs number of simulations (log scale)
      3) Absolute error vs Black–Scholes vs number of simulations (log scale)
    """
    np.random.seed(seed)

    prices = []
    ses = []
    abs_errs = []

    print("Convergence Study: Monte Carlo European Call (GBM)")
    print(f"S0={S0}, K={K}, mu={mu}, sigma={sigma}, r={r}, T={T}, steps={steps}")

    # Black–Scholes comparison (risk-neutral closed form)
    bs_price = black_scholes_call(S0, K, r, sigma, T)
    print(f"\nBlack–Scholes Call Price (closed-form): {bs_price:.4f}\n")

    print("-" * 90)
    print(f"{'N Sims':>10} | {'MC Price':>10} | {'Std Error':>10} | {'Abs Err':>10} | {'95% CI':>26}")
    print("-" * 90)

    for n in simulation_counts:
        paths = simulate_gbm(S0, mu, sigma, T, steps, n)
        price, se = european_call_price(paths, K, r, T)

        ci_low = price - 1.96 * se
        ci_high = price + 1.96 * se
        abs_err = abs(price - bs_price)

        prices.append(price)
        ses.append(se)
        abs_errs.append(abs_err)

        print(f"{n:>10} | {price:>10.4f} | {se:>10.4f} | {abs_err:>10.4f} | [{ci_low:>10.4f}, {ci_high:>10.4f}]")

    # Plot 1: MC price vs N
    plt.figure()
    plt.plot(simulation_counts, prices, marker="o")
    plt.axhline(bs_price, linestyle="--")  # default color/style, dashed line
    plt.xscale("log")
    plt.xlabel("Number of Simulations (log scale)")
    plt.ylabel("Estimated Call Price")
    plt.title("Monte Carlo Convergence: Price vs # Simulations (BS dashed)")
    plt.show()

    # Plot 2: Standard error vs N
    plt.figure()
    plt.plot(simulation_counts, ses, marker="o")
    plt.xscale("log")
    plt.xlabel("Number of Simulations (log scale)")
    plt.ylabel("Standard Error")
    plt.title("Monte Carlo Convergence: Std Error vs # Simulations")
    plt.show()

    # Plot 3: Absolute error vs BS vs N
    plt.figure()
    plt.plot(simulation_counts, abs_errs, marker="o")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Simulations (log scale)")
    plt.ylabel("Absolute Error |MC - BS| (log scale)")
    plt.title("Monte Carlo Convergence: Absolute Error vs Black–Scholes")
    plt.show()


if __name__ == "__main__":
    run_convergence_study()
