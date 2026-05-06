import numpy as np
import matplotlib.pyplot as plt

from src.gbm import simulate_gbm_terminal
from src.option_pricing import discounted_payoff_call, estimate_mean_se_ci
from src.black_scholes import black_scholes_call


def run_convergence_study(
    S0: float = 100.0,
    K: float = 100.0,
    r: float = 0.03,
    sigma: float = 0.2,
    T: float = 1.0,
    steps: int = 252,
    simulation_counts=(500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000),
    seed: int = 42,
):
    """
    Convergence study comparing standard MC and antithetic variates against
    the Black-Scholes closed-form price.
    """

    bs = black_scholes_call(S0, K, r, sigma, T)

    print("Convergence Study: European Call (Risk-Neutral GBM)")
    print(f"S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}, steps={steps}")
    print(f"Black–Scholes Call Price (closed-form): {bs:.4f}\n")

    header = (
        f"{'N Sims':>10} | "
        f"{'MC Price':>10} {'SE':>8} {'AbsErr':>8} | "
        f"{'Anti Price':>10} {'SE':>8} {'AbsErr':>8}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    Ns = []
    mc_prices, mc_ses, mc_abs = [], [], []
    av_prices, av_ses, av_abs = [], [], []

    for n in simulation_counts:
        ST = simulate_gbm_terminal(S0, r, sigma, T, steps, n, seed=seed, antithetic=False)
        disc = discounted_payoff_call(ST, K, r, T)
        price, se, _ci = estimate_mean_se_ci(disc)
        abs_err = abs(price - bs)

        ST_av = simulate_gbm_terminal(S0, r, sigma, T, steps, n, seed=seed, antithetic=True)
        disc_av = discounted_payoff_call(ST_av, K, r, T)
        price_av, se_av, _ci_av = estimate_mean_se_ci(disc_av)
        abs_err_av = abs(price_av - bs)

        print(
            f"{n:>10} | "
            f"{price:>10.4f} {se:>8.4f} {abs_err:>8.4f} | "
            f"{price_av:>10.4f} {se_av:>8.4f} {abs_err_av:>8.4f}"
        )

        Ns.append(n)
        mc_prices.append(price); mc_ses.append(se); mc_abs.append(abs_err)
        av_prices.append(price_av); av_ses.append(se_av); av_abs.append(abs_err_av)

    plt.figure()
    plt.plot(Ns, mc_prices, marker="o", label="MC")
    plt.plot(Ns, av_prices, marker="o", label="Antithetic MC")
    plt.axhline(bs, linestyle="--", label="Black-Scholes")
    plt.xscale("log")
    plt.xlabel("Number of Simulations (log scale)")
    plt.ylabel("Estimated Call Price")
    plt.title("Convergence: MC vs Antithetic vs Black-Scholes")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(Ns, mc_ses, marker="o", label="MC SE")
    plt.plot(Ns, av_ses, marker="o", label="Antithetic SE")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Simulations (log scale)")
    plt.ylabel("Standard Error (log scale)")
    plt.title("Standard Error vs N")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(Ns, mc_abs, marker="o", label="|MC - BS|")
    plt.plot(Ns, av_abs, marker="o", label="|Antithetic - BS|")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Simulations (log scale)")
    plt.ylabel("Absolute Error (log scale)")
    plt.title("Absolute Error vs Black-Scholes")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_convergence_study()
