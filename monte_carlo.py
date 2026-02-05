from src.gbm import simulate_gbm
from src.option_pricing import european_call_price

def run():
    # Parameters
    S0 = 100.0
    K = 100.0
    mu = 0.05
    sigma = 0.2
    r = 0.03
    T = 1.0
    steps = 252
    n_simulations = 10_000

    paths = simulate_gbm(S0, mu, sigma, T, steps, n_simulations)
    price, se = european_call_price(paths, K, r, T)

    print("Monte Carlo European Call Option")
    print(f"S0={S0}, K={K}, mu={mu}, sigma={sigma}, r={r}, T={T}")
    print(f"Price: {price:.4f}")
    print(f"Std. Error: {se:.4f}")

if __name__ == "__main__":
    run()
