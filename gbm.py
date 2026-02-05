import numpy as np

def simulate_gbm(S0: float, mu: float, sigma: float, T: float, steps: int, n_simulations: int) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion price paths.

    Returns:
        paths: shape (n_simulations, steps+1)
    """
    dt = T / steps
    paths = np.zeros((n_simulations, steps + 1), dtype=float)
    paths[:, 0] = S0

    for t in range(1, steps + 1):
        Z = np.random.standard_normal(n_simulations)
        paths[:, t] = paths[:, t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )
    return paths
