import numpy as np
from typing import Optional


def simulate_gbm_terminal(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    steps: int,
    n_simulations: int,
    seed: Optional[int] = None,
    antithetic: bool = False,
) -> np.ndarray:
    """
    Simulate terminal prices S_T under risk-neutral GBM:
        S_{t+dt} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Returns:
        terminal_prices: shape (n_simulations,)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps

    if antithetic:
        half = (n_simulations + 1) // 2
        Z = np.random.standard_normal((half, steps))
        Z = np.vstack([Z, -Z])[:n_simulations, :]
    else:
        Z = np.random.standard_normal((n_simulations, steps))

    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.log(S0) + np.cumsum(increments, axis=1)
    ST = np.exp(log_paths[:, -1])

    return ST