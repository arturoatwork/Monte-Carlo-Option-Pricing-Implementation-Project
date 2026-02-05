import numpy as np


def discounted_payoff_call(ST: np.ndarray, K: float, r: float, T: float) -> np.ndarray:
    payoff = np.maximum(ST - K, 0.0)
    return np.exp(-r * T) * payoff


def discounted_payoff_put(ST: np.ndarray, K: float, r: float, T: float) -> np.ndarray:
    payoff = np.maximum(K - ST, 0.0)
    return np.exp(-r * T) * payoff


def estimate_mean_se_ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float, tuple[float, float]]:
    """
    Returns:
        mean, standard_error, (ci_low, ci_high) using normal approx (CLT).
        Default: 95% CI (alpha=0.05).
    """
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    z = 1.96  # for 95% CI
    ci = (mean - z * se, mean + z * se)
    return mean, se, ci
