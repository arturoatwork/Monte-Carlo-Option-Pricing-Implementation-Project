import math
from scipy.stats import norm


def black_scholes_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return max(S0 - K, 0.0)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))


def black_scholes_put(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return max(K - S0, 0.0)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1))
