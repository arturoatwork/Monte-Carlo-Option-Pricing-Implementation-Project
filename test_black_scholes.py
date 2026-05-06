import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import math
from src.black_scholes import black_scholes_call, black_scholes_put


class TestBlackScholesCall(unittest.TestCase):

    def test_known_atm_value(self):
        """ATM call with known parameters against a pre-computed reference."""
        price = black_scholes_call(S0=100, K=100, r=0.03, sigma=0.2, T=1.0)
        self.assertAlmostEqual(price, 9.4134, places=3)

    def test_zero_maturity_returns_intrinsic(self):
        self.assertAlmostEqual(black_scholes_call(110, 100, 0.03, 0.2, T=0), 10.0)
        self.assertAlmostEqual(black_scholes_call(90,  100, 0.03, 0.2, T=0), 0.0)

    def test_deep_itm_call(self):
        """Deep ITM call should approach S0 - K*exp(-rT)."""
        price = black_scholes_call(S0=200, K=100, r=0.03, sigma=0.2, T=1.0)
        lower = 200 - 100 * math.exp(-0.03)
        self.assertGreater(price, lower - 0.01)

    def test_deep_otm_call_near_zero(self):
        price = black_scholes_call(S0=50, K=200, r=0.03, sigma=0.2, T=1.0)
        self.assertAlmostEqual(price, 0.0, places=4)

    def test_increasing_in_sigma(self):
        c1 = black_scholes_call(100, 100, 0.03, 0.1, 1.0)
        c2 = black_scholes_call(100, 100, 0.03, 0.3, 1.0)
        self.assertLess(c1, c2)

    def test_increasing_in_T(self):
        c1 = black_scholes_call(100, 100, 0.03, 0.2, 0.5)
        c2 = black_scholes_call(100, 100, 0.03, 0.2, 2.0)
        self.assertLess(c1, c2)


class TestBlackScholesPut(unittest.TestCase):

    def test_zero_maturity_returns_intrinsic(self):
        self.assertAlmostEqual(black_scholes_put(90, 100, 0.03, 0.2, T=0), 10.0)
        self.assertAlmostEqual(black_scholes_put(110, 100, 0.03, 0.2, T=0), 0.0)

    def test_put_call_parity(self):
        """C - P = S0 - K*exp(-rT)."""
        S0, K, r, sigma, T = 100, 100, 0.03, 0.2, 1.0
        c = black_scholes_call(S0, K, r, sigma, T)
        p = black_scholes_put(S0, K, r, sigma, T)
        lhs = c - p
        rhs = S0 - K * math.exp(-r * T)
        self.assertAlmostEqual(lhs, rhs, places=6)

    def test_deep_otm_put_near_zero(self):
        price = black_scholes_put(S0=200, K=50, r=0.03, sigma=0.2, T=1.0)
        self.assertAlmostEqual(price, 0.0, places=4)


if __name__ == '__main__':
    unittest.main()
