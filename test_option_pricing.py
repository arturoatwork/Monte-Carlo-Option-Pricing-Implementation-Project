import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from src.option_pricing import (
    discounted_payoff_call,
    discounted_payoff_put,
    estimate_mean_se_ci,
)


class TestDiscountedPayoffs(unittest.TestCase):

    def test_call_otm_zero(self):
        ST = np.array([80.0, 90.0, 95.0])
        payoffs = discounted_payoff_call(ST, K=100, r=0.03, T=1.0)
        np.testing.assert_array_equal(payoffs, 0.0)

    def test_call_itm_positive(self):
        ST = np.array([110.0, 120.0])
        payoffs = discounted_payoff_call(ST, K=100, r=0.0, T=1.0)
        np.testing.assert_allclose(payoffs, [10.0, 20.0])

    def test_call_discounting(self):
        ST = np.array([110.0])
        r, T = 0.05, 2.0
        payoff = discounted_payoff_call(ST, K=100, r=r, T=T)
        self.assertAlmostEqual(float(payoff[0]), 10 * np.exp(-r * T), places=8)

    def test_put_itm_positive(self):
        ST = np.array([80.0, 90.0])
        payoffs = discounted_payoff_put(ST, K=100, r=0.0, T=1.0)
        np.testing.assert_allclose(payoffs, [20.0, 10.0])

    def test_put_otm_zero(self):
        ST = np.array([110.0, 120.0])
        payoffs = discounted_payoff_put(ST, K=100, r=0.03, T=1.0)
        np.testing.assert_array_equal(payoffs, 0.0)

    def test_call_put_nonnegative(self):
        rng = np.random.default_rng(0)
        ST = rng.uniform(50, 150, 1000)
        self.assertTrue(np.all(discounted_payoff_call(ST, 100, 0.03, 1.0) >= 0))
        self.assertTrue(np.all(discounted_payoff_put(ST, 100, 0.03, 1.0) >= 0))


class TestEstimateMeanSECI(unittest.TestCase):

    def test_returns_three_values(self):
        result = estimate_mean_se_ci(np.ones(100))
        self.assertEqual(len(result), 3)

    def test_known_mean(self):
        x = np.full(1000, 5.0)
        mean, se, ci = estimate_mean_se_ci(x)
        self.assertAlmostEqual(mean, 5.0)

    def test_zero_variance_zero_se(self):
        x = np.full(100, 3.0)
        _, se, _ = estimate_mean_se_ci(x)
        self.assertAlmostEqual(se, 0.0)

    def test_ci_contains_mean(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(500)
        mean, _, (lo, hi) = estimate_mean_se_ci(x)
        self.assertGreater(mean, lo)
        self.assertLess(mean, hi)

    def test_wider_ci_with_more_variance(self):
        rng = np.random.default_rng(0)
        x_tight = rng.standard_normal(500) * 0.01
        x_wide  = rng.standard_normal(500) * 1.0
        _, _, (lo1, hi1) = estimate_mean_se_ci(x_tight)
        _, _, (lo2, hi2) = estimate_mean_se_ci(x_wide)
        self.assertGreater((hi2 - lo2), (hi1 - lo1))


if __name__ == '__main__':
    unittest.main()
