import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from src.gbm import simulate_gbm_terminal


class TestSimulateGBMTerminal(unittest.TestCase):

    def setUp(self):
        self.S0    = 100.0
        self.r     = 0.03
        self.sigma = 0.20
        self.T     = 1.0
        self.steps = 252
        self.n     = 5000

    def test_output_shape(self):
        ST = simulate_gbm_terminal(self.S0, self.r, self.sigma, self.T, self.steps, self.n)
        self.assertEqual(ST.shape, (self.n,))

    def test_all_positive(self):
        ST = simulate_gbm_terminal(self.S0, self.r, self.sigma, self.T, self.steps, self.n, seed=0)
        self.assertTrue(np.all(ST > 0))

    def test_reproducible_with_seed(self):
        ST1 = simulate_gbm_terminal(self.S0, self.r, self.sigma, self.T, self.steps, self.n, seed=7)
        ST2 = simulate_gbm_terminal(self.S0, self.r, self.sigma, self.T, self.steps, self.n, seed=7)
        np.testing.assert_array_equal(ST1, ST2)

    def test_antithetic_correct_count(self):
        ST = simulate_gbm_terminal(self.S0, self.r, self.sigma, self.T, self.steps, self.n,
                                   seed=0, antithetic=True)
        self.assertEqual(ST.shape, (self.n,))

    def test_risk_neutral_drift(self):
        """E[S_T] under risk-neutral measure should be approx S0 * exp(r*T)."""
        ST = simulate_gbm_terminal(self.S0, self.r, self.sigma, self.T, self.steps,
                                   n_simulations=50000, seed=1)
        expected = self.S0 * np.exp(self.r * self.T)
        self.assertAlmostEqual(ST.mean(), expected, delta=0.5)

    def test_zero_vol_terminal_price(self):
        """With sigma=0, terminal price should equal S0 * exp(r*T) exactly."""
        ST = simulate_gbm_terminal(self.S0, self.r, sigma=0.0, T=self.T,
                                   steps=self.steps, n_simulations=10, seed=0)
        expected = self.S0 * np.exp(self.r * self.T)
        np.testing.assert_allclose(ST, expected, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
