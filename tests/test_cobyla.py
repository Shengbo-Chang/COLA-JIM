import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from baselines import FORCE_SCALE_N, run_cobyla
from cola import Config


class COBYLATests(unittest.TestCase):
    def test_repeated_projected_candidates_take_fresh_measurements(self):
        cfg = Config(noise_std_mm=0.0)
        acquired = []

        def acquire(forces):
            acquired.append(forces.copy())
            self.assertGreaterEqual(float(np.min(forces)), -1e-9)
            self.assertLessEqual(float(np.max(forces)), cfg.force_max_N + 1e-9)
            self.assertLessEqual(float(np.sum(forces)), cfg.budget_N + 1e-9)
            return np.full(29, np.sum(forces) * 1e-3)

        def fake_minimize(fun, x0, method, constraints, options):
            np.testing.assert_array_equal(x0, np.zeros(18))
            self.assertEqual(method, "COBYLA")
            self.assertEqual(options["maxiter"], 1000)
            self.assertAlmostEqual(options["tol"], 1e-4 / FORCE_SCALE_N)
            self.assertAlmostEqual(options["catol"], 1e-8 / FORCE_SCALE_N)
            self.assertNotIn("f_target", options)

            first = np.full(18, 2.0)
            second = np.full(18, 3.0)
            self.assertLess(float(np.min(constraints["fun"](first))), 0.0)
            value1 = fun(first)
            value2 = fun(second)
            self.assertAlmostEqual(value1, value2, places=10)
            return SimpleNamespace(x=second, fun=value2, success=False, status=3,
                                   message="test native outcome")

        with patch("baselines.minimize", side_effect=fake_minimize):
            result = run_cobyla(acquire, cfg)
        self.assertEqual(len(acquired), 40)
        self.assertEqual(result.state_queries, 2)
        self.assertEqual(result.acquisitions, 40)
        self.assertFalse(result.success)
        self.assertEqual(result.status, 3)
        np.testing.assert_allclose(acquired[0], acquired[20], atol=1e-9)
        self.assertAlmostEqual(result.dd_observed, cfg.budget_N * 1e-3)

    def test_deterministic_quadratic_has_known_feasible_solution(self):
        cfg = Config(noise_std_mm=0.0)
        target = np.linspace(5.0, 30.0, 18)
        curvature = 0.0005

        def acquire(forces):
            value = 1.0 - curvature * np.sum((forces - target) ** 2)
            return np.full(29, value)

        result = run_cobyla(acquire, cfg)
        expected = target - cfg.penalty_lambda / (2.0 * curvature)
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.forces, expected, atol=2e-3, rtol=0)
        self.assertEqual(result.acquisitions, 20 * result.state_queries)
        self.assertLessEqual(result.state_queries, 1000)


if __name__ == "__main__":
    unittest.main()
