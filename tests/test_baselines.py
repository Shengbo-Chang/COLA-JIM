import unittest

import numpy as np

from baselines import run_fixed_response, run_scan_ga, run_uniform
from cola import Config


class LoadingTests(unittest.TestCase):
    def test_uniform_loads_remain_equal_and_reach_the_bound(self):
        cfg = Config(noise_std_mm=0.0, budget_N=200.0)
        calls = []

        def acquire(forces):
            calls.append(forces.copy())
            return np.full(29, -0.001 * forces.sum())

        result = run_uniform(acquire, cfg)
        self.assertEqual(result.stop_reason, "load_bound")
        np.testing.assert_allclose(result.forces, cfg.budget_N / 18)
        self.assertAlmostEqual(result.dd_observed, -0.2)
        self.assertEqual(result.acquisitions, len(calls))
        self.assertEqual(result.acquisitions, 20 * (result.steps + 1))
        for before, after in zip(result.history, result.history[1:]):
            np.testing.assert_allclose(after.forces, after.forces[0])
            self.assertLessEqual(np.linalg.norm(after.forces - before.forces),
                                 cfg.step_N + 1e-10)

    def test_uniform_stops_at_observed_displacement_plateau(self):
        def acquire(forces):
            return np.full(29, 0.001 * min(forces.sum(), 50.0))

        result = run_uniform(acquire, Config(noise_std_mm=0.0), step_N=10.0)
        self.assertEqual(result.stop_reason, "displacement_plateau")
        self.assertEqual(result.steps, 3)
        self.assertAlmostEqual(result.dd_observed, 0.05)
        self.assertLess(result.forces.sum(), 970.0)

    def test_fixed_response_solves_known_linear_allocation(self):
        cfg = Config(noise_std_mm=0.0, budget_N=170.0)
        coefficients = np.linspace(-0.003, 0.005, 18)

        def acquire(forces):
            self.assertLessEqual(forces.sum(), cfg.budget_N + 1e-9)
            return np.full(29, 1.0 + coefficients @ forces)

        result = run_fixed_response(acquire, cfg)
        expected = np.zeros(18)
        expected[-1] = cfg.force_max_N
        expected[-2] = cfg.budget_N - cfg.force_max_N
        np.testing.assert_allclose(result.forces, expected, atol=1e-8)
        self.assertAlmostEqual(result.dd_observed, 1.0 + coefficients @ expected)
        self.assertEqual(result.state_queries, 42)
        self.assertEqual(result.acquisitions, 840)


class ScanGATests(unittest.TestCase):
    def test_seeded_search_improves_a_feasible_quadratic_objective(self):
        cfg = Config(force_max_N=80.0, budget_N=600.0)
        target = np.linspace(0.0, 60.0, 18)
        calls = []

        def mean_gap(forces):
            self.assertGreaterEqual(forces.min(), -1e-10)
            self.assertLessEqual(forces.max(), cfg.force_max_N + 1e-10)
            self.assertLessEqual(forces.sum(), cfg.budget_N + 1e-9)
            calls.append(forces.copy())
            return float(np.mean((forces - target) ** 2))

        options = dict(seed=42, population_size=24, max_generations=40)
        result = run_scan_ga(mean_gap, cfg, **options)
        self.assertEqual(result.evaluations, len(calls))
        self.assertLess(result.mean_gap_mm, 100.0)
        self.assertAlmostEqual(result.mean_gap_mm,
                               np.mean((result.forces - target) ** 2))
        values = np.array([step.mean_gap_mm for step in result.history])
        self.assertTrue(np.all(np.diff(values) <= 0))
        repeated = run_scan_ga(mean_gap, cfg, **options)
        np.testing.assert_array_equal(result.forces, repeated.forces)
        self.assertEqual(result.evaluations, repeated.evaluations)

    def test_constant_fitness_stops_after_stagnation_limit(self):
        calls = []

        def mean_gap(forces):
            calls.append(forces.copy())
            return 1.0

        result = run_scan_ga(mean_gap, population_size=10,
                             stagnation_generations=3, max_generations=50)
        self.assertEqual(result.generations, 3)
        self.assertEqual(result.stop_reason, "stagnation")
        self.assertEqual(result.evaluations, len(calls))
        self.assertEqual(result.mean_gap_mm, 1.0)

    def test_zero_budget_evaluates_only_zero_load(self):
        cfg = Config(budget_N=0.0)
        result = run_scan_ga(lambda forces: float(np.sum(forces) + 2.0), cfg)
        np.testing.assert_array_equal(result.forces, np.zeros(18))
        self.assertEqual(result.evaluations, 1)
        self.assertEqual(result.generations, 0)
        self.assertEqual(result.mean_gap_mm, 2.0)


if __name__ == "__main__":
    unittest.main()
