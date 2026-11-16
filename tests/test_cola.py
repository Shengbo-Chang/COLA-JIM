import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cola import (
    Config, MeasurementOracle, fit_response, limited_step, noise_allowance,
    paley_directions, probe_loads, project_forces, run_cola,
)


class RecordingPlant:
    def __init__(self, response):
        self.response = response
        self.loads = []

    def __call__(self, forces):
        self.loads.append(forces.copy())
        return np.full(29, self.response(forces), dtype=float)


class NumericalHelpersTests(unittest.TestCase):
    def test_paley_orthogonality_and_projected_probe_rank(self):
        D = paley_directions()
        self.assertEqual(D.shape, (20, 18))
        np.testing.assert_array_equal(D.T @ D, 20 * np.eye(18))
        self.assertTrue(np.isin(D, [-1, 1]).all())
        config = Config()
        for f in (np.zeros(18), np.full(18, config.budget_N / 18)):
            loads = probe_loads(f, config)
            self.assertEqual(loads.shape, (40, 18))
            self.assertGreaterEqual(loads.min(), -1e-10)
            self.assertLessEqual(loads.max(), config.force_max_N + 1e-10)
            self.assertTrue((loads.sum(axis=1) <= config.budget_N + 1e-9).all())
            self.assertTrue((np.linalg.norm(loads - f, axis=1) <= config.probe_N + 1e-9).all())
            self.assertEqual(np.linalg.matrix_rank((loads - f).T), 18)

    def test_projection_matches_known_euclidean_solution(self):
        np.testing.assert_allclose(project_forces(np.array([6., 4., -2.]), 5., 6.),
                                   [4., 2., 0.], atol=1e-12)
        np.testing.assert_array_equal(project_forces(np.array([2., -1.]), 5., 10.),
                                      [2., 0.])

    def test_response_fit(self):
        rng = np.random.default_rng(20)
        A = rng.normal(size=(29, 18))
        f = np.full(18, Config().budget_N / 18)
        D = (probe_loads(f) - f).T
        recovered = fit_response(D, A @ D)
        np.testing.assert_allclose(recovered, A, atol=1e-12, rtol=1e-12)

    def test_step_optimality(self):
        f = np.array([2.5, 0.5, 1.5, 1.0, 0.0])
        cap, budget, radius = 3.0, 6.0, 1.7
        for seed in range(4):
            with self.subTest(seed=seed):
                r = np.random.default_rng(seed).normal(size=len(f))
                d = limited_step(f, r, cap, budget, radius)
                reference = minimize(
                    lambda x: -r @ x, np.zeros_like(f), jac=lambda x: -r,
                    method="SLSQP", bounds=list(zip(-f, cap - f)),
                    constraints=[
                        {"type": "ineq", "fun": lambda x: budget - (f + x).sum(),
                         "jac": lambda x: -np.ones_like(x)},
                        {"type": "ineq", "fun": lambda x: radius**2 - x @ x,
                         "jac": lambda x: -2 * x},
                    ], options={"ftol": 1e-8, "maxiter": 1000},
                )
                self.assertTrue(reference.success, reference.message)
                self.assertGreaterEqual((f + d).min(), -1e-9)
                self.assertLessEqual((f + d).max(), cap + 1e-9)
                self.assertLessEqual((f + d).sum(), budget + 1e-9)
                self.assertLessEqual(np.linalg.norm(d), radius + 1e-9)
                self.assertAlmostEqual(float(r @ d), float(r @ reference.x), places=7)

    def test_active_total_budget_allows_redistribution(self):
        f = np.array([3., 3.])
        d = limited_step(f, np.array([1., 4.]), 5., 6., 1.)
        np.testing.assert_allclose(d, [-1 / np.sqrt(2), 1 / np.sqrt(2)], atol=1e-10)


class MeasurementAndControllerTests(unittest.TestCase):
    def assert_query_counts(self, result, plant, config, expected):
        self.assertEqual(result.query_counts, expected)
        self.assertEqual(result.state_queries, sum(expected.values()))
        self.assertEqual(result.acquisitions, len(plant.loads))
        self.assertEqual(result.acquisitions, config.averages * result.state_queries)

    def test_averaging_and_query_accounting(self):
        calls = []

        def acquire(f):
            calls.append(f.copy())
            return np.arange(29) + len(calls)

        oracle = MeasurementOracle(acquire, Config())
        first = oracle.measure(np.zeros(18), "trial")
        second = oracle.measure(np.zeros(18), "trial")
        np.testing.assert_allclose(first, np.arange(29) + 10.5)
        np.testing.assert_allclose(second, np.arange(29) + 30.5)
        self.assertEqual(oracle.state_queries, 2)
        self.assertEqual(oracle.acquisitions, 40)

    def test_linear_response_is_reused_and_every_query_is_feasible(self):
        config = Config(noise_std_mm=0., max_cycles=5)
        queries = []
        A = np.full((29, 18), .002)
        A[:, 0] = .004

        def acquire(f):
            queries.append(f.copy())
            return A @ f

        result = run_cola(acquire, config)
        self.assertEqual(result.identifications, 1)
        self.assertEqual(result.mismatch_events, 0)
        self.assertEqual(len(result.history), 5)
        self.assertEqual(result.query_counts, {"current": 5, "initial_probe": 40, "trial": 5})
        self.assertEqual(result.acquisitions, result.state_queries * 20)
        self.assertEqual(len(queries), result.acquisitions)
        self.assertTrue(all(h.step_norm_N <= 20 + 1e-9 for h in result.history))
        self.assertEqual([h.reidentified for h in result.history], [True, False, False, False, False])
        loads = np.asarray(queries)
        self.assertGreaterEqual(loads.min(), -1e-9)
        self.assertLessEqual(loads.max(), config.force_max_N + 1e-9)
        self.assertTrue((loads.sum(axis=1) <= config.budget_N + 1e-9).all())

    def test_convergence_reuses_last_accepted_measurement(self):
        config = Config(noise_std_mm=0., force_max_N=1., budget_N=1.,
                        step_N=2., probe_N=.2, max_cycles=5)
        plant = RecordingPlant(lambda f: 2. + f[0])
        result = run_cola(plant, config)
        self.assertEqual(result.stop_reason, "small_update")
        self.assertEqual(len(result.history), 1)
        self.assertEqual(result.identifications, 1)
        self.assertAlmostEqual(result.forces[0], 1.)
        np.testing.assert_allclose(result.displacements, np.full(29, 3.))
        self.assertEqual(result.dd_observed, result.history[-1].dd_observed)
        self.assert_query_counts(result, plant, config,
                                 {"current": 1, "initial_probe": 40, "trial": 1})

    def test_gain_stop_precedes_current_query(self):
        config = Config(noise_std_mm=0., force_max_N=1., budget_N=2.,
                        step_N=1., probe_N=.2, penalty_lambda=0.,
                        min_gain_mm=.001, max_cycles=5)
        benefit = np.zeros(18)
        benefit[:2] = [.01, .0001]
        plant = RecordingPlant(lambda f: benefit @ f)
        result = run_cola(plant, config)
        self.assertEqual(result.stop_reason, "small_predicted_gain")
        self.assertEqual(len(result.history), 1)
        next_step = limited_step(result.forces, benefit, config.force_max_N,
                                 config.budget_N, config.step_N)
        self.assertGreater(np.max(np.abs(next_step)), config.min_update_N)
        self.assertLessEqual(float(benefit @ next_step), config.min_gain_mm)
        self.assertEqual(result.dd_observed, result.history[-1].dd_observed)
        self.assert_query_counts(result, plant, config,
                                 {"current": 1, "initial_probe": 40, "trial": 1})

    def test_reidentification_measurements_are_counted_before_stopping(self):
        config = Config(noise_std_mm=0., probe_N=.1, penalty_lambda=0.,
                        max_cycles=5)
        plant = RecordingPlant(lambda f: min(f[0], 1.))
        result = run_cola(plant, config)
        self.assertEqual(result.stop_reason, "small_update")
        self.assertEqual(len(result.history), 1)
        self.assertEqual(result.identifications, 2)
        self.assertEqual(result.mismatch_events, 1)
        self.assertAlmostEqual(result.forces[0], 1.09375)
        np.testing.assert_allclose(result.displacements, np.ones(29))
        self.assert_query_counts(
            result, plant, config,
            {"current": 2, "initial_probe": 40, "trial": 1,
             "bisection": 7, "reidentification_probe": 40},
        )

    @staticmethod
    def piecewise_observations(f):

        value = f[0] + max(f[0] - 30.0, 0.0)
        return np.full(29, value)

    def test_bisection_returns_cached_inconsistent_upper_endpoint(self):
        calls = []

        def acquire(f):
            calls.append(f.copy())
            return self.piecewise_observations(f)

        result = run_cola(acquire, Config(noise_std_mm=0., max_cycles=2))
        self.assertEqual(result.mismatch_events, 1)
        self.assertEqual(result.identifications, 1)
        self.assertEqual(result.query_counts["bisection"], 7)
        self.assertAlmostEqual(result.history[-1].accepted_alpha, .53125)
        self.assertAlmostEqual(result.forces[0], 30.625, places=7)
        np.testing.assert_allclose(result.displacements, self.piecewise_observations(result.forces))

        self.assertFalse(np.allclose(calls[-1], result.forces))
        self.assertEqual(result.state_queries, 2 + 40 + 2 + 7)
        self.assertEqual(len(calls), result.state_queries * 20)

    def test_mismatch_triggers_reidentification_on_next_cycle(self):
        result = run_cola(self.piecewise_observations,
                          Config(noise_std_mm=0., max_cycles=3))
        self.assertEqual(result.identifications, 2)
        self.assertEqual(result.query_counts["reidentification_probe"], 40)
        self.assertEqual([h.reidentified for h in result.history], [True, False, True])
        self.assertTrue(result.history[1].mismatch)

    def test_noise_allowance_increases_with_noise_and_extrapolation(self):
        short = np.ones(18)
        nominal = noise_allowance(short, Config())
        self.assertGreater(nominal, 0)
        self.assertAlmostEqual(noise_allowance(short, Config(noise_std_mm=.04)), 2 * nominal)
        self.assertGreater(noise_allowance(20 * short, Config()), nominal)
        self.assertEqual(noise_allowance(short, Config(noise_std_mm=0.)), 0)


if __name__ == "__main__":
    unittest.main()
