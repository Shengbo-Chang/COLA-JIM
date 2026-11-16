import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cola import project_forces
from fem import PanelModel


class PanelModelTests(unittest.TestCase):
    def assert_equilibrium(self, model, f, state):
        q, reaction = state.displacements, state.reactions
        free = np.ones(280, dtype=bool)
        free[model.boundary_indices] = False
        gap = model.gap - q
        np.testing.assert_allclose(model.K @ q + reaction, model.B @ f, atol=1e-9)
        np.testing.assert_array_equal(q[model.boundary_indices], np.zeros(20))
        self.assertGreaterEqual(q.min(), -1e-9)
        self.assertGreaterEqual(gap.min(), -1e-9)
        self.assertGreaterEqual(reaction[free].min(), -1e-9)
        np.testing.assert_allclose(gap[free] * reaction[free], 0., atol=1e-9)
        inactive = np.ones(280, dtype=bool)
        inactive[list(state.active)] = False
        np.testing.assert_allclose(reaction[inactive], 0., atol=1e-9)

    def test_dimensions_and_distinct_observation_maps(self):
        model = PanelModel.load()
        self.assertEqual(model.K.shape, (280, 280))
        self.assertEqual(model.B.shape, (280, 18))
        self.assertEqual(model.H.shape, (29, 280))
        self.assertEqual(model.H_targets.shape, (56, 280))
        np.testing.assert_array_equal(model.H_targets[:29], model.H)
        np.testing.assert_allclose(model.H_targets.sum(axis=1), 1.)
        self.assertEqual(len(model.boundary_indices), 20)
        self.assertEqual(model.targets_xy.shape, (56, 2))

    def test_zero_load_and_metric_definitions(self):
        model = PanelModel.load(noise_std_mm=0.)
        f = np.zeros(18)
        state = model.solve(f)
        self.assert_equilibrium(model, f, state)
        np.testing.assert_array_equal(state.displacements, np.zeros(280))
        metrics = model.evaluate(f)
        interior = np.delete(model.gap, model.boundary_indices)
        self.assertEqual(metrics["DD_mm"], 0.)
        self.assertEqual(metrics["DD_observed_mm"], 0.)
        self.assertAlmostEqual(metrics["MeanRG_mm"], interior.mean())
        self.assertAlmostEqual(metrics["MaxRG_mm"], model.gap.max())
        f = np.full(18, 35.)
        q = model.solve(f).displacements
        metrics = model.evaluate(f)
        self.assertAlmostEqual(metrics["DD_mm"], (model.H_targets @ q).mean())
        self.assertAlmostEqual(metrics["DD_observed_mm"], (model.H @ q).mean())
        self.assertGreater(abs(metrics["DD_mm"] - metrics["DD_observed_mm"]), 1e-5)

    def test_case_equilibrium_and_bounded_cache(self):
        rng = np.random.default_rng(19)
        for case in (1, 2, 11, 20):
            with self.subTest(case=case):
                model = PanelModel.load(case, noise_std_mm=0., cache_size=2)
                loads = [np.full(18, 970. / 18), project_forces(np.r_[np.full(8, 119.55), np.zeros(10)])]
                loads += [project_forces(rng.uniform(0, 119.55, 18)) for _ in range(3)]
                for f in loads:
                    self.assert_equilibrium(model, f, model.solve(f))
                    self.assertLessEqual(model.cache_entries, 2)
                fresh = PanelModel.load(case, noise_std_mm=0.)
                np.testing.assert_allclose(model.solve(loads[0]).displacements,
                                           fresh.solve(loads[0]).displacements, atol=1e-9)

    def test_gap_copy_preserves_matrices_and_is_independent(self):
        model = PanelModel.load(11, seed=7)
        gap = model.gap + .15
        gap[model.boundary_indices] = 0.
        scanned = model.with_gap(gap)
        for name in ("K", "B", "H", "H_targets", "boundary_indices"):
            np.testing.assert_array_equal(getattr(scanned, name), getattr(model, name))
        np.testing.assert_array_equal(scanned.gap, gap)
        self.assertFalse(np.array_equal(scanned.gap, model.gap))
        scanned.acquire(np.zeros(18))
        self.assertEqual(scanned.acquisitions, 1)
        self.assertEqual(model.acquisitions, 0)

    def test_noise_stream(self):
        first = PanelModel.load(seed=23)
        second = PanelModel.load(seed=23)
        f = np.full(18, 35.)
        a = first.acquire(f)
        np.testing.assert_array_equal(a, second.acquire(f))
        first.evaluate(f)
        b = first.acquire(f)
        np.testing.assert_array_equal(b, second.acquire(f))
        self.assertFalse(np.array_equal(a, b))
        self.assertEqual(first.acquisitions, 2)
        self.assertEqual(first.solves, 1)
        first.reset(seed=23)
        np.testing.assert_array_equal(a, first.acquire(f))
        self.assertEqual(first.acquisitions, 1)

    def test_scanned_gap_is_reproducible_and_preserves_boundaries(self):
        model = PanelModel.load()
        original_gap = model.gap.copy()
        np.testing.assert_array_equal(model.scanned(0.).gap, original_gap)
        first = model.scanned(.5, seed=23)
        second = model.scanned(.5, seed=23)
        np.testing.assert_array_equal(first.gap, second.gap)
        np.testing.assert_array_equal(model.gap, original_gap)
        np.testing.assert_array_equal(first.gap[model.boundary_indices], np.zeros(20))
        self.assertGreaterEqual(first.gap.min(), 0.)
        self.assertFalse(np.array_equal(first.gap, original_gap))
        np.testing.assert_array_equal(first.H_targets, model.H_targets)
        with self.assertRaises(ValueError):
            model.scanned(-.5)

    def test_invalid_loads_are_rejected(self):
        model = PanelModel.load()
        for f in (np.full(18, -1.), np.full(18, 120.), np.full(18, 80.), np.zeros(17)):
            with self.assertRaises(ValueError):
                model.solve(f)


if __name__ == "__main__":
    unittest.main()
