"""Elastic panel with unilateral contact; forces in N and displacements in mm."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


@lru_cache(maxsize=1)
def _load_data() -> dict[str, np.ndarray]:
    path = Path(__file__).resolve().parent / "model" / "panel.npz"
    with np.load(path, allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}
    distance = data["targets_xy"][:, None, :] - data["nodes_xy"][None, :, :]
    weights = np.exp(-0.5 * np.sum(distance**2, axis=2) / (
        float(data["rbf_sigma_mm"])**2 + 1e-12
    ))
    weights /= weights.sum(axis=1, keepdims=True)
    weights[:29] = data["H"]
    data["H_targets"] = weights
    return {key: _readonly(value) for key, value in data.items()}


@dataclass(frozen=True)
class ContactState:
    displacements: np.ndarray
    reactions: np.ndarray
    active: tuple[int, ...]


class PanelModel:
    """Solve K q + reaction = B f with q <= gap and fixed boundary nodes."""

    def __init__(
        self, data: dict[str, np.ndarray], case: int, gap: np.ndarray, *,
        noise_std_mm: float, seed: int, cache_size: int,
        force_max_N: float, budget_N: float,
    ):
        if not np.isfinite(noise_std_mm) or noise_std_mm < 0:
            raise ValueError("noise_std_mm must be nonnegative")
        if not isinstance(cache_size, (int, np.integer)) or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        if not np.isfinite(force_max_N) or force_max_N <= 0:
            raise ValueError("force_max_N must be positive")
        if not np.isfinite(budget_N) or budget_N < 0:
            raise ValueError("budget_N must be nonnegative")
        self._data = data
        self.case = case
        self.K = data["K"]
        self.B = _readonly(data["load_map"] * data["case_gains"][case - 1])
        self.H = data["H"]
        self.H_targets = data["H_targets"]
        self.nodes_xy = data["nodes_xy"]
        self.targets_xy = data["targets_xy"]
        self.actuator_xy = data["actuator_xy"]
        self.boundary_indices = data["boundary_indices"]
        self.n_nodes, self.n_actuators = self.B.shape
        self.n_channels = self.H.shape[0]
        gap = np.asarray(gap, dtype=float)
        if gap.shape != (self.n_nodes,) or not np.isfinite(gap).all() or (gap < 0).any():
            raise ValueError("gap must contain 280 finite nonnegative values")
        if np.any(gap[self.boundary_indices] != 0):
            raise ValueError("Fixed boundary gaps must be zero")
        self.gap = _readonly(gap.copy())
        self.force_max_N = float(force_max_N)
        self.budget_N = float(budget_N)
        self.noise_std_mm = float(noise_std_mm)
        self.cache_size = int(cache_size)
        self._boundary = np.zeros(self.n_nodes, dtype=bool)
        self._boundary[self.boundary_indices] = True
        self._stiffness = csr_matrix(self.K)
        self._cache: OrderedDict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.reset(seed=seed)

    @classmethod
    def load(
        cls, case: int = 1, *, noise_std_mm: float = 0.02, seed: int = 0,
        cache_size: int = 32, force_max_N: float = 119.55, budget_N: float = 970.0,
    ) -> PanelModel:
        """Load one of the 20 bundled gap and load-response cases."""
        data = _load_data()
        if not isinstance(case, (int, np.integer)) or not 1 <= case <= len(data["gaps"]):
            raise ValueError("case must be an integer from 1 to 20")
        return cls(data, int(case), data["gaps"][case - 1], noise_std_mm=noise_std_mm,
                   seed=seed, cache_size=cache_size, force_max_N=force_max_N,
                   budget_N=budget_N)

    @property
    def gaps(self) -> np.ndarray:
        return self.gap

    @property
    def cache_entries(self) -> int:
        return len(self._cache)

    def reset(self, *, seed: int | None = None) -> None:
        """Reset the observation stream, counters and warm state."""
        self._seed = int(seed) if seed is not None else self._seed
        self._rng = np.random.default_rng(self._seed)
        self.acquisitions = 0
        self.solves = 0
        self._last_forces: np.ndarray | None = None
        self._last_state: ContactState | None = None

    def with_gap(self, gap: np.ndarray) -> PanelModel:
        """Copy the model with a new initial gap field and independent state."""
        return type(self)(
            self._data, self.case, gap, noise_std_mm=self.noise_std_mm,
            seed=self._seed, cache_size=self.cache_size,
            force_max_N=self.force_max_N, budget_N=self.budget_N,
        )

    def scanned(self, sigma_mm: float = 0.5, seed: int = 0) -> PanelModel:
        """Apply smooth gap-field mismatch with the given pre-clipping RMS in mm."""
        if not np.isfinite(sigma_mm) or sigma_mm < 0:
            raise ValueError("Gap-field RMS must be finite and nonnegative")
        if sigma_mm == 0:
            return self.with_gap(self.gap)
        rng = np.random.default_rng(seed)
        xy = self.nodes_xy
        interior = ~self._boundary
        scaled = (xy - xy.mean(axis=0)) / np.maximum(np.ptp(xy, axis=0), 1.0)
        z0, zx, zy = rng.normal(size=3)
        smooth = np.zeros(self.n_nodes)
        for _ in range(6):
            center = rng.uniform(scaled.min(axis=0), scaled.max(axis=0))
            width = rng.uniform(0.1, 0.4)
            smooth += rng.normal() * np.exp(-np.sum((scaled - center) ** 2, axis=1) / (2 * width**2))
        field = 0.35 * z0 + 0.225 * zx * scaled[:, 0] + 0.225 * zy * scaled[:, 1] + 0.20 * smooth
        rms = float(np.sqrt(np.mean(field[interior] ** 2)))
        if rms == 0:
            raise ValueError("Gap-field realization has zero RMS")
        gap = np.maximum(self.gap + sigma_mm * field / rms, 0.0)
        gap[self.boundary_indices] = 0.0
        return self.with_gap(gap)

    def _forces(self, forces: np.ndarray) -> np.ndarray:
        f = np.asarray(forces, dtype=float)
        if f.shape != (self.n_actuators,) or not np.isfinite(f).all():
            raise ValueError("forces must contain 18 finite values")
        if f.min() < -1e-8 or f.max() > self.force_max_N + 1e-8 or f.sum() > self.budget_N + 1e-8:
            raise ValueError("Forces exceed the actuator or total-load bounds")
        f = np.clip(f, 0, self.force_max_N)
        if f.sum() > self.budget_N:
            f *= self.budget_N / f.sum()
        return f

    def _affine(self, active: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        if active in self._cache:
            self._cache.move_to_end(active)
            return self._cache[active]
        constrained = np.asarray(active, dtype=int)
        free_mask = np.ones(self.n_nodes, dtype=bool)
        free_mask[constrained] = False
        free = np.flatnonzero(free_mask)
        response = np.zeros_like(self.B)
        offset = np.zeros(self.n_nodes)
        offset[constrained] = self.gap[constrained]
        if free.size:
            rhs = np.column_stack((
                self.B[free], -self.K[np.ix_(free, constrained)] @ offset[constrained],
            ))
            factor = cho_factor(self.K[np.ix_(free, free)], lower=True, check_finite=False)
            solved = cho_solve(factor, rhs, check_finite=False)
            response[free] = solved[:, :self.n_actuators]
            offset[free] = solved[:, self.n_actuators]
        self._cache[active] = response, offset
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return response, offset

    def _candidate(self, f: np.ndarray, active: tuple[int, ...]) -> np.ndarray:
        response, offset = self._affine(active)
        return response @ f + offset

    def _feasible_active_set(self, f: np.ndarray, load: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        """Primal active-set fallback with feasible blocking steps."""
        tolerance = 1e-10
        q = np.zeros(self.n_nodes) if self._last_state is None else self._last_state.displacements.copy()
        active_mask = self._boundary | (np.abs(q - self.gap) <= tolerance)
        for _ in range(10 * self.n_nodes):
            active = tuple(np.flatnonzero(active_mask))
            target = self._candidate(f, active)
            direction = target - q
            blocking = np.flatnonzero((~active_mask) & (target > self.gap + tolerance))
            if blocking.size:
                fractions = (self.gap[blocking] - q[blocking]) / direction[blocking]
                alpha = max(0.0, float(fractions.min()))
                q += alpha * direction
                selected = blocking[fractions <= alpha + 1e-12]
                q[selected] = self.gap[selected]
                active_mask[selected] = True
                continue
            q = target
            reactions = load - self._stiffness @ q
            release = np.flatnonzero(active_mask & ~self._boundary & (reactions < -tolerance))
            if not release.size:
                return q, active
            active_mask[release[np.argmin(reactions[release])]] = False
        raise RuntimeError("Contact solver did not converge")

    def solve(self, forces: np.ndarray) -> ContactState:
        """Solve equilibrium and unilateral-contact complementarity."""
        f = self._forces(forces)
        if self._last_forces is not None and np.array_equal(f, self._last_forces):
            return self._last_state
        load = self.B @ f
        active = tuple(self.boundary_indices) if self._last_state is None else self._last_state.active
        seen = set()
        tolerance = 1e-10
        for _ in range(80):
            if active in seen:
                break
            seen.add(active)
            q = self._candidate(f, active)
            reactions = load - self._stiffness @ q
            active_mask = np.zeros(self.n_nodes, dtype=bool)
            active_mask[list(active)] = True
            active_mask |= q > self.gap + tolerance
            active_mask &= ~((reactions < -tolerance) & ~self._boundary)
            active_mask |= self._boundary
            new_active = tuple(np.flatnonzero(active_mask))
            if new_active == active:
                break
            active = new_active
        else:
            q, active = self._feasible_active_set(f, load)
        reactions = load - self._stiffness @ q
        if np.any(q > self.gap + tolerance) or np.any(reactions[~self._boundary] < -tolerance):
            q, active = self._feasible_active_set(f, load)
            reactions = load - self._stiffness @ q
        state = ContactState(_readonly(q.copy()), _readonly(reactions.copy()), active)
        self._last_forces, self._last_state = f.copy(), state
        self.solves += 1
        return state

    def acquire(self, forces: np.ndarray) -> np.ndarray:
        """Acquire one 29-channel displacement vector with independent noise."""
        values = self.H @ self.solve(forces).displacements
        if self.noise_std_mm:
            values = values + self._rng.normal(0, self.noise_std_mm, self.n_channels)
        self.acquisitions += 1
        return values

    def evaluate(self, forces: np.ndarray) -> dict[str, float]:
        """Evaluate noise-free DD over 56 targets and interface-gap metrics."""
        f = self._forces(forces)
        q = self.solve(f).displacements
        gap = np.maximum(self.gap - q, 0)
        return {
            "DD_mm": float((self.H_targets @ q).mean()),
            "DD_observed_mm": float((self.H @ q).mean()),
            "MeanRG_mm": float(gap[~self._boundary].mean()),
            "MaxRG_mm": float(gap.max()),
            "total_load_N": float(f.sum()),
        }
