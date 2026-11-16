"""Contact-aware online load allocation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import chi2

N_ACTUATORS = 18
N_CHANNELS = 29
N_PROBE_PAIRS = 20


@dataclass(frozen=True)
class Config:
    """Load, probing, stopping, and measurement parameters."""

    force_max_N: float = 119.55
    budget_N: float = 970.0
    step_N: float = 20.0
    probe_N: float = 20.0
    eta: float = 0.05
    penalty_lambda: float = 1e-5
    noise_std_mm: float = 0.02
    averages: int = 20
    max_cycles: int = 1000
    min_update_N: float = 1e-4
    min_gain_mm: float = 1e-8
    eps_mm: float = 1e-12
    bisection_tol: float = 0.01

    def validate(self) -> None:
        positive = (
            self.force_max_N, self.step_N, self.probe_N, self.eta,
            self.min_update_N, self.eps_mm,
        )
        nonnegative = (
            self.budget_N, self.penalty_lambda, self.noise_std_mm,
            self.min_gain_mm,
        )
        if not all(np.isfinite(x) and x > 0 for x in positive):
            raise ValueError("Force, step, probe, discrepancy and update scales must be positive")
        if not all(np.isfinite(x) and x >= 0 for x in nonnegative):
            raise ValueError("Budget, penalty, noise and gain threshold must be nonnegative")
        if not np.isfinite(self.bisection_tol) or not 0 < self.bisection_tol < 1:
            raise ValueError("bisection_tol must be between zero and one")
        if not isinstance(self.averages, (int, np.integer)) or self.averages < 1:
            raise ValueError("averages must be a positive integer")
        if not isinstance(self.max_cycles, (int, np.integer)) or self.max_cycles < 1:
            raise ValueError("max_cycles must be a positive integer")


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite, nonempty one-dimensional vector")
    return result


def _feasible(forces: np.ndarray, cap: float, budget: float, tol: float = 2e-8) -> bool:
    return bool(
        np.all(forces >= -tol)
        and np.all(forces <= cap + tol)
        and forces.sum() <= budget + tol
    )


def project_forces(
    forces: np.ndarray, force_max_N: float = 119.55, budget_N: float = 970.0,
) -> np.ndarray:
    """Euclidean projection onto actuator bounds and the total-load budget."""
    z = _vector(forces, "forces")
    if not np.isfinite(force_max_N) or force_max_N <= 0:
        raise ValueError("force_max_N must be positive")
    if not np.isfinite(budget_N) or budget_N < 0:
        raise ValueError("budget_N must be nonnegative")
    if budget_N == 0:
        return np.zeros_like(z)
    clipped = np.clip(z, 0.0, force_max_N)
    if clipped.sum() <= budget_N:
        return clipped
    low, high = 0.0, float(np.max(z))
    for _ in range(80):
        middle = (low + high) / 2
        if np.clip(z - middle, 0.0, force_max_N).sum() > budget_N:
            low = middle
        else:
            high = middle
    return np.clip(z - high, 0.0, force_max_N)


def paley_directions() -> np.ndarray:
    """Return the 20 x 18 Paley-Hadamard probe directions."""
    q = 19
    residues = np.array([1, 4, 5, 6, 7, 9, 11, 16, 17])
    offsets = (np.arange(q)[None, :] - np.arange(q)[:, None]) % q
    Q = np.where(np.isin(offsets, residues), 1.0, -1.0)
    np.fill_diagonal(Q, 0.0)
    H = np.ones((20, 20))
    H[1:, 1:] = -Q - np.eye(q)
    return H[:, 1:19].copy()


def probe_loads(forces: np.ndarray, config: Config = Config()) -> np.ndarray:
    """Return 40 feasible paired probe loads with nominal L2 radius probe_N."""
    config.validate()
    f = _vector(forces, "forces")
    if f.shape != (N_ACTUATORS,):
        raise ValueError("Probing requires 18 actuator forces")
    if not _feasible(f, config.force_max_N, config.budget_N):
        raise ValueError("The probing base load must be feasible")
    directions = paley_directions() / np.sqrt(N_ACTUATORS)
    return np.asarray([
        project_forces(f + sign * config.probe_N * d,
                       config.force_max_N, config.budget_N)
        for d in directions for sign in (1.0, -1.0)
    ])


def fit_response(
    force_increments: np.ndarray, displacement_increments: np.ndarray,
) -> np.ndarray:
    """Compute DeltaU @ pinv(DeltaF), with probe samples in columns."""
    D = np.asarray(force_increments, dtype=float)
    U = np.asarray(displacement_increments, dtype=float)
    if D.ndim != 2 or U.ndim != 2 or D.shape[1] != U.shape[1] or not D.shape[1]:
        raise ValueError("Increment matrices must have the same nonzero number of columns")
    if not np.isfinite(D).all() or not np.isfinite(U).all():
        raise ValueError("Response-fit increments must be finite")
    return U @ np.linalg.pinv(D)


def limited_step(
    forces: np.ndarray, net_benefit: np.ndarray, force_max_N: float,
    budget_N: float, radius_N: float,
) -> np.ndarray:
    """Maximize r.T @ df under load bounds and an L2 step constraint."""
    f = _vector(forces, "forces")
    r = _vector(net_benefit, "net_benefit")
    if r.shape != f.shape:
        raise ValueError("forces and net_benefit must have matching shapes")
    if not np.isfinite(force_max_N) or force_max_N <= 0:
        raise ValueError("force_max_N must be positive")
    if not np.isfinite(budget_N) or budget_N < 0:
        raise ValueError("budget_N must be nonnegative")
    if not np.isfinite(radius_N) or radius_N <= 0:
        raise ValueError("radius_N must be positive")
    if not _feasible(f, force_max_N, budget_N):
        raise ValueError("The step base load must be feasible")
    scale = float(np.max(np.abs(r)))
    if scale < 1e-30:
        return np.zeros_like(f)
    c = r / scale
    lower, upper = -f, force_max_N - f
    slack = budget_N - float(f.sum())
    target = np.zeros_like(f)
    remaining = budget_N
    for j in np.argsort(-c, kind="stable"):
        if c[j] <= 0:
            break
        target[j] = min(force_max_N, max(remaining, 0.0))
        remaining -= target[j]
    unconstrained_step = target - f
    if np.linalg.norm(unconstrained_step) <= radius_N:
        return unconstrained_step

    def at(multiplier: float) -> np.ndarray:
        d = np.clip(c / multiplier, lower, upper)
        if d.sum() <= slack:
            return d
        knots = np.unique(np.maximum(
            np.r_[0.0, c - multiplier * upper, c - multiplier * lower], 0.0,
        ))
        sums = np.clip(
            (c[None, :] - knots[:, None]) / multiplier, lower, upper,
        ).sum(axis=1)
        eligible = np.flatnonzero(sums <= slack + 1e-12)
        j = int(eligible[0]) if len(eligible) else len(knots) - 1
        if j == 0:
            budget_multiplier = 0.0
        else:
            denominator = sums[j - 1] - sums[j]
            fraction = (sums[j - 1] - slack) / denominator if denominator > 0 else 1.0
            budget_multiplier = knots[j - 1] + np.clip(fraction, 0, 1) * (
                knots[j] - knots[j - 1]
            )
        budget_multiplier = np.nextafter(budget_multiplier, np.inf)
        return np.clip((c - budget_multiplier) / multiplier, lower, upper)

    low, high = 0.0, max(1.0, float(np.linalg.norm(c) / radius_N))
    while np.linalg.norm(at(high)) > radius_N:
        high *= 2
    for _ in range(60):
        middle = (low + high) / 2
        if np.linalg.norm(at(middle)) > radius_N:
            low = middle
        else:
            high = middle
    return at(high)


class MeasurementOracle:
    """Average 29-channel acquisitions and count state queries."""

    def __init__(self, acquire: Callable[[np.ndarray], np.ndarray], config: Config):
        config.validate()
        self.acquire = acquire
        self.config = config
        self.query_counts: Counter[str] = Counter()
        self._acquisitions = 0

    def measure(self, forces: np.ndarray, kind: str = "objective") -> np.ndarray:
        f = _vector(forces, "forces")
        if f.shape != (N_ACTUATORS,):
            raise ValueError("The callback interface requires 18 actuator forces")
        if not _feasible(f, self.config.force_max_N, self.config.budget_N):
            raise ValueError("Every queried load must satisfy the physical load limits")
        samples = []
        for _ in range(self.config.averages):
            y = np.asarray(self.acquire(f.copy()), dtype=float)
            if y.shape != (N_CHANNELS,) or not np.isfinite(y).all():
                raise ValueError("Each acquisition must return 29 finite displacement values")
            samples.append(y.copy())
            self._acquisitions += 1
        self.query_counts[kind] += 1
        return np.mean(samples, axis=0)

    @property
    def state_queries(self) -> int:
        return sum(self.query_counts.values())

    @property
    def acquisitions(self) -> int:
        return self._acquisitions


def noise_allowance(increment: np.ndarray, config: Config) -> float:
    """Chi-square mismatch allowance in mm."""
    sigma = config.noise_std_mm
    if sigma == 0:
        return 0.0
    sigma_A = sigma / (config.probe_N * np.sqrt(2 * config.averages))
    variance = 2 * sigma**2 / config.averages + sigma_A**2 * float(
        np.dot(increment, increment)
    )
    return float(np.sqrt(chi2.ppf(0.95, N_CHANNELS) * variance))


def _inconsistent(
    measured_increment: np.ndarray, predicted_increment: np.ndarray,
    force_increment: np.ndarray, config: Config,
) -> bool:
    residual = np.linalg.norm(measured_increment - predicted_increment)
    predicted_norm = np.linalg.norm(predicted_increment)
    if config.noise_std_mm == 0:
        return bool(residual > config.eta * (predicted_norm + config.eps_mm))
    return bool(residual > config.eta * predicted_norm + noise_allowance(force_increment, config))


@dataclass
class Iteration:
    cycle: int
    forces: np.ndarray
    dd_observed: float
    predicted_gain_mm: float
    accepted_alpha: float
    step_norm_N: float
    reidentified: bool
    mismatch: bool


@dataclass
class Result:
    forces: np.ndarray
    displacements: np.ndarray
    dd_observed: float
    stop_reason: str
    cycles: int
    identifications: int
    mismatch_events: int
    query_counts: dict[str, int]
    state_queries: int
    acquisitions: int
    history: list[Iteration]


def run_cola(
    acquire: Callable[[np.ndarray], np.ndarray], config: Config = Config(),
    initial_forces: np.ndarray | None = None,
) -> Result:
    """Allocate loads using displacement feedback and response re-identification."""
    config.validate()
    f = np.zeros(N_ACTUATORS) if initial_forces is None else _vector(
        initial_forces, "initial_forces",
    ).copy()
    if f.shape != (N_ACTUATORS,) or not _feasible(f, config.force_max_N, config.budget_N):
        raise ValueError("initial_forces must contain 18 feasible actuator forces")
    oracle = MeasurementOracle(acquire, config)
    A = None
    identifications = 0
    mismatch_events = 0
    history: list[Iteration] = []
    stop_reason = "max_cycles"
    for cycle in range(config.max_cycles):
        reidentified = A is None
        if reidentified:
            y = oracle.measure(f, "current")
            loads = probe_loads(f, config)
            kind = "initial_probe" if identifications == 0 else "reidentification_probe"
            observed = np.column_stack([oracle.measure(p, kind) - y for p in loads])
            A = fit_response((loads - f).T, observed)
            identifications += 1
        r = A.mean(axis=0) - config.penalty_lambda
        df = limited_step(f, r, config.force_max_N, config.budget_N, config.step_N)
        gain = float(r @ df)
        if np.max(np.abs(df)) < config.min_update_N:
            stop_reason = "small_update"
            break
        if gain <= config.min_gain_mm:
            stop_reason = "small_predicted_gain"
            break
        if not reidentified:
            y = oracle.measure(f, "current")
        predicted = A @ df
        trial_y = oracle.measure(f + df, "trial")
        mismatch = _inconsistent(trial_y - y, predicted, df, config)
        alpha = 1.0
        accepted_y = trial_y
        if mismatch:
            mismatch_events += 1
            low, high = 0.0, 1.0
            upper_y = trial_y
            while high - low >= config.bisection_tol:
                middle = (low + high) / 2
                middle_y = oracle.measure(f + middle * df, "bisection")
                if _inconsistent(middle_y - y, middle * predicted, middle * df, config):
                    high = middle
                    upper_y = middle_y
                else:
                    low = middle
            alpha, accepted_y = high, upper_y
            A = None
        f = f + alpha * df
        y = accepted_y
        history.append(Iteration(
            cycle=cycle + 1, forces=f.copy(), dd_observed=float(y.mean()),
            predicted_gain_mm=gain, accepted_alpha=alpha,
            step_norm_N=float(np.linalg.norm(alpha * df)),
            reidentified=reidentified, mismatch=mismatch,
        ))
    return Result(
        forces=f.copy(), displacements=y.copy(), dd_observed=float(y.mean()),
        stop_reason=stop_reason, cycles=cycle + 1, identifications=identifications,
        mismatch_events=mismatch_events, query_counts=dict(oracle.query_counts),
        state_queries=oracle.state_queries, acquisitions=oracle.acquisitions,
        history=history,
    )
