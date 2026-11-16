"""Uniform, fixed-response, Scan-GA, and COBYLA load allocation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from cola import (
    Config, MeasurementOracle, N_ACTUATORS, fit_response, probe_loads,
    project_forces,
)


FORCE_SCALE_N = 119.55


@dataclass
class LoadStep:
    step: int
    forces: np.ndarray
    dd_observed: float


@dataclass
class LoadingResult:
    forces: np.ndarray
    dd_observed: float
    steps: int
    state_queries: int
    acquisitions: int
    history: list[LoadStep]
    stop_reason: str


@dataclass
class GAGeneration:
    generation: int
    mean_gap_mm: float
    evaluations: int


@dataclass
class ScanGAResult:
    forces: np.ndarray
    mean_gap_mm: float
    generations: int
    evaluations: int
    history: list[GAGeneration]
    stop_reason: str


def run_uniform(
    acquire: Callable[[np.ndarray], np.ndarray],
    config: Config | None = None,
    *,
    step_N: float | None = None,
    max_steps: int | None = None,
    displacement_tol_mm: float = 1e-8,
) -> LoadingResult:
    """Increase equal loads to an observed-DD plateau using full-vector L2 steps."""
    cfg = Config() if config is None else config
    step = cfg.step_N if step_N is None else step_N
    limit = cfg.max_cycles if max_steps is None else max_steps
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step_N must be positive")
    if not isinstance(limit, (int, np.integer)) or limit < 1:
        raise ValueError("max_steps must be a positive integer")
    if not np.isfinite(displacement_tol_mm) or displacement_tol_mm < 0:
        raise ValueError("displacement_tol_mm must be nonnegative")
    oracle = MeasurementOracle(acquire, cfg)
    forces = np.zeros(N_ACTUATORS)
    dd = float(oracle.measure(forces).mean())
    history = [LoadStep(0, forces.copy(), dd)]
    per_actuator_limit = min(cfg.force_max_N, cfg.budget_N / N_ACTUATORS)
    stop_reason = "load_bound" if per_actuator_limit == 0 else "max_steps"
    for index in range(1, limit + 1):
        if forces[0] >= per_actuator_limit:
            break
        level = min(per_actuator_limit, forces[0] + step / np.sqrt(N_ACTUATORS))
        forces = np.full(N_ACTUATORS, level)
        new_dd = float(oracle.measure(forces).mean())
        difference = abs(new_dd - dd)
        dd = new_dd
        history.append(LoadStep(index, forces.copy(), dd))
        if level >= per_actuator_limit:
            stop_reason = "load_bound"
            break
        if difference <= displacement_tol_mm:
            stop_reason = "displacement_plateau"
            break
    return LoadingResult(
        forces=forces.copy(), dd_observed=dd, steps=len(history) - 1,
        state_queries=oracle.state_queries, acquisitions=oracle.acquisitions,
        history=history, stop_reason=stop_reason,
    )


def run_fixed_response(
    acquire: Callable[[np.ndarray], np.ndarray],
    config: Config | None = None,
) -> LoadingResult:
    """Identify the initial response and apply its linear optimum once."""
    cfg = Config() if config is None else config
    oracle = MeasurementOracle(acquire, cfg)
    zero = np.zeros(N_ACTUATORS)
    initial = oracle.measure(zero, "current")
    history = [LoadStep(0, zero.copy(), float(initial.mean()))]
    loads = probe_loads(zero, cfg)
    changes = np.column_stack([
        oracle.measure(forces, "initial_probe") - initial for forces in loads
    ])
    response = fit_response(loads.T, changes)
    benefit = response.mean(axis=0) - cfg.penalty_lambda
    forces = np.zeros(N_ACTUATORS)
    remaining = cfg.budget_N
    for actuator in np.argsort(-benefit, kind="stable"):
        if benefit[actuator] <= 0 or remaining <= 0:
            break
        forces[actuator] = min(cfg.force_max_N, remaining)
        remaining -= forces[actuator]
    dd = float(oracle.measure(forces, "terminal").mean())
    history.append(LoadStep(1, forces.copy(), dd))
    return LoadingResult(
        forces=forces.copy(), dd_observed=dd, steps=1,
        state_queries=oracle.state_queries, acquisitions=oracle.acquisitions,
        history=history, stop_reason="fixed_plan",
    )


def run_scan_ga(
    mean_gap: Callable[[np.ndarray], float],
    config: Config | None = None,
    *,
    seed: int = 0,
    population_size: int = 80,
    max_generations: int = 1000,
    stagnation_generations: int = 20,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.2,
    improvement_tol_mm: float = 1e-8,
) -> ScanGAResult:
    """Minimize scanned MeanRG (mm); mutation_rate is per component."""
    cfg = Config() if config is None else config
    cfg.validate()
    for value, name, minimum in (
        (population_size, "population_size", 4),
        (max_generations, "max_generations", 1),
        (stagnation_generations, "stagnation_generations", 1),
    ):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if not all(np.isfinite(rate) and 0 <= rate <= 1
               for rate in (crossover_rate, mutation_rate)):
        raise ValueError("Crossover and mutation rates must be between zero and one")
    if not np.isfinite(improvement_tol_mm) or improvement_tol_mm < 0:
        raise ValueError("improvement_tol_mm must be nonnegative")
    rng = np.random.default_rng(seed)
    evaluations = 0

    def evaluate(forces: np.ndarray) -> float:
        nonlocal evaluations
        value = mean_gap(forces.copy())
        if np.ndim(value) != 0 or not np.isfinite(value):
            raise ValueError("mean_gap must return one finite scalar in mm")
        evaluations += 1
        return float(value)

    def feasible(forces: np.ndarray) -> np.ndarray:
        return project_forces(forces, cfg.force_max_N, cfg.budget_N)

    if cfg.budget_N == 0:
        forces = np.zeros(N_ACTUATORS)
        value = evaluate(forces)
        return ScanGAResult(forces, value, 0, evaluations,
                            [GAGeneration(0, value, evaluations)], "load_bound")
    population = rng.uniform(0, cfg.force_max_N, (population_size, N_ACTUATORS))
    targets = rng.uniform(0, min(cfg.budget_N, N_ACTUATORS * cfg.force_max_N),
                          population_size)
    population *= (targets / population.sum(axis=1))[:, None]
    population = np.array([feasible(forces) for forces in population])
    population[0] = 0.0
    population[1] = min(cfg.force_max_N, cfg.budget_N / N_ACTUATORS)
    scores = np.array([evaluate(forces) for forces in population])
    best = int(np.argmin(scores))
    best_value, best_forces = float(scores[best]), population[best].copy()
    history = [GAGeneration(0, best_value, evaluations)]
    elite_count = max(1, int(0.1 * population_size))
    stale = 0
    stop_reason = "max_generations"
    for generation in range(1, max_generations + 1):
        elite = np.argsort(scores, kind="stable")[:elite_count]
        children = []
        while len(children) < population_size - elite_count:
            candidates = rng.integers(population_size, size=(2, 3))
            parents = [population[group[np.argmin(scores[group])]] for group in candidates]
            child = parents[0].copy()
            if rng.random() < crossover_rate:
                alpha = rng.random(N_ACTUATORS)
                child = alpha * parents[0] + (1 - alpha) * parents[1]
            mutate = rng.random(N_ACTUATORS) < mutation_rate
            child[mutate] += rng.normal(0, 0.08 * cfg.force_max_N, int(mutate.sum()))
            children.append(feasible(child))
        child_scores = np.array([evaluate(child) for child in children])
        population = np.vstack((population[elite], children))
        scores = np.r_[scores[elite], child_scores]
        best = int(np.argmin(scores))
        value = float(scores[best])
        stale = 0 if best_value - value > improvement_tol_mm else stale + 1
        best_value, best_forces = value, population[best].copy()
        history.append(GAGeneration(generation, best_value, evaluations))
        if stale >= stagnation_generations:
            stop_reason = "stagnation"
            break
    return ScanGAResult(
        forces=best_forces, mean_gap_mm=best_value, generations=generation,
        evaluations=evaluations, history=history, stop_reason=stop_reason,
    )


@dataclass
class COBYLAResult:
    """Final load and optimizer result."""

    forces: np.ndarray
    optimizer_variables: np.ndarray
    dd_observed: float
    objective_mm: float
    success: bool
    status: int
    message: str
    state_queries: int
    acquisitions: int


def run_cobyla(
    acquire: Callable[[np.ndarray], np.ndarray],
    config: Config | None = None,
    *,
    max_evaluations: int = 1000,
) -> COBYLAResult:
    """Optimize averaged displacement using projected loads."""
    cfg = Config() if config is None else config
    if not isinstance(max_evaluations, int) or max_evaluations < N_ACTUATORS + 2:
        raise ValueError("max_evaluations must be an integer of at least 20")
    oracle = MeasurementOracle(acquire, cfg)

    def applied_load(x: np.ndarray) -> np.ndarray:
        return project_forces(
            FORCE_SCALE_N * np.asarray(x, dtype=float),
            force_max_N=cfg.force_max_N,
            budget_N=cfg.budget_N,
        )

    def objective(x: np.ndarray) -> float:
        forces = applied_load(x)
        displacements = oracle.measure(forces, kind="objective")
        return float(-np.mean(displacements) + cfg.penalty_lambda * np.sum(forces))

    def constraints(x: np.ndarray) -> np.ndarray:
        # Evaluate constraints in native optimizer coordinates.
        return np.r_[
            x,
            cfg.force_max_N / FORCE_SCALE_N - x,
            cfg.budget_N / FORCE_SCALE_N - np.sum(x),
        ]

    native = minimize(
        objective,
        np.zeros(N_ACTUATORS),
        method="COBYLA",
        constraints={"type": "ineq", "fun": constraints},
        options={
            "rhobeg": 0.1,
            "tol": 1e-4 / FORCE_SCALE_N,
            "catol": 1e-8 / FORCE_SCALE_N,
            "maxiter": max_evaluations,
            "disp": 0,
        },
    )
    forces = applied_load(native.x)
    objective_mm = float(native.fun)
    return COBYLAResult(
        forces=forces,
        optimizer_variables=np.asarray(native.x, dtype=float).copy(),
        dd_observed=float(-objective_mm + cfg.penalty_lambda * np.sum(forces)),
        objective_mm=objective_mm,
        success=bool(native.success),
        status=int(native.status),
        message=str(native.message),
        state_queries=oracle.state_queries,
        acquisitions=oracle.acquisitions,
    )
