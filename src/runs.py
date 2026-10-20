"""
Command-line options and saving of the simulation archives.

Every figure script and validation script uses `parser` and `apply_overrides`
for its options, and saves its results as compressed `.npz` archives in
`data/` (paths.DATA_DIR). Archives are written under a temporary name and
renamed at the end, so an interrupted run never leaves an incomplete archive.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import numpy as np

from .laws import DEFAULT_SUPPORT, ArrayLike
from .paths import DATA_DIR
from .simulator import FiniteLinearNetworkSimulator, FiniteSimulationConfig, FiniteSimulationResult, run_simulation


# ----------------------------------------------------------------------------
# Command-line options
# ----------------------------------------------------------------------------

def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--out-dir", type=Path, default=DATA_DIR, help="Directory where .npz files are saved (default: data/).")
    p.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default=None,
        help="Override torch dtype (float64 is used anyway when projector distances are stored).",
    )
    p.add_argument("--device", default=None, help="Override torch device, e.g. cpu, cuda, cuda:0.")
    p.add_argument("--seed", type=int, default=None, help="Override random seed.")
    p.add_argument("--reps", type=int, default=None, help="Override the number of repetitions.")
    p.add_argument("--t-max", type=float, default=None, help="Override the final simulation time.")
    p.add_argument("--save-grid", choices=("log", "linear"), default=None, help="Override automatic save-time spacing.")
    p.add_argument("--quick", action="store_true", help="Small smoke-test configuration.")
    p.add_argument("--no-progress", action="store_true", help="Disable progress prints.")
    return p


def apply_overrides(cfg: FiniteSimulationConfig, args: argparse.Namespace) -> FiniteSimulationConfig:
    updates = {"progress": not args.no_progress}
    if args.dtype is not None:
        updates["dtype"] = args.dtype
    if args.device is not None:
        updates["device"] = args.device
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.reps is not None:
        updates["reps"] = args.reps
    if args.t_max is not None:
        updates["t_max"] = args.t_max
        updates["save_times"] = None
    if args.save_grid is not None:
        updates["save_grid"] = args.save_grid
    cfg = replace(cfg, **updates)
    if args.quick:
        cfg = replace(
            cfg,
            d=min(cfg.d, 80),
            reps=min(cfg.reps, 1),
            rhos=cfg.rhos[:1],
            eta=max(cfg.eta, 2e-2),
            t_max=min(cfg.t_max, 1.0),
            num_save=min(cfg.num_save, 8),
            save_times=None,
            progress=not args.no_progress,
        )
    return cfg


# ----------------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------------

def augment_npz(path: Path, **arrays: Any) -> None:
    """Add arrays to an existing `.npz` archive."""
    with np.load(path, allow_pickle=False) as stored:
        payload = {key: stored[key] for key in stored.files}
    payload.update({key: np.asarray(value) for key, value in arrays.items()})
    np.savez_compressed(path, **payload)


def save_npz_atomic(result: FiniteSimulationResult, path: Path, **extra: Any) -> None:
    """Save `result` plus the `extra` arrays to `path`, which only appears once complete.

    The archive is written as `<name>.partial.npz` and renamed at the end, so an
    interrupted run never leaves an archive without its extra arrays at `path`
    (an older archive at `path` is kept until the new one replaces it).
    """
    path = Path(path)
    partial = path.with_name(f"{path.stem}.partial.npz")
    result.save_npz(partial)
    if extra:
        augment_npz(partial, **extra)
    partial.replace(path)


# ----------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------

def run_and_save(
    name: str,
    cfg: FiniteSimulationConfig,
    args: argparse.Namespace,
    *,
    phi_c: Callable[[ArrayLike], np.ndarray],
) -> Path:
    """Apply the command-line overrides to `cfg`, run it and save `<out-dir>/<name>.npz`."""
    cfg = apply_overrides(cfg, args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===")
    print(cfg)
    result = run_simulation(cfg, phi_c=phi_c)
    path = args.out_dir / f"{name}.npz"
    save_npz_atomic(result, path)
    print(f"saved {path}")
    return path


def run_many(
    jobs: Iterable[tuple[str, FiniteSimulationConfig, Callable[[ArrayLike], np.ndarray]]],
    args: argparse.Namespace,
) -> None:
    for name, cfg, phi in jobs:
        run_and_save(name, cfg, args, phi_c=phi)


def run_simulator(
    name: str,
    simulator: FiniteLinearNetworkSimulator,
    args: argparse.Namespace,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Run an already built simulator and save `<out-dir>/<name>.npz` with the `extra` arrays."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===")
    print(simulator.config)
    result = simulator.run()
    path = args.out_dir / f"{name}.npz"
    save_npz_atomic(result, path, **(extra or {}))
    print(f"saved {path}")
    return path


# ----------------------------------------------------------------------------
# Step-size checks
# ----------------------------------------------------------------------------

def eta_tag(factor: int) -> str:
    """File-name tag of a step-size variant: 'eta', 'eta_over2', 'eta_over4'."""
    return "eta" if int(factor) == 1 else f"eta_over{int(factor)}"


def save_times_of(
    cfg: FiniteSimulationConfig,
    phi_c: Callable[[ArrayLike], np.ndarray],
) -> np.ndarray:
    """Save times produced by `cfg` in the simulator.

    With a log save grid these depend on eta, so a run with a smaller stepsize
    must be given these times explicitly to be compared time by time.
    """
    return FiniteLinearNetworkSimulator(cfg, phi_c, DEFAULT_SUPPORT).times.copy()


def step_variant(
    cfg: FiniteSimulationConfig,
    factor: int,
    times: Sequence[float],
    *,
    t_cut: Optional[float] = None,
    reps: Optional[int] = None,
    rhos: Optional[Sequence[float]] = None,
    **updates: Any,
) -> FiniteSimulationConfig:
    """Copy of `cfg` with stepsize eta / factor and the save times `times` (cut at `t_cut`).

    The seeds are unchanged, so repetition r of the variant uses the same teacher
    and initialization as repetition r of `cfg`.
    """
    times = np.asarray(times, dtype=float)
    if t_cut is not None:
        times = times[times <= float(t_cut) + 1e-12]
    if times.size == 0:
        raise ValueError("No save time left below t_cut.")
    new = dict(
        eta=cfg.eta / int(factor),
        t_max=float(times[-1]),
        save_times=tuple(float(t) for t in times),
    )
    if reps is not None:
        new["reps"] = int(reps)
    if rhos is not None:
        new["rhos"] = tuple(float(r) for r in rhos)
    new.update(updates)
    return replace(cfg, **new)