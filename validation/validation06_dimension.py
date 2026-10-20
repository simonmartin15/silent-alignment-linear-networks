"""Validation 6: convergence of finite-d simulations to the high-dimensional theory.

Simulates the Figure 4 setting at a few fixed times, for d = 100 to 800 and
tau = 0.7, 1, 1.4, to compare with Theorems 4 and 6 at the same rho. The singular
values of C are deterministic quantiles of phi_C, so only R is random.

Output: validation06_dimension_tau<tau>_d<d>.npz.
"""


from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.laws import paper_phi, quantile_spectrum
from src.runs import apply_overrides, parser, run_simulator
from src.simulator import FiniteSimulationConfig, PrescribedSpectrumSimulator


DIMENSIONS = (100, 200, 400, 800)
TAUS = (0.7, 1.0, 1.4)
REPS = 16
SAVE_TIMES = (0.0, 0.5, 1.2, 3.5, 8.0)


def tau_tag(tau: float) -> str:
    return f"{tau:g}".replace(".", "p")


def base_config() -> FiniteSimulationConfig:
    return FiniteSimulationConfig(
        d=100,
        kappa=0.55,
        tau=1.0,
        kappa_star=0.90,
        rhos=(1e-1,),
        eta=1e-3,
        t_max=SAVE_TIMES[-1],
        save_times=SAVE_TIMES,
        reps=REPS,
        dtype="float64",
        store_uv_spectra=True,
        store_prediction_spectrum=True,
        store_uv_range_projectors=False,
        store_uv_top_projectors=False,
        store_prediction_projectors=True,
        store_prediction_top_projectors=False,
        store_imbalance=False,
    )


def teacher_values(cfg: FiniteSimulationConfig) -> np.ndarray:
    """Deterministic quantiles of phi_C, one per non-zero singular value of C (same m* as src/simulator.py)."""
    p = int(round(cfg.tau * cfg.d))
    min_pd = min(p, int(cfg.d))
    m_star = min(int(round(cfg.kappa_star * min_pd)), min_pd)
    values, *_ = quantile_spectrum(m_star, paper_phi())
    return values


def main() -> None:
    args = parser(__doc__).parse_args()
    for tau in TAUS:
        for d in DIMENSIONS:
            cfg = apply_overrides(replace(base_config(), d=d, tau=tau), args)
            simulator = PrescribedSpectrumSimulator(cfg, teacher_values(cfg), paper_phi())
            run_simulator(f"validation06_dimension_tau{tau_tag(tau)}_d{d}", simulator, args)


if __name__ == "__main__":
    main()