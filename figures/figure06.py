"""Run Figure 6: continuous and gapped top-space alignment rates.

`--regime continuous` or `--regime gapped` runs a single regime (default: both).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs import ALPHA, figure06_config, gapped_density, prescribed_spectrum
from src.laws import paper_phi
from src.runs import apply_overrides, parser, save_npz_atomic
from src.simulator import FiniteSimulationConfig, PrescribedSpectrumSimulator


def top_space_distance(factor: torch.Tensor, rank: int) -> float:
    """Return ||P_rank - P_rank^C||_F^2 / rank, the normalization used by src/simulator.py.

    Theorem 10 divides by d instead; figures.ipynb multiplies by rank / d.
    """
    left_vectors, _, _ = torch.linalg.svd(factor, full_matrices=False)
    top_vectors = left_vectors[:, :rank]
    target_overlap = torch.sum(top_vectors[:rank] ** 2)
    distance = 2.0 - 2.0 * target_overlap / float(rank)
    return max(0.0, float(distance.detach().cpu()))


def rho_zero_envelope(
    simulator: PrescribedSpectrumSimulator,
    singular_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the finite-d rho->0 top-projector limit from the linearized flow."""
    cfg = simulator.config
    if simulator.p != simulator.d:
        raise ValueError("The rho->0 envelope assumes tau = 1.")
    rank = int(np.floor(ALPHA * simulator.d))
    spectrum = torch.zeros(simulator.d, device=simulator.device, dtype=simulator.dtype)
    spectrum[: simulator.m_star] = torch.as_tensor(
        singular_values, device=simulator.device, dtype=simulator.dtype
    )
    left_delta = np.full((simulator.times.size, cfg.reps), np.nan, dtype=float)
    right_delta = np.full_like(left_delta, np.nan)

    for rep in range(cfg.reps):
        R = simulator._sample_init(rep, 0)
        for time_index, time in enumerate(simulator.times):
            if time <= 0.0:
                continue
            left_scale = torch.sinh(float(time) * spectrum)
            right_scale = torch.cosh(float(time) * spectrum)
            left_scale /= torch.max(left_scale)
            right_scale /= torch.max(right_scale)
            left_factor = left_scale.unsqueeze(1) * R
            right_factor = right_scale.unsqueeze(1) * R
            left_delta[time_index, rep] = top_space_distance(left_factor, rank)
            right_delta[time_index, rep] = top_space_distance(right_factor, rank)

    return left_delta, right_delta


def run_regime(
    name: str,
    regime: str,
    base_cfg: FiniteSimulationConfig,
    args,
) -> None:
    cfg = apply_overrides(replace(base_cfg), args)
    values, density_grid, density, x_alpha, epsilon_alpha = prescribed_spectrum(cfg, regime)
    density_pdf = paper_phi() if regime == "continuous" else gapped_density
    simulator = PrescribedSpectrumSimulator(cfg, values, density_pdf)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name} ===")
    print(cfg)
    result = simulator.run()

    print("computing the direct rho->0 envelope")
    rho0_left, rho0_right = rho_zero_envelope(simulator, values)
    path = args.out_dir / f"{name}.npz"
    save_npz_atomic(
        result,
        path,
        rho0_prediction_top_left_delta=rho0_left,
        rho0_prediction_top_right_delta=rho0_right,
        teacher_density_grid=density_grid,
        teacher_density=density,
        x_alpha=np.array(x_alpha),
        epsilon_alpha=np.array(epsilon_alpha),
    )
    print(f"saved {path}")


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--regime", choices=("continuous", "gapped", "both"), default="both", help="Regime to run (default: both).")
    args = p.parse_args()
    if args.regime in ("continuous", "both"):
        run_regime("figure06_continuous_rates", "continuous", figure06_config(), args)
    if args.regime in ("gapped", "both"):
        run_regime("figure06_gapped_rates", "gapped", figure06_config(), args)


if __name__ == "__main__":
    main()