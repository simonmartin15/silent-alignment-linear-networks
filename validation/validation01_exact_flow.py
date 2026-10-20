"""Validation 1: the Euler simulator against the exact gradient flow, at small d.

For both rank regimes and three output ratios tau, the simulator is compared with
two references on the same teacher and initialization: RK4 with a finer step, and
the closed form of Proposition 1.

Output: validation01_exact_flow_<regime>_tau<tau>.npz (simulator, RK4 and errors).
"""


from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.exact_flow import as_tensor, closed_form_prediction, relative_frobenius, rk4_flow
from src.laws import DEFAULT_SUPPORT, paper_phi
from src.runs import apply_overrides, parser, save_npz_atomic
from src.simulator import FiniteLinearNetworkSimulator, FiniteSimulationConfig


RK4_SUBSTEPS = 2
CLOSED_FORM_COND_MAX = 1e10
REGIMES = {
    "under": dict(kappa=0.35, kappa_star=0.90),
    "over": dict(kappa=0.75, kappa_star=0.55),
}
TAUS = (0.6, 1.0, 1.5)


def tau_tag(tau: float) -> str:
    return f"{tau:g}".replace(".", "p")


def base_config() -> FiniteSimulationConfig:
    return FiniteSimulationConfig(
        d=40,
        kappa=0.35,
        tau=1.0,
        kappa_star=0.90,
        rhos=(1e-2, 1e-6),
        eta=1e-3,
        t_max=25.0,
        num_save=80,
        save_grid="log",
        reps=2,
        top_eig_fracs=(0.1, 0.3),
        dtype="float64",
        device="cpu",
        store_uv_spectra=True,
        store_prediction_spectrum=True,
        store_uv_range_projectors=True,
        store_uv_top_projectors=True,
        store_prediction_projectors=True,
        store_prediction_top_projectors=True,
        store_imbalance=True,
    )


class RecordingSimulator(FiniteLinearNetworkSimulator):
    """Simulator that also keeps U and V at every save time.

    `run` visits repetitions, then values of rho, then save times, so the k-th
    record is (rep, rho_index, time_index) in that order.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _save_observables(self, U, V, Z, *args, **kwargs):
        # Kept as float64 CPU tensors, so that no NumPy linear algebra is needed later.
        self.records.append((as_tensor(U.detach()).clone(), as_tensor(V.detach()).clone()))
        return super()._save_observables(U, V, Z, *args, **kwargs)


def reference_outputs(sim: RecordingSimulator) -> dict[str, np.ndarray]:
    """RK4 observables and Z distances, computed with the same C and R as the simulator run."""
    cfg = sim.config
    n_rho, n_time, reps = len(cfg.rhos), len(sim.times), cfg.reps
    shape = (n_rho, n_time, reps)
    out = {
        "rk4_loss": np.full(shape, np.nan),
        "rk4_prediction_left_delta": np.full(shape, np.nan),
        "rk4_prediction_right_delta": np.full(shape, np.nan),
        "rk4_imbalance_error": np.full(shape, np.nan),
        "rk4_singular_values_z": np.full((*shape, sim.min_pd), np.nan),
        "z_error_euler": np.full(shape, np.nan),
        "z_error_closed": np.full(shape, np.nan),
        "closed_form_cond": np.full(shape, np.nan),
    }
    if len(sim.records) != n_rho * n_time * reps:
        raise RuntimeError("Unexpected number of recorded states.")

    for rep in range(reps):
        for rho_index, rho in enumerate(cfg.rhos):
            # Same teacher and initialization as in FiniteLinearNetworkSimulator.run.
            C, left, right, singulars = sim._sample_teacher(rep if cfg.same_teacher_across_rho else 1000 * rho_index + rep)
            R = sim._sample_init(rep, rho_index)
            Z_inf = sim._prediction_limit(left, right, singulars)
            p_inf_u, p_inf_v, p_signal_u, p_signal_v, top_u, top_v = sim._teacher_projectors(left, right, R, C)

            C_ref = as_tensor(C.detach())
            R_ref = as_tensor(R.detach())
            U0 = torch.zeros((sim.p, sim.m), dtype=C_ref.dtype, device=C_ref.device)
            V0 = math.sqrt(2.0 * rho) * R_ref
            print(f"RK4 reference: rep {rep + 1}/{reps}, rho {rho:g}", flush=True)
            us, vs = rk4_flow(C_ref, U0, V0, sim.save_steps, cfg.eta, substeps=RK4_SUBSTEPS)
            z_closed, cond = closed_form_prediction(C_ref, R_ref, rho, sim.times, cond_max=CLOSED_FORM_COND_MAX)

            for i_time in range(n_time):
                U_rk, V_rk = us[i_time], vs[i_time]
                Z_rk = U_rk @ V_rk.mT
                U_t = U_rk.to(device=sim.device, dtype=sim.dtype)
                V_t = V_rk.to(device=sim.device, dtype=sim.dtype)
                # Call the base method directly, so that the reference states are not recorded.
                obs = FiniteLinearNetworkSimulator._save_observables(
                    sim, U_t, V_t, U_t @ V_t.mT, C, Z_inf, rho,
                    p_inf_u, p_inf_v, p_signal_u, p_signal_v, top_u, top_v,
                )
                out["rk4_loss"][rho_index, i_time, rep] = obs["loss"]
                out["rk4_prediction_left_delta"][rho_index, i_time, rep] = obs["prediction_left_delta"]
                out["rk4_prediction_right_delta"][rho_index, i_time, rep] = obs["prediction_right_delta"]
                out["rk4_imbalance_error"][rho_index, i_time, rep] = obs["imbalance_error"]
                out["rk4_singular_values_z"][rho_index, i_time, rep] = obs["singular_values_z"]

                U_eu, V_eu = sim.records[(rep * n_rho + rho_index) * n_time + i_time]
                out["z_error_euler"][rho_index, i_time, rep] = relative_frobenius(U_eu @ V_eu.mT, Z_rk)
                out["z_error_closed"][rho_index, i_time, rep] = relative_frobenius(z_closed[i_time], Z_rk)
                out["closed_form_cond"][rho_index, i_time, rep] = float(cond[i_time])
    return out


def main() -> None:
    args = parser(__doc__).parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for regime, params in REGIMES.items():
        for tau in TAUS:
            name = f"validation01_exact_flow_{regime}_tau{tau_tag(tau)}"
            cfg = apply_overrides(replace(base_config(), tau=tau, **params), args)
            sim = RecordingSimulator(cfg, paper_phi(), DEFAULT_SUPPORT)
            print(f"\n=== {name} ===")
            print(cfg)
            result = sim.run()
            references = reference_outputs(sim)
            path = args.out_dir / f"{name}.npz"
            save_npz_atomic(
                result,
                path,
                rk4_substeps=RK4_SUBSTEPS,
                closed_form_cond_max=CLOSED_FORM_COND_MAX,
                **references,
            )
            print(f"saved {path}")


if __name__ == "__main__":
    main()
