"""Validation 5: rotated teacher, and factor versus prediction projectors.

Since R is uniform on the Stiefel manifold, the law of the dynamics does not
depend on the singular vectors of C. This run compares a diagonal teacher, as in
all figures, with a teacher with random singular vectors, and stores all
projector observables to compare the ranges of U and V with those of Z.

Output: validation05_rotation_<regime>_<basis>.npz, basis = diagonal, rotated.
"""


from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.laws import paper_phi
from src.runs import parser, run_many
from src.simulator import FiniteSimulationConfig


REGIMES = {
    "under": dict(kappa=0.55, kappa_star=0.90),
    "over": dict(kappa=0.75, kappa_star=0.55),
}
BASES = {"diagonal": False, "rotated": True}


def base_config() -> FiniteSimulationConfig:
    return FiniteSimulationConfig(
        d=200,
        kappa=0.55,
        tau=1.0,
        kappa_star=0.90,
        rhos=(1e-5,),
        eta=1e-3,
        t_max=60.0,
        num_save=150,
        save_grid="log",
        reps=4,
        top_eig_fracs=(0.1, 0.3),
        dtype="float64",
        store_uv_spectra=False,
        store_prediction_spectrum=True,
        store_uv_range_projectors=True,
        store_uv_top_projectors=True,
        store_prediction_projectors=True,
        store_prediction_top_projectors=True,
        store_imbalance=True,
    )


def main() -> None:
    args = parser(__doc__).parse_args()
    jobs = [
        (
            f"validation05_rotation_{regime}_{basis}",
            replace(base_config(), random_teacher_basis=random_basis, **params),
            paper_phi(),
        )
        for regime, params in REGIMES.items()
        for basis, random_basis in BASES.items()
    ]
    run_many(jobs, args)


if __name__ == "__main__":
    main()
