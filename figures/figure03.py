"""Run finite simulations for Figure 3, prediction convergence and learned spectra."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.laws import paper_phi
from src.runs import parser, run_many
from src.simulator import FiniteSimulationConfig


def main() -> None:
    args = parser(__doc__).parse_args()
    common = dict(
        tau=1.0,
        rhos=(0.1,),
        eta=1e-3,
        t_max=500.0,
        num_save=500,
        reps=10,
        dtype="float64",
        store_uv_spectra=True,
        store_prediction_spectrum=True,
        store_uv_range_projectors=False,
        store_uv_top_projectors=False,
        store_prediction_projectors=False,
        store_prediction_top_projectors=False,
    )
    jobs = [
        ("figure03_convergence_under", FiniteSimulationConfig(d=500, kappa=0.55, kappa_star=0.90, **common), paper_phi()),
        ("figure03_convergence_over", FiniteSimulationConfig(d=500, kappa=0.75, kappa_star=0.55, **common), paper_phi()),
    ]

    run_many(jobs, args)


if __name__ == "__main__":
    main()
