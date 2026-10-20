"""Run finite simulations for Figure 5, range-projector alignment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.laws import edge_power_phi, paper_phi
from src.runs import parser, run_many
from src.simulator import FiniteSimulationConfig


def main() -> None:
    args = parser(__doc__).parse_args()
    common = dict(
        tau=1.0,
        rhos=(1e-5,),
        eta=1e-3,
        t_max=200.0,
        num_save=500,
        reps=10,
        dtype="float64",
        promote_projectors_to_float64=True,
        store_uv_spectra=False,
        store_prediction_spectrum=False,
        store_uv_range_projectors=True,
        store_uv_top_projectors=False,
        store_prediction_projectors=True,
        store_prediction_top_projectors=False,
    )
    jobs = [
        ("figure05_under", FiniteSimulationConfig(d=500, kappa=0.55, kappa_star=0.90, **common), paper_phi()),
        ("figure05_over_exponent_l0", FiniteSimulationConfig(d=500, kappa=0.75, kappa_star=0.55, **common), edge_power_phi(0)),
        ("figure05_over_exponent_l1", FiniteSimulationConfig(d=500, kappa=0.75, kappa_star=0.55, **common), edge_power_phi(1)),

    ]
    run_many(jobs, args)


if __name__ == "__main__":
    main()
