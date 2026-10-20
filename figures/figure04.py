"""Run finite simulations for Figure 4, deterministic prediction density."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.laws import paper_phi
from src.runs import parser, run_and_save
from src.simulator import FiniteSimulationConfig


def main() -> None:
    args = parser(__doc__).parse_args()
    cfg = FiniteSimulationConfig(
        d=500,
        kappa=0.55,
        tau=1.0,
        kappa_star=0.90,
        rhos=(1e-1,),
        eta=1e-3,
        t_max=50.0,
        save_times=(0.5, 1.2, 3.5, 8.0, 50.0),
        reps=20,
        dtype="float32",
        store_uv_spectra=True,
        store_prediction_spectrum=True,
        store_uv_range_projectors=False,
        store_uv_top_projectors=False,
        store_prediction_projectors=False,
        store_prediction_top_projectors=False,
    )
    run_and_save("figure04_density_comparison", cfg, args, phi_c=paper_phi())


if __name__ == "__main__":
    main()
