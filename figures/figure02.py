"""Run finite simulations for Figure 2, the visual abstract."""

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
        kappa=0.75,
        tau=1.0,
        kappa_star=0.55,
        rhos=(0.3, 0.1, 0.01, 1e-4, 1e-8),
        eta=1e-3,
        t_max=15.0,
        num_save=500,
        reps=10,
        top_eig_fracs=(0.05, 0.10, 0.20, 0.30),
        dtype="float64",
        promote_projectors_to_float64=True,
        store_uv_spectra=False,
        store_prediction_spectrum=True,
        store_uv_range_projectors=True,
        store_uv_top_projectors=False,
        store_prediction_projectors=True,
        store_prediction_top_projectors=False,
        save_grid="linear",
    )
    run_and_save("figure02_visual_abstract", cfg, args, phi_c=paper_phi())


if __name__ == "__main__":
    main()
