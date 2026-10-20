"""Validation 4: step-size check of Figure 8 (eta/2 and eta/4).

Reruns the first repetitions of Figure 8 with the seeds and save times of
figures/figure08.py, up to the end of its lower panels.

Output: validation04_stepsize_fig8_eta_over<k>.npz, k = 2, 4.
"""


from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs import figure08_config
from src.laws import DEFAULT_SUPPORT, paper_phi
from src.runs import apply_overrides, eta_tag, parser, run_simulator, save_times_of, step_variant
from src.simulator import FiniteLinearNetworkSimulator


CHECK_REPS = 2
FACTORS = (2, 4)
T_CUT = 65.0


def main() -> None:
    args = parser(__doc__).parse_args()
    base = figure08_config()
    times = save_times_of(base, paper_phi())
    for factor in FACTORS:
        # Only the loss and the singular values of Z are compared.
        cfg = step_variant(base, factor, times, t_cut=T_CUT, reps=CHECK_REPS, store_uv_spectra=False)
        cfg = apply_overrides(cfg, args)
        simulator = FiniteLinearNetworkSimulator(cfg, paper_phi(), DEFAULT_SUPPORT)
        run_simulator(
            f"validation04_stepsize_fig8_{eta_tag(factor)}",
            simulator,
            args,
            extra=dict(eta_factor=factor, paired_with="figure08_spectral_measures.npz"),
        )


if __name__ == "__main__":
    main()
