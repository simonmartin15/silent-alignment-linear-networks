"""Validation 3: step-size check of Figure 7 (eta, eta/2 and eta/4).

Reruns both Figure 7 panels with three stepsizes and stores the singular values
of Z, so that the activation times of the modes can be compared. The run at eta
is repeated because the figure archives do not store spectra.

Output: validation03_stepsize_fig7_d<d>_<eta tag>.npz.
"""


from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.configs import figure07_configs
from src.laws import DEFAULT_SUPPORT, paper_phi
from src.runs import apply_overrides, eta_tag, parser, run_simulator, save_times_of, step_variant
from src.simulator import FiniteLinearNetworkSimulator


FACTORS = (1, 2, 4)


def main() -> None:
    args = parser(__doc__).parse_args()
    for base in figure07_configs():
        times = save_times_of(base, paper_phi())
        for factor in FACTORS:
            cfg = step_variant(base, factor, times, store_prediction_spectrum=True, device="cpu")
            cfg = apply_overrides(cfg, args)
            simulator = FiniteLinearNetworkSimulator(cfg, paper_phi(), DEFAULT_SUPPORT)
            run_simulator(
                f"validation03_stepsize_fig7_d{base.d}_{eta_tag(factor)}",
                simulator,
                args,
                extra=dict(eta_factor=factor),
            )


if __name__ == "__main__":
    main()
