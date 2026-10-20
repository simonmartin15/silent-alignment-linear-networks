"""Validation 2: step-size check of Figure 6 (eta/2 and eta/4).

Reruns repetition 0 of both Figure 6 regimes, for the smallest and largest rho,
with the teacher, initialization and save times of figures/figure06.py. Run that
script first: the curves are compared with its archive.

Output: validation02_stepsize_fig6_<regime>_eta_over<k>.npz, k = 2, 4.
"""


from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs import figure06_config, gapped_density, prescribed_spectrum
from src.laws import paper_phi
from src.runs import apply_overrides, eta_tag, parser, run_simulator, save_times_of, step_variant
from src.simulator import PrescribedSpectrumSimulator


CHECK_RHOS = (1e-5, 1e-25)
CHECK_REPS = 1
FACTORS = (2, 4)


def main() -> None:
    args = parser(__doc__).parse_args()
    base = figure06_config()
    times = save_times_of(base, paper_phi())
    for regime in ("continuous", "gapped"):
        density_pdf = paper_phi() if regime == "continuous" else gapped_density
        for factor in FACTORS:
            cfg = step_variant(base, factor, times, rhos=CHECK_RHOS, reps=CHECK_REPS)
            cfg = apply_overrides(cfg, args)
            values, *_ = prescribed_spectrum(cfg, regime)
            simulator = PrescribedSpectrumSimulator(cfg, values, density_pdf)
            run_simulator(
                f"validation02_stepsize_fig6_{regime}_{eta_tag(factor)}",
                simulator,
                args,
                extra=dict(eta_factor=factor, paired_with=f"figure06_{regime}_rates.npz"),
            )


if __name__ == "__main__":
    main()
