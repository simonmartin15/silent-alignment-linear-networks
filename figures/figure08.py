"""Run finite simulations for Figure 8, log-time spectral measures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs import figure08_config
from src.laws import paper_phi
from src.runs import parser, run_and_save


def main() -> None:
    args = parser(__doc__).parse_args()
    run_and_save("figure08_spectral_measures", figure08_config(), args, phi_c=paper_phi())


if __name__ == "__main__":
    main()