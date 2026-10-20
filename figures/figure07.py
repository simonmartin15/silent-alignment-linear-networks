"""Run the finite dynamics for Figure 7"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configs import figure07_configs
from src.laws import paper_phi
from src.runs import parser, run_and_save


def main() -> None:
    args = parser(__doc__).parse_args()
    small_cfg, large_cfg = figure07_configs()
    run_and_save("figure07_rho_staircase", small_cfg, args, phi_c=paper_phi())
    run_and_save("figure07_large_dynamics", large_cfg, args, phi_c=paper_phi())


if __name__ == "__main__":
    main()