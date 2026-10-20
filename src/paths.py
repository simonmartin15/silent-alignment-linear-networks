"""
Folders of the repository.

They are located from this file, so they do not depend on the working
directory.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"                                # simulation archives (.npz)
FIGURES_DIR = ROOT / "figures" / "pdf"                  # the figures of the paper
VALIDATION_PLOTS_DIR = ROOT / "validation" / "plots"    # plots of the numerical checks
GIF_DIR = ROOT / "animations" / "gifs"                  # animations of the spectra
GIF_PREVIEW_DIR = GIF_DIR / "previews"                  # frame previews of the animations