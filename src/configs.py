"""
Configurations shared by the figure scripts and the validation runs.

The step-size checks (validations 2 to 4) reuse these configurations, with the
same seeds and save times, so that their runs are paired with the figure
archives. The configurations of 
Figures 2 to 5 are only used by their own scripts, and stay there.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .laws import ArrayLike, normalized_density_grid, paper_phi, quantile_spectrum
from .simulator import FiniteSimulationConfig


# ----------------------------------------------------------------------------
# Figure 6: top-space alignment rates
# ----------------------------------------------------------------------------

ALPHA = 0.30
KAPPA = 0.55
KAPPA_STAR = 0.90
RHOS = (1e-5, 1e-10, 1e-15, 1e-20, 1e-25)


def gapped_density(x: ArrayLike) -> np.ndarray:
    """Two uniform bands with exactly ALPHA/KAPPA_STAR mass above the gap."""
    xx = np.asarray(x, dtype=float)
    lower = (0.35, 1.05)
    upper = (1.25, 2.00)
    upper_mass = ALPHA / KAPPA_STAR
    out = np.zeros_like(xx)
    lower_mask = (xx >= lower[0]) & (xx <= lower[1])
    upper_mask = (xx >= upper[0]) & (xx <= upper[1])
    out[lower_mask] = (1.0 - upper_mass) / (lower[1] - lower[0])
    out[upper_mask] = upper_mass / (upper[1] - upper[0])
    return out


def prescribed_spectrum(
    cfg: FiniteSimulationConfig,
    regime: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Teacher singular values, density grid, density, x_alpha and gap width of a Figure 6 regime."""
    d = int(cfg.d)
    m_star = min(int(round(cfg.kappa_star * d)), d)
    rank = int(np.floor(ALPHA * d))
    if not (0 < rank < m_star):
        raise ValueError("The chosen alpha must give a non-trivial rank below m_star.")

    if regime == "continuous":
        pdf = paper_phi()
        values, grid, density, cdf = quantile_spectrum(m_star, pdf)
        x_alpha = float(np.interp(1.0 - ALPHA / cfg.kappa_star, cdf, grid))
        epsilon_alpha = 0.0
    elif regime == "gapped":
        upper = 1.25 + (np.arange(rank, dtype=float) + 0.5) * (2.00 - 1.25) / rank
        lower_count = m_star - rank
        lower = 0.35 + (np.arange(lower_count, dtype=float) + 0.5) * (1.05 - 0.35) / lower_count
        values = np.concatenate((upper[::-1], lower[::-1]))
        grid, density, _ = normalized_density_grid(gapped_density)
        x_alpha = 1.25
        epsilon_alpha = 0.20
    else:
        raise ValueError(f"Unknown regime: {regime}.")

    return values, grid, density, x_alpha, epsilon_alpha


def figure06_config() -> FiniteSimulationConfig:
    """Configuration of both Figure 6 regimes (also used by validation 2)."""
    return FiniteSimulationConfig(
        d=500,
        kappa=KAPPA,
        tau=1.0,
        kappa_star=KAPPA_STAR,
        rhos=RHOS,
        eta=1e-3,
        num_save=360,
        reps=10,
        top_eig_fracs=(ALPHA,),
        dtype="float64",
        store_uv_spectra=False,
        store_prediction_spectrum=False,
        store_uv_range_projectors=False,
        store_uv_top_projectors=True,
        store_prediction_projectors=False,
        store_prediction_top_projectors=True,
        store_imbalance=False,
        save_grid="log",
        t_max=40.0,
    )


# ----------------------------------------------------------------------------
# Figure 7: staircase of activation times
# ----------------------------------------------------------------------------

def figure07_configs() -> tuple[FiniteSimulationConfig, FiniteSimulationConfig]:
    """Configurations of the two Figure 7 panels, d=24 and d=96 (also used by validation 3)."""
    kappa = 0.55
    kappa_star = 0.90
    small_cfg = FiniteSimulationConfig(
        d=24,
        kappa=kappa,
        tau=1.0,
        kappa_star=kappa_star,
        rhos=(1e-8, 1e-16, 1e-32, 1e-64),
        eta=2e-3,
        t_max=110.0,
        num_save=1800,
        reps=1,
        dtype="float64",
        store_uv_spectra=False,
        store_prediction_spectrum=False,
        store_uv_range_projectors=False,
        store_uv_top_projectors=False,
        store_prediction_projectors=False,
        store_prediction_top_projectors=False,
        store_imbalance=False,
        save_grid="linear",
    )
    large_cfg = replace(small_cfg, d=96)
    return small_cfg, large_cfg


# ----------------------------------------------------------------------------
# Figure 8: spectral measures at logarithmic times
# ----------------------------------------------------------------------------

def figure08_config() -> FiniteSimulationConfig:
    """Configuration of Figure 8 (also used by validation 4)."""
    return FiniteSimulationConfig(
        d=500,
        kappa=0.55,
        tau=1.0,
        kappa_star=0.90,
        rhos=(1e-40,),
        eta=1e-3,
        t_max=100.0,
        num_save=400,
        reps=10,
        dtype="float64",
        store_uv_spectra=True,
        store_prediction_spectrum=True,
        store_uv_range_projectors=False,
        store_uv_top_projectors=False,
        store_prediction_projectors=False,
        store_prediction_top_projectors=False,
        save_grid="linear",
    )