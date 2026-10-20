"""
Densities of the non-zero singular values of C, and deterministic spectra.

All densities are supported on DEFAULT_SUPPORT unless stated otherwise. They
need only NumPy and SciPy.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.integrate import trapezoid


ArrayLike = np.ndarray

DEFAULT_SUPPORT = (0.35, 2.0)


def edge_power_phi(
    exponent: int,
    a: float = DEFAULT_SUPPORT[0],
    b: float = DEFAULT_SUPPORT[1],
) -> Callable[[ArrayLike], np.ndarray]:
    """Density proportional to (x-a)^exponent on [a,b]."""
    if exponent < 0:
        raise ValueError("exponent must be nonnegative.")
    norm = (exponent + 1.0) / ((b - a) ** (exponent + 1.0))

    def pdf(x: ArrayLike) -> np.ndarray:
        xx = np.asarray(x, dtype=float)
        out = np.zeros_like(xx)
        mask = (xx >= a) & (xx <= b)
        out[mask] = norm * (xx[mask] - a) ** exponent
        return out

    return pdf


def paper_phi(a: float = DEFAULT_SUPPORT[0], b: float = DEFAULT_SUPPORT[1]) -> Callable[[ArrayLike], np.ndarray]:
    """Smooth non-uniform density used by the main paper figures."""
    width = b - a

    def raw(x: np.ndarray) -> np.ndarray:
        u = (x - a) / width
        left_bump = 0.75 * np.exp(-0.5 * ((u - 0.30) / 0.13) ** 2)
        right_bump = 0.45 * np.exp(-0.5 * ((u - 0.70) / 0.18) ** 2)
        shoulder = 0.16 * (1.0 + 0.35 * np.cos(2.0 * np.pi * u))
        return np.maximum(left_bump + right_bump + shoulder, 0.0)

    grid = np.linspace(a, b, 20000)
    mass = trapezoid(raw(grid), grid)

    def pdf(x: ArrayLike) -> np.ndarray:
        xx = np.asarray(x, dtype=float)
        out = np.zeros_like(xx)
        mask = (xx >= a) & (xx <= b)
        out[mask] = raw(xx[mask]) / mass
        return out

    return pdf


def cumulative_trapezoid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(grid)
    return np.concatenate(([0.0], np.cumsum(increments)))


def normalized_density_grid(
    pdf: Callable[[ArrayLike], np.ndarray],
    support: Tuple[float, float] = DEFAULT_SUPPORT,
    *,
    size: int = 100_001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(*support, size)
    density = np.maximum(np.asarray(pdf(grid), dtype=float), 0.0)
    cdf = cumulative_trapezoid(density, grid)
    density /= cdf[-1]
    cdf /= cdf[-1]
    return grid, density, cdf


def quantile_spectrum(
    count: int,
    pdf: Callable[[ArrayLike], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic spectrum: the (i + 1/2)/count quantiles of pdf, in decreasing order."""
    grid, density, cdf = normalized_density_grid(pdf)
    probabilities = (np.arange(count, dtype=float) + 0.5) / count
    singular_values = np.interp(probabilities, cdf, grid)[::-1].copy()
    return singular_values, grid, density, cdf