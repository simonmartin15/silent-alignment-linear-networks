"""
Animated GIFs of the spectral measures mu_t^U, mu_t^V and mu_t^Z: theory curves of
Theorem 4, optionally overlaid with gradient-descent histograms. Results are saved
in data/ (animation_<name>_theory.npz and animation_<name>_gd.npz).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .paths import DATA_DIR, GIF_PREVIEW_DIR
from .paths import GIF_DIR
from .theory import SignalLaw, make_theory


MAX_T_GAMMA = 350.0
THEORY_CACHE_VERSION = 2
MIN_AUTO_NODES = 600
MAX_AUTO_NODES = 4000
CURVES = ("alignment", "loss")


# ----------------------------------------------------------------------------
# Laws of the singular values of C (the density phi_C)
# ----------------------------------------------------------------------------

def paper_pdf(a: float = 0.35, b: float = 2.0) -> Callable[[np.ndarray], np.ndarray]:
    """Two-bump law of the paper figures."""
    width = b - a

    def pdf(x):
        x = np.asarray(x, dtype=float)
        u = (x - a) / width
        left_bump = 0.75 * np.exp(-0.5 * ((u - 0.30) / 0.13) ** 2)
        right_bump = 0.45 * np.exp(-0.5 * ((u - 0.70) / 0.18) ** 2)
        shoulder = 0.16 * (1.0 + 0.35 * np.cos(2.0 * np.pi * u))
        out = np.zeros_like(x)
        mask = (x >= a) & (x <= b)
        out[mask] = np.maximum(left_bump[mask] + right_bump[mask] + shoulder[mask], 0.0)
        return out

    return pdf


def uniform_pdf(a: float = 0.35, b: float = 2.0) -> Callable[[np.ndarray], np.ndarray]:
    def pdf(x):
        x = np.asarray(x, dtype=float)
        return np.where((x >= a) & (x <= b), 1.0 / (b - a), 0.0)

    return pdf


def edge_pdf(exponent: float, a: float = 0.35, b: float = 2.0) -> Callable[[np.ndarray], np.ndarray]:
    """Density proportional to (x - a)^exponent on [a, b], as in Figure 5."""

    def pdf(x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        mask = (x >= a) & (x <= b)
        out[mask] = (x[mask] - a) ** exponent
        return out

    return pdf


LAWS: Dict[str, Callable[[float, float], Callable[[np.ndarray], np.ndarray]]] = {
    "paper": paper_pdf,
    "uniform": uniform_pdf,
    "edge0": lambda a, b: edge_pdf(0.0, a, b),
    "edge1": lambda a, b: edge_pdf(1.0, a, b),
}


def law_pdf(name: str, support: Tuple[float, float]) -> Callable[[np.ndarray], np.ndarray]:
    if name not in LAWS:
        raise ValueError(f"Unknown law {name!r}. Available: {sorted(LAWS)}.")
    return LAWS[name](*support)


# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------

@dataclass
class GifSettings:
    """Everything that defines one GIF. Only `name` is required.

    kappa = m/d, kappa_star = rank(C)/min(p, d), tau = p/d, rho = initialization scale.
    time_axis: "linear", "log" or "s" (s = t / log(1/rho)).
    reference: "target" (kappa* phi_C), "endpoint" (t -> infinity limit), "none" or "auto".
    curves: "alignment" and/or "loss", drawn only with simulate=True.
    """

    name: str
    title: Optional[str] = None
    kappa: float = 0.55
    kappa_star: float = 0.90
    tau: float = 1.0
    rho: float = 0.1
    rhos: Optional[Tuple[float, ...]] = None
    law: str = "paper"
    support: Tuple[float, float] = (0.35, 2.0)
    time_axis: str = "linear"
    t_max: float = 10.0
    t_min: Optional[float] = None
    s_range: Optional[Tuple[float, float]] = None
    n_frames: int = 60
    n_energies: int = 400
    x_min: float = 0.01
    x_max: Optional[float] = None
    smoothing: float = 5e-3
    n_quad: Optional[int] = None
    panels: Tuple[str, ...] = ("Z",)
    reference: str = "auto"
    show_thresholds: bool = True
    show_timeline: bool = True
    y_max: Optional[float] = None
    y_headroom: float = 1.15
    x_star_label: str = "legend"
    curves: Tuple[str, ...] = ()
    n_bins: int = 50
    fps: int = 12
    hold_last: int = 15
    dpi: int = 110
    simulate: bool = False
    d: int = 400
    reps: int = 4
    eta: float = 1e-3
    dtype: str = "float64"
    device: Optional[str] = None
    seed: int = 0

    def __post_init__(self) -> None:
        self.support = tuple(float(v) for v in self.support)
        self.panels = tuple(self.panels)
        if self.s_range is not None:
            self.s_range = tuple(float(v) for v in self.s_range)
        unknown = set(self.panels) - {"U", "V", "Z"}
        if unknown or not self.panels:
            raise ValueError(f"panels must be a non-empty subset of ('U', 'V', 'Z'); got {self.panels}.")
        self.panels = tuple(p for p in ("U", "V", "Z") if p in self.panels)
        if set(self.curves) - set(CURVES):
            raise ValueError(f"curves must be a subset of {CURVES}; got {tuple(self.curves)}.")
        self.curves = tuple(c for c in CURVES if c in self.curves)
        if self.time_axis not in {"linear", "log", "s"}:
            raise ValueError("time_axis must be 'linear', 'log' or 's'.")
        if self.time_axis == "s" and self.s_range is None:
            raise ValueError("time_axis='s' needs s_range=(s_min, s_max).")
        if self.reference not in {"auto", "target", "endpoint", "none"}:
            raise ValueError("reference must be 'auto', 'target', 'endpoint' or 'none'.")
        if self.x_star_label not in {"legend", "plot"}:
            raise ValueError("x_star_label must be 'legend' or 'plot'.")
        if self.law not in LAWS:
            raise ValueError(f"Unknown law {self.law!r}. Available: {sorted(LAWS)}.")
        if self.rhos is not None:
            self.rhos = tuple(float(r) for r in self.rhos)
            if self.panels != ("Z",):
                raise ValueError("Several values of rho are drawn on a single Z panel: use panels=('Z',).")
            if self.time_axis == "s":
                raise ValueError("Several values of rho need a common time axis: use 'linear' or 'log'.")
            self.rho = max(self.rhos)
        if self.rho <= 0.0 or (self.rhos is not None and min(self.rhos) <= 0.0):
            raise ValueError("rho must be positive.")

    @property
    def log_inv_rho(self) -> float:
        return float(np.log(1.0 / self.rho))

    @property
    def x_upper(self) -> float:
        """Right end of the horizontal axis."""
        if self.x_max is not None:
            return float(self.x_max)
        return 1.12 * (self.support[1] + 2.0 * self.rho)

    @property
    def needed_nodes(self) -> int:
        """Quadrature nodes so that no node shows as a bump: n >= (pi/2)(b-a) log(1/rho) / (4 smoothing)."""
        a, b = self.support
        return int(np.ceil(0.5 * np.pi * (b - a) * self.log_inv_rho / (4.0 * self.smoothing)))

    @property
    def quad_nodes(self) -> int:
        """Number of quadrature nodes used: `n_quad` if given, else the automatic choice."""
        if self.n_quad is not None:
            return int(self.n_quad)
        return int(min(MAX_AUTO_NODES, max(MIN_AUTO_NODES, self.needed_nodes)))

    @property
    def reference_kind(self) -> str:
        if self.reference != "auto":
            return self.reference
        return "target" if self.panels == ("Z",) else "endpoint"

    def theory_fields(self) -> Dict[str, Any]:
        """Fields that change the theory computation (used for the cache key)."""
        keys = ("kappa", "kappa_star", "tau", "rho", "law", "support", "time_axis", "t_max", "t_min",
                "s_range", "n_frames", "n_energies", "x_min", "x_max", "smoothing", "eta")
        out = {k: getattr(self, k) for k in keys}
        out["n_quad"] = self.quad_nodes
        out["branches"] = list(theory_branches(self))
        out["version"] = THEORY_CACHE_VERSION
        return out

    def simulation_fields(self) -> Dict[str, Any]:
        """Fields that change the gradient-descent run (used for the cache key)."""
        keys = ("kappa", "kappa_star", "tau", "rho", "law", "support", "time_axis", "t_max", "t_min",
                "s_range", "n_frames", "d", "reps", "eta", "dtype", "seed")
        return {k: getattr(self, k) for k in keys}


def theory_branches(settings: GifSettings) -> Tuple[str, ...]:
    """Branches of the self-consistent equation needed for the panels (Z is computed from U)."""
    return ("U", "V") if "V" in settings.panels else ("U",)


def series_settings(settings: GifSettings) -> List[GifSettings]:
    """One single-rho setting per curve (a single one unless `rhos` is given)."""
    if settings.rhos is None:
        return [settings]
    return [
        replace(settings, name=f"{settings.name}_rho{k}", rho=rho, rhos=None, x_max=settings.x_upper)
        for k, rho in enumerate(settings.rhos)
    ]


def frame_times(settings: GifSettings) -> np.ndarray:
    """Frame times, rounded to positive multiples of the gradient-descent step."""
    s = settings
    n = int(s.n_frames)
    if s.time_axis == "linear":
        t0 = s.t_min if s.t_min is not None else s.t_max / n
        times = np.linspace(t0, s.t_max, n)
    elif s.time_axis == "log":
        t0 = s.t_min if s.t_min is not None else 1e-3 * s.t_max
        times = np.geomspace(t0, s.t_max, n)
    else:
        times = np.linspace(s.s_range[0], s.s_range[1], n) * s.log_inv_rho
    steps = np.unique(np.maximum(np.rint(times / s.eta), 1).astype(np.int64))
    times = steps * s.eta
    t_gamma = times[-1] * np.hypot(s.support[1], s.rho)
    if t_gamma > MAX_T_GAMMA:
        raise ValueError(
            f"The last frame has t * gamma = {t_gamma:.0f} > {MAX_T_GAMMA:.0f}, beyond the density solver. "
            "Reduce t_max (or s_range)."
        )
    return times


def frame_labels(settings: GifSettings, times: np.ndarray) -> np.ndarray:
    """Values shown on the clock and the time bar: t, or s = t / log(1/rho) on the s axis."""
    return times / settings.log_inv_rho if settings.time_axis == "s" else times


# ----------------------------------------------------------------------------
# Computations (with cache)
# ----------------------------------------------------------------------------

def _key(fields: Dict[str, Any]) -> str:
    text = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _load_cache(path: Path, key: str, required: Tuple[str, ...] = ()) -> Optional[Dict[str, np.ndarray]]:
    """Cached arrays, or None if the cache is missing, was made with other settings, or lacks `required`."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "key" not in data.files or str(data["key"]) != key:
            return None
        if any(name not in data.files for name in required):
            return None
        return {k: data[k] for k in data.files}


def endpoint_densities(settings: GifSettings, law: SignalLaw, x: np.ndarray, beta: float,
                       threshold: float) -> Dict[str, np.ndarray]:
    """Continuous part of the t -> infinity limits of mu_t^U, mu_t^V, mu_t^Z (Corollary 5)."""
    rho, ks, tau = settings.rho, settings.kappa_star, settings.tau
    tiny = np.finfo(float).tiny

    def pull_back(gamma, weight):
        valid = gamma > rho
        xs = np.sqrt(np.where(valid, gamma * gamma - rho * rho, 0.0))
        dens = weight * np.asarray(law.pdf(xs), dtype=float) * gamma / np.maximum(xs, tiny)
        return np.where(valid & (xs > threshold), dens, 0.0)

    return {
        "endpoint_U": pull_back(x + rho, beta * ks / tau),
        "endpoint_V": pull_back(x - rho, beta * ks),
        "endpoint_Z": np.where(x > threshold, ks * np.asarray(law.pdf(x), dtype=float), 0.0),
    }


def theory_frames(settings: GifSettings, *, recompute: bool = False, verbose: bool = True) -> Dict[str, np.ndarray]:
    """High-dimensional densities at every frame time (saved in data/animation_<name>_theory.npz)."""
    s = settings
    if s.rhos is not None:
        raise ValueError("theory_frames takes a single-rho setting; use series_settings first.")
    key = _key(s.theory_fields())
    path = DATA_DIR / f"animation_{s.name}_theory.npz"
    if not recompute:
        cached = _load_cache(path, key)
        if cached is not None:
            if verbose:
                print(f"theory: loaded {path.name}")
            return cached

    times = frame_times(s)
    law = SignalLaw.from_pdf(law_pdf(s.law, s.support), s.support, normalize=True, name=s.law)
    theory = make_theory(law, kappa=s.kappa, tau=s.tau, kappa_star=s.kappa_star, rho=s.rho, n_quad=s.quad_nodes)
    x = np.linspace(s.x_min, s.x_upper, int(s.n_energies))
    if verbose:
        print(f"theory ({s.name}): {len(times)} frames x {len(theory_branches(s))} branch(es), "
              f"{s.quad_nodes} quadrature nodes, a few seconds per frame")
        if s.quad_nodes < s.needed_nodes:
            print(f"  warning: about {s.needed_nodes} nodes would be needed to avoid ripples at smoothing "
                  f"{s.smoothing:g}. Increase `smoothing` (e.g. to {2 * s.smoothing:g}) or set `n_quad`.")
    out = theory.density_grid(
        x, times, branches=theory_branches(s), eta=s.smoothing, include_prediction_from_u=True, verbose=False,
    )
    threshold = float(theory.x_star)
    frames = {
        "key": np.array(key),
        "times": times,
        "x": x,
        "density_U": out["density_U_unnormalized"],
        "density_Z": out["density_Z_unnormalized"],
        "target_Z": s.kappa_star * np.asarray(law.pdf(x), dtype=float),
        "x_star": np.array(theory.paper_x_star),
        "x_threshold": np.array(threshold),
        "underparameterized": np.array(s.kappa < theory.signal_weight),
        **endpoint_densities(s, law, x, theory.beta, threshold),
    }
    if "V" in theory_branches(s):
        frames["density_V"] = out["density_V_unnormalized"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **frames)
    if verbose:
        print(f"theory: saved {path.name}")
    return frames


def simulation_frames(settings: GifSettings, *, recompute: bool = False, verbose: bool = True) -> Dict[str, np.ndarray]:
    """Gradient-descent spectra, loss and alignment error at every frame time (needs PyTorch)."""
    s = settings
    if s.rhos is not None:
        raise ValueError("simulation_frames takes a single-rho setting; use series_settings first.")
    key = _key(s.simulation_fields())
    path = DATA_DIR / f"animation_{s.name}_gd.npz"
    if not recompute:
        cached = _load_cache(path, key, ("loss", "loss_initial", "alignment_error") if s.curves else ())
        if cached is not None:
            if verbose:
                print(f"gradient descent: loaded {path.name}")
            return cached

    try:
        from .simulator import FiniteSimulationConfig, run_simulation
    except ImportError as exc:
        raise ImportError("Gradient-descent frames need simulator.py and PyTorch. Use simulate=False otherwise.") from exc

    times = frame_times(s)
    cfg = FiniteSimulationConfig(
        d=s.d,
        kappa=s.kappa,
        tau=s.tau,
        kappa_star=s.kappa_star,
        rhos=(s.rho,),
        eta=s.eta,
        t_max=float(times[-1]),
        save_times=tuple(float(t) for t in times),
        reps=s.reps,
        seed=s.seed,
        dtype=s.dtype,
        device=s.device,
        store_uv_spectra=True,
        store_prediction_spectrum=True,
        store_uv_range_projectors=False,
        store_uv_top_projectors=False,
        store_prediction_projectors=True,
        store_prediction_top_projectors=False,
        store_imbalance=False,
        progress=verbose,
    )
    if verbose:
        print(f"gradient descent ({s.name}): d = {s.d}, {s.reps} repetitions, {int(round(times[-1] / s.eta))} steps each")
    result = run_simulation(cfg, phi_c=law_pdf(s.law, s.support), support_phi=s.support)
    if not np.allclose(result.times, times, rtol=0.0, atol=0.5 * s.eta):
        raise RuntimeError("The simulator did not save at the frame times.")
    frames = {
        "key": np.array(key),
        "times": result.times,
        "eigvals_U": result.eigvals_u[0],
        "eigvals_V": result.eigvals_v[0],
        "singular_values_Z": result.singular_values_z[0],
        "rank": np.array(result.dimensions["selected_rank"]),
        "loss": result.loss[0],
        "loss_initial": np.sum(result.teacher_singular_values ** 2, axis=1) / result.dimensions["min_pd"],
        "alignment_error": result.prediction_right_delta[0],
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **frames)
    if verbose:
        print(f"gradient descent: saved {path.name}")
    return frames


def histogram_heights(simulation: Dict[str, np.ndarray], panel: str, edges: np.ndarray) -> np.ndarray:
    """Histogram of the continuous part of mu_t^panel at every frame, normalized by all the eigenvalues."""
    values = np.asarray(simulation[{"U": "eigvals_U", "V": "eigvals_V", "Z": "singular_values_Z"}[panel]], dtype=float)
    n_frames, reps, n_total = values.shape
    rank = int(simulation["rank"])
    widths = np.diff(edges)
    heights = np.empty((n_frames, widths.size))
    for i in range(n_frames):
        kept = values[i, :, :rank].ravel()
        counts, _ = np.histogram(kept[np.isfinite(kept)], bins=edges)
        heights[i] = counts / (reps * n_total * widths)
    return heights


def curve_values(settings: GifSettings, simulation: Dict[str, np.ndarray], curve: str) -> np.ndarray:
    """Normalized alignment error or loss of gradient descent at every frame, averaged over the repetitions."""
    if curve == "loss":
        values = np.asarray(simulation["loss"], dtype=float) / np.asarray(simulation["loss_initial"], dtype=float)
    else:
        beta = min(1.0, settings.tau)
        random_value = 2.0 * (1.0 - settings.kappa) * min(settings.kappa, beta * settings.kappa_star)
        values = np.asarray(simulation["alignment_error"], dtype=float) / random_value
    return np.nanmean(values, axis=1)


Series = Tuple[GifSettings, Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]


def compute_series(settings: GifSettings, *, simulate: Optional[bool] = None, recompute: bool = False,
                   verbose: bool = True) -> List[Series]:
    """Theory (and optionally gradient-descent) frames for every curve of the GIF."""
    use_gd = settings.simulate if simulate is None else simulate
    series: List[Series] = []
    for sub in series_settings(settings):
        theory = theory_frames(sub, recompute=recompute, verbose=verbose)
        simulation = simulation_frames(sub, recompute=recompute, verbose=verbose) if use_gd else None
        if simulation is not None and not np.allclose(simulation["times"], theory["times"], rtol=0.0, atol=0.5 * sub.eta):
            raise RuntimeError(f"Cached theory and simulation of {sub.name!r} have different frame times. Use recompute=True.")
        series.append((sub, theory, simulation))
    first = series[0][1]["times"]
    if any(not np.allclose(item[1]["times"], first) for item in series[1:]):
        raise RuntimeError("The curves do not share the same frame times.")
    return series


# ----------------------------------------------------------------------------
# Appearance
# ----------------------------------------------------------------------------

PALETTE = {
    "U": "#2B6F9E",
    "V": "#009E8E",
    "Z": "#6D4C9B",
    "ink": "#202124",
    "muted": "#80868B",
    "accent": "#D95F02",
    "track": "#E8EAED",
}

RHO_COLORS = ("#D1495B", "#F28E2B", "#1B9E77", "#4E79A7", "#B07AA1")

DARK_PALETTE = {
    "U": "#58A6D8",
    "V": "#2EC4B6",
    "Z": "#A887D6",
    "ink": "#E6EDF3",
    "muted": "#8B949E",
    "accent": "#F0883E",
    "track": "#30363D",
}

DARK_RHO_COLORS = ("#F07178", "#F6A04D", "#3FC59A", "#79A6E0", "#C9A0C4")

PANEL_TITLES = {
    "U": r"$\mu_t^{\mathbf{U}}$",
    "V": r"$\mu_t^{\mathbf{V}}$",
    "Z": r"$\mu_t^{\mathbf{Z}}$",
}

PANEL_XLABELS = {
    "U": r"eigenvalue of $\mathbf{U}\mathbf{U}^\top$",
    "V": r"eigenvalue of $\mathbf{V}\mathbf{V}^\top$",
    "Z": r"singular value of $\mathbf{Z}$",
}

CURVE_TITLES = {
    "alignment": "Alignment error",
    "loss": "Loss",
}

STYLE = {
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#5F6368",
    "axes.linewidth": 0.8,
    "xtick.color": "#5F6368",
    "ytick.color": "#5F6368",
    "axes.grid": True,
    "grid.color": "#E8EAED",
    "grid.linewidth": 0.6,
}

DARK_STYLE = {
    **STYLE,
    "axes.facecolor": "#0D1117",
    "axes.edgecolor": "#8B949E",
    "xtick.color": "#8B949E",
    "ytick.color": "#8B949E",
    "grid.color": "#30363D",
    "text.color": "#E6EDF3",
    "axes.labelcolor": "#E6EDF3",
    "axes.titlecolor": "#E6EDF3",
    "legend.labelcolor": "#E6EDF3",
}

THEMES = {
    "light": dict(palette=PALETTE, rho_colors=RHO_COLORS, style=STYLE, background="white", frame="#5F6368", suffix=""),
    "dark": dict(palette=DARK_PALETTE, rho_colors=DARK_RHO_COLORS, style=DARK_STYLE, background="#0D1117",
                 frame="#8B949E", suffix="_dark"),
}

# Layout in inches.
LAYOUT = dict(
    panel_w=3.0,
    panel_h=2.15,
    curve_h=1.75,
    row_gap=0.12,
    left=0.55,
    right=0.25,
    gap=0.55,
    header=0.40,
    title_gap=0.32,
    below=0.50,
    legend_gap=0.06,
    legend_row=0.24,
    bar_gap=0.10,
    bar_h=0.07,
    bar_labels=0.26,
    bottom=0.10,
    max_cols_single=3,
)


def rho_label(rho: float) -> str:
    exponent = int(np.floor(np.log10(rho) + 1e-9))
    mantissa = rho / 10.0 ** exponent
    if exponent >= -2:
        return rf"$\rho = {rho:g}$"
    if abs(mantissa - 1.0) < 1e-9:
        return rf"$\rho = 10^{{{exponent}}}$"
    return rf"$\rho = {mantissa:g}\times 10^{{{exponent}}}$"


def _format_time(value: float, axis: str) -> str:
    name = "s" if axis == "s" else "t"
    digits = 2 if (axis == "s" or value < 10) else 1
    return rf"${name} = {value:.{digits}f}$"


def _reference(settings: GifSettings, theory: Dict[str, np.ndarray], panel: str):
    """Dashed reference curve of a panel and its legend label, or (None, None)."""
    kind = settings.reference_kind
    if kind == "none":
        return None, None
    if panel == "Z" and kind == "target":
        return theory["target_Z"], r"$\kappa^\ast\phi_C$"
    return theory[f"endpoint_{panel}"], r"$t \to \infty$"


def _grown_region(settings: GifSettings, panel: str, rho: float, x: np.ndarray) -> np.ndarray:
    """Points away from the peak of the directions not grown yet (at 0, or at 2 rho for V V^T)."""
    start = 0.1 * settings.x_upper + (2.0 * rho if panel == "V" else 0.0)
    return x >= start


def _common_y_max(settings: GifSettings, series: List[Series],
                  heights: Dict[Tuple[int, str], np.ndarray], edges: np.ndarray) -> float:
    """One vertical range for all panels: late frames and the references, away from the not-yet-grown peak."""
    if settings.y_max is not None:
        return float(settings.y_max)
    centers = 0.5 * (edges[:-1] + edges[1:])
    top = 0.0
    for panel in settings.panels:
        for k, (sub, theory, _) in enumerate(series):
            x = theory["x"]
            density = theory[f"density_{panel}"]
            late = density[int(0.6 * len(density)):, :]
            top = max(top, float(np.max(late[:, _grown_region(settings, panel, sub.rho, x)])))
            reference, _ = _reference(settings, theory, panel)
            if reference is not None:
                top = max(top, float(np.max(reference)))
            if (k, panel) in heights:
                h = heights[(k, panel)]
                late_h = h[int(0.6 * len(h)):, :]
                top = max(top, float(np.max(late_h[:, _grown_region(settings, panel, sub.rho, centers)])))
    return settings.y_headroom * top if top > 0 else 1.0


def build_animation(settings: GifSettings, series: List[Series], theme: str = "light"):
    """Figure, update function and number of frames. Call inside rc_context(THEMES[theme]["style"])."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

    s = settings
    L = LAYOUT
    palette = THEMES[theme]["palette"]
    rho_colors = THEMES[theme]["rho_colors"]
    multi = len(series) > 1
    theory0 = series[0][1]
    times = np.asarray(theory0["times"], dtype=float)
    labels = frame_labels(s, times)
    n_frames = len(times)
    x = theory0["x"]
    edges = np.linspace(s.x_min, s.x_upper, int(s.n_bins) + 1)
    widths = np.diff(edges)
    heights = {
        (k, panel): histogram_heights(sim, panel, edges)
        for k, (_, _, sim) in enumerate(series) if sim is not None for panel in s.panels
    }
    has_gd = bool(heights)
    show_front = s.time_axis == "s" and s.show_thresholds
    show_x_star = s.show_thresholds and bool(theory0["underparameterized"])
    threshold = float(theory0["x_threshold"])
    curve_names = s.curves if has_gd else ()
    curve_data = {
        (k, curve): curve_values(sub, sim, curve)
        for k, (sub, _, sim) in enumerate(series) if sim is not None for curve in curve_names
    }

    def color_of(k: int, panel: str) -> str:
        return rho_colors[k % len(rho_colors)] if multi else palette[panel]

    n_panels = len(s.panels)
    entries: Dict[str, Any] = {}
    if multi:
        for k, (sub, _, _) in enumerate(series):
            entries[rho_label(sub.rho)] = Line2D([], [], color=color_of(k, "Z"), lw=2.0)
    else:
        single_color = palette[s.panels[0]] if n_panels == 1 else palette["ink"]
        entries["theory"] = Line2D([], [], color=single_color, lw=2.0)
    if has_gd:
        if multi:
            entries["gradient descent"] = Line2D([], [], color=palette["muted"], lw=0.9, drawstyle="steps-mid")
        else:
            gd_color = palette[s.panels[0]] if n_panels == 1 else palette["muted"]
            entries["gradient descent"] = Patch(color=gd_color, alpha=0.35)
    for panel in s.panels:
        _, ref_label = _reference(s, theory0, panel)
        if ref_label is not None:
            entries.setdefault(ref_label, Line2D([], [], color=palette["ink"], lw=1.1, ls=(0, (4, 3)), alpha=0.75))
    if "Z" in s.panels and show_x_star and s.x_star_label == "legend":
        entries[r"$x_\ast$"] = Line2D([], [], color=palette["ink"], lw=0.9, ls=":")
    if "Z" in s.panels and show_front:
        entries[r"$x_s$"] = Line2D([], [], color=palette["accent"], lw=1.3)
    n_curves = len(curve_names)
    n_wide = max(n_panels, n_curves)
    n_cols = len(entries) if n_wide > 1 else min(len(entries), L["max_cols_single"])
    n_rows = -(-len(entries) // n_cols)

    panels_w = n_wide * L["panel_w"] + (n_wide - 1) * L["gap"]
    panel_w = (panels_w - (n_panels - 1) * L["gap"]) / n_panels
    curve_w = (panels_w - (n_curves - 1) * L["gap"]) / max(n_curves, 1)
    width = L["left"] + panels_w + L["right"]
    bar_y = L["bar_labels"] if s.show_timeline else 0.0
    legend_bottom = (bar_y + L["bar_h"] + L["bar_gap"]) if s.show_timeline else L["bottom"]
    legend_top = legend_bottom + n_rows * L["legend_row"]
    curve_y = legend_top + L["legend_gap"] + L["below"]
    panel_y = curve_y + (L["curve_h"] + L["title_gap"] + L["row_gap"] + L["below"] if n_curves else 0.0)
    height = panel_y + L["panel_h"] + L["title_gap"] + L["header"]

    fig = Figure(figsize=(width, height), dpi=s.dpi)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(THEMES[theme]["background"])

    y_top = _common_y_max(s, series, heights, edges)
    lines, bars, steps, front_lines = {}, {}, {}, []
    for j, panel in enumerate(s.panels):
        left = L["left"] + j * (panel_w + L["gap"])
        ax = fig.add_axes([left / width, panel_y / height, panel_w / width, L["panel_h"] / height])
        ax.set_xlim(0.0, s.x_upper)
        ax.set_ylim(0.0, y_top)
        ax.set_title(PANEL_TITLES[panel], pad=5)
        ax.set_xlabel(PANEL_XLABELS[panel], labelpad=2)
        ax.locator_params(axis="y", nbins=4)

        reference, _ = _reference(s, theory0, panel)
        if reference is not None:
            ax.plot(x, reference, color=palette["ink"], lw=1.1, ls=(0, (4, 3)), alpha=0.75, zorder=2)
        if panel == "Z" and show_x_star:
            ax.axvline(float(theory0["x_star"]), color=palette["ink"], lw=0.9, ls=":", zorder=2)
            if s.x_star_label == "plot":
                ax.annotate(r"$x_\ast$", xy=(float(theory0["x_star"]), 1.0), xycoords=("data", "axes fraction"),
                            xytext=(3, -3), textcoords="offset points", ha="left", va="top",
                            color=palette["ink"], zorder=5)
        if panel == "Z" and show_front:
            front_lines.append(ax.axvline(0.0, color=palette["accent"], lw=1.3, zorder=3))

        for k, (sub, theory, _) in enumerate(series):
            color = color_of(k, panel)
            if (k, panel) in heights:
                if multi:
                    steps[(k, panel)] = ax.stairs(heights[(k, panel)][0], edges, color=color, lw=0.9, alpha=0.7, zorder=1)
                else:
                    bars[(k, panel)] = ax.bar(edges[:-1], heights[(k, panel)][0], width=widths, align="edge",
                                              color=color, alpha=0.30, edgecolor="none", zorder=1)
            (lines[(k, panel)],) = ax.plot(x, theory[f"density_{panel}"][0], color=color, lw=2.0, zorder=4)

    curve_lines, curve_tips = {}, {}
    for j, curve in enumerate(curve_names):
        left = L["left"] + j * (curve_w + L["gap"])
        ax = fig.add_axes([left / width, curve_y / height, curve_w / width, L["curve_h"] / height])
        if s.time_axis == "log":
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(float(labels[0]), float(labels[-1]))
        ax.set_ylim(0.0, 1.05)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_title(CURVE_TITLES[curve], pad=5)
        ax.set_xlabel(r"$s = t/\log(1/\rho)$" if s.time_axis == "s" else r"time $t$", labelpad=2)
        for k in range(len(series)):
            if (k, curve) in curve_data:
                color = color_of(k, "Z")
                (curve_lines[(k, curve)],) = ax.plot([], [], color=color, lw=1.8, zorder=3)
                (curve_tips[(k, curve)],) = ax.plot([], [], ls="none", marker="o", ms=4.5, color=color, zorder=4)

    fig.legend(list(entries.values()), list(entries.keys()), loc="upper center",
               bbox_to_anchor=((L["left"] + 0.5 * panels_w) / width, legend_top / height),
               ncol=n_cols, frameon=False, handlelength=1.6, columnspacing=1.4, borderaxespad=0.0,
               borderpad=0.0, labelspacing=0.4)

    header_y = (height - 0.5 * L["header"]) / height
    if s.title:
        fig.text(0.10 / width, header_y, s.title, ha="left", va="center", fontsize=11, color=palette["ink"])
    clock = fig.text((L["left"] + panels_w) / width, header_y, "", ha="right", va="center", fontsize=12,
                     color=palette["ink"])

    progress = None
    if s.show_timeline:
        bar = fig.add_axes([L["left"] / width, bar_y / height, panels_w / width, L["bar_h"] / height])
        start, end = float(labels[0]), float(labels[-1])
        bar.set_xlim(start, end)
        bar.set_ylim(0.0, 1.0)
        bar.barh(0.5, end - start, left=start, height=1.0, color=palette["track"], zorder=0)
        (progress,) = bar.barh(0.5, 0.0, left=start, height=1.0, color=palette["muted"], zorder=1)
        bar.set_xticks([])
        bar.set_yticks([])
        bar.grid(False)
        for spine in bar.spines.values():
            spine.set_visible(False)
        for frac, value, align in ((0.0, start, "left"), (1.0, end, "right")):
            bar.annotate(_format_time(value, s.time_axis), (frac, 0.0), xycoords="axes fraction", xytext=(0, -4),
                         textcoords="offset points", ha=align, va="top", fontsize=9, color=THEMES[theme]["frame"])

    def update(frame: int):
        i = min(frame, n_frames - 1)
        for (k, panel), line in lines.items():
            line.set_ydata(series[k][1][f"density_{panel}"][i])
        for key, container in bars.items():
            for rect, h in zip(container.patches, heights[key][i]):
                rect.set_height(h)
        for key, patch in steps.items():
            patch.set_data(values=heights[key][i])
        if front_lines:
            # The front stops at x_* (rank constraint) or at the lower edge of the spectrum.
            xs = max(1.0 / (2.0 * float(labels[i])), threshold)
            for vline in front_lines:
                vline.set_xdata([xs, xs])
                vline.set_visible(xs < s.x_upper)
        for key, line in curve_lines.items():
            line.set_data(labels[: i + 1], curve_data[key][: i + 1])
            curve_tips[key].set_data(labels[i : i + 1], curve_data[key][i : i + 1])
        clock.set_text(_format_time(float(labels[i]), s.time_axis))
        if progress is not None:
            progress.set_width(float(labels[i]) - float(labels[0]))
        return []

    update(0)
    return fig, update, n_frames + int(s.hold_last)


def preview(settings: GifSettings, frame: int = -1, *, simulate: Optional[bool] = None,
            recompute: bool = False, verbose: bool = True, theme: str = "light") -> Path:
    """Save one frame as animations/gifs/previews/<name>_preview.png and return its path (uses the saved archives)."""
    import matplotlib as mpl

    series = compute_series(settings, simulate=simulate, recompute=recompute, verbose=verbose)
    path = GIF_PREVIEW_DIR / f"{settings.name}{THEMES[theme]['suffix']}_preview.png"
    GIF_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(THEMES[theme]["style"]):
        fig, update, _ = build_animation(settings, series, theme)
        update(frame % len(series[0][1]["times"]))
        fig.savefig(path, dpi=settings.dpi, facecolor=THEMES[theme]["background"])
    return path


def make_gif(settings: GifSettings, *, simulate: Optional[bool] = None, recompute: bool = False,
             out_dir: Path = GIF_DIR, verbose: bool = True, theme: str = "light") -> Path:
    """Compute (or load) the frames and write animations/gifs/<name>.gif. Returns its path."""
    import io

    import matplotlib as mpl
    from PIL import Image

    series = compute_series(settings, simulate=simulate, recompute=recompute, verbose=verbose)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{settings.name}{THEMES[theme]['suffix']}.gif"
    no_dither = getattr(Image, "Dither", Image).NONE
    frames = []
    with mpl.rc_context(THEMES[theme]["style"]):
        fig, update, total = build_animation(settings, series, theme)
        for frame in range(total):
            update(frame)
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=settings.dpi, facecolor=THEMES[theme]["background"])
            buffer.seek(0)
            with Image.open(buffer) as image:
                frames.append(image.convert("RGB").quantize(colors=256, dither=no_dither))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / settings.fps), loop=0)
    if verbose:
        print(f"saved {path}")
    return path


def describe(settings: GifSettings) -> str:
    """One-line summary of a setting, for printing in the notebook."""
    regime = "under" if settings.kappa < min(1.0, settings.tau) * settings.kappa_star else "over"
    rho = ", ".join(f"{r:g}" for r in settings.rhos) if settings.rhos else f"{settings.rho:g}"
    return (f"{settings.name}: {regime}parameterized, kappa={settings.kappa}, kappa*={settings.kappa_star}, "
            f"tau={settings.tau}, rho={rho}, law={settings.law}, panels={''.join(settings.panels)}, "
            f"{settings.n_frames} frames on the {settings.time_axis} time axis")

