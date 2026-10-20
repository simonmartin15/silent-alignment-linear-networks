"""
High-dimensional deterministic equivalents for the linear-network figures.

The conventions follow the paper:

    mu_C = (1 - kappa_star) delta_0 + kappa_star phi_C(x) dx,
    tilde_mu_C = (1 - beta*kappa_star) delta_0 + beta*kappa_star phi_C(x) dx,
    beta = min(1, tau).

The density phi_C is always the conditional density of the nonzero teacher
singular values.  It should integrate to one on its support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import scipy.optimize as opt
from scipy.integrate import quad

ArrayLike = np.ndarray


def _as_array(x: ArrayLike | float) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _pdf_values(pdf: Callable[[ArrayLike], ArrayLike], x: ArrayLike) -> np.ndarray:
    y = np.asarray(pdf(np.asarray(x, dtype=float)), dtype=float)
    if y.shape == ():
        y = np.full_like(np.asarray(x, dtype=float), float(y))
    return y


@dataclass
class SignalLaw:
    """Continuous nonzero singular-value density phi_C on a compact interval."""

    pdf: Callable[[ArrayLike], ArrayLike]
    support: Tuple[float, float]
    name: str = "custom"

    @classmethod
    def from_pdf(
        cls,
        pdf: Callable[[ArrayLike], ArrayLike],
        support: Tuple[float, float],
        *,
        normalize: bool = True,
        name: str = "custom",
        quad_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "SignalLaw":
        a, b = map(float, support)
        if not (np.isfinite(a) and np.isfinite(b) and 0.0 <= a < b):
            raise ValueError("support must be a finite nonnegative interval (a, b).")

        qkw = dict(epsabs=1e-11, epsrel=1e-11, limit=300)
        if quad_kwargs:
            qkw.update(quad_kwargs)

        def raw(x: ArrayLike) -> np.ndarray:
            xx = np.asarray(x, dtype=float)
            yy = _pdf_values(pdf, xx)
            return np.where((xx >= a) & (xx <= b), np.maximum(yy, 0.0), 0.0)

        mass = quad(lambda z: float(raw(np.array([z]))[0]), a, b, **qkw)[0]
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"phi_C has nonpositive or infinite mass: {mass}.")

        if normalize:
            def normalized(x: ArrayLike) -> np.ndarray:
                return raw(x) / mass

            return cls(pdf=normalized, support=(a, b), name=name)

        if abs(mass - 1.0) > 5e-5:
            raise ValueError(
                f"phi_C should integrate to one; got {mass:.8g}. "
                "Pass normalize=True to normalize it automatically."
            )
        return cls(pdf=raw, support=(a, b), name=name)

    def pdf_scalar(self, x: float) -> float:
        return float(_pdf_values(self.pdf, np.array([x]))[0])

    def integrate(self, func: Callable[[float], float], **quad_kwargs: Any) -> float:
        a, b = self.support
        qkw = dict(epsabs=1e-11, epsrel=1e-11, limit=300)
        qkw.update(quad_kwargs)
        return float(quad(lambda z: self.pdf_scalar(z) * func(z), a, b, **qkw)[0])

    def cdf(self, x: float, **quad_kwargs: Any) -> float:
        a, b = self.support
        if x <= a:
            return 0.0
        if x >= b:
            return 1.0
        # Integrate the density up to x. An indicator over the full support can
        # make quad miss the jump and return a wrong value.
        qkw = dict(epsabs=1e-11, epsrel=1e-11, limit=300)
        qkw.update(quad_kwargs)
        return float(quad(lambda z: self.pdf_scalar(z), a, x, **qkw)[0])

    def tail(self, x: float, **quad_kwargs: Any) -> float:
        a, b = self.support
        if x <= a:
            return 1.0
        if x >= b:
            return 0.0
        qkw = dict(epsabs=1e-11, epsrel=1e-11, limit=300)
        qkw.update(quad_kwargs)
        return float(quad(lambda z: self.pdf_scalar(z), x, b, **qkw)[0])

    def partial_second_moment(self, upper: float, **quad_kwargs: Any) -> float:
        a, b = self.support
        if upper <= a:
            return 0.0
        if upper >= b:
            return self.integrate(lambda z: z * z, **quad_kwargs)
        qkw = dict(epsabs=1e-11, epsrel=1e-11, limit=300)
        qkw.update(quad_kwargs)
        return float(quad(lambda z: z * z * self.pdf_scalar(z), a, upper, **qkw)[0])

    def quantile(self, prob: float, **quad_kwargs: Any) -> float:
        a, b = self.support
        p = float(np.clip(prob, 0.0, 1.0))
        if p <= 0.0:
            return a
        if p >= 1.0:
            return b

        def f(x: float) -> float:
            return self.cdf(x, **quad_kwargs) - p

        return float(opt.brentq(f, a, b, xtol=1e-12, rtol=1e-11, maxiter=200))

    def gauss_legendre(self, n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
        a, b = self.support
        nodes, weights = np.polynomial.legendre.leggauss(int(n_nodes))
        xs = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        ws = 0.5 * (b - a) * weights * _pdf_values(self.pdf, xs)
        return xs, ws


@dataclass
class HighDimConfig:
    kappa: float
    tau: float = 1.0
    kappa_star: float = 1.0
    rho: float = 1e-2
    n_quad: int = 700

    def __post_init__(self) -> None:
        if not (0.0 < self.kappa < 1.0):
            raise ValueError("kappa must lie in (0,1).")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive.")
        if not (0.0 < self.kappa_star <= 1.0):
            raise ValueError("kappa_star must lie in (0,1].")
        if self.rho < 0.0:
            raise ValueError("rho must be nonnegative.")
        if self.n_quad < 32:
            raise ValueError("n_quad should be at least 32.")


def stable_q_r(
    x: ArrayLike,
    t: float,
    rho: float,
    branch: str,
    *,
    max_t_gamma: float = 350.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return q_t(gamma_x) and r_t^branch(gamma_x) from the paper.

    These are evaluated without rescaling, so they overflow in float64 once
    t * gamma exceeds about 355. Larger values raise an error instead of
    silently returning inf. The projector formulas use stable_log_r instead.
    """
    x = np.asarray(x, dtype=float)
    t = float(t)
    rho = float(rho)
    branch = branch.upper()
    if branch not in {"U", "V"}:
        raise ValueError("branch must be 'U' or 'V'.")

    gamma = np.sqrt(x * x + rho * rho)
    t_gamma = float(abs(t) * np.max(gamma)) if gamma.size else 0.0
    if t_gamma > max_t_gamma:
        raise OverflowError(
            f"stable_q_r: t * gamma = {t_gamma:.4g} > {max_t_gamma:g}, "
            "q_t and r_t would overflow in float64."
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(rho, gamma, out=np.ones_like(gamma), where=gamma > 0)
        u = 2.0 * t * gamma
        v = t * gamma
        q = 1.0 + ratio * np.sinh(u) + ratio * ratio * (np.cosh(u) - 1.0)
        if branch == "U":
            r = 2.0 * rho * (1.0 - ratio * ratio) * np.sinh(v) ** 2
        else:
            r = 2.0 * rho * (np.cosh(v) + ratio * np.sinh(v)) ** 2
    return np.maximum(q, 1.0), np.maximum(r, 0.0)


def _log_sinh(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 1e-8
    out[small] = np.log(np.maximum(x[small], np.finfo(float).tiny))
    xs = x[~small]
    out[~small] = xs - np.log(2.0) + np.log1p(-np.exp(-2.0 * xs))
    return out


def _log_cosh_plus_ratio_sinh(x: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    plus = 0.5 * (1.0 + ratio)
    minus = 0.5 * (1.0 - ratio)
    log_plus = np.log(np.maximum(plus, np.finfo(float).tiny)) + x
    log_minus = np.full_like(log_plus, -np.inf)
    mask = minus > 0.0
    log_minus[mask] = np.log(minus[mask]) - x[mask]
    return np.logaddexp(log_plus, log_minus)


def stable_log_r(
    x: ArrayLike,
    t: float,
    rho: float,
    branch: str,
    *,
    zero_rho: bool = False,
) -> np.ndarray:
    """Stable log of r_t^branch(gamma_x), allowing -inf when r_t=0."""
    x = np.asarray(x, dtype=float)
    t = float(t)
    branch = branch.upper()
    if branch not in {"U", "V"}:
        raise ValueError("branch must be 'U' or 'V'.")

    if zero_rho:
        u = np.maximum(t * x, 0.0)
        if branch == "U":
            out = 2.0 * _log_sinh(u)
            out[x <= 0.0] = -np.inf
            return out
        return 2.0 * _log_cosh_plus_ratio_sinh(u, np.zeros_like(u))

    rho = float(rho)
    if rho <= 0.0:
        return stable_log_r(x, t, rho, branch, zero_rho=True)
    gamma = np.sqrt(x * x + rho * rho)
    ratio = np.divide(rho, gamma, out=np.ones_like(gamma), where=gamma > 0.0)
    u = t * gamma
    if branch == "U":
        coeff = np.maximum(1.0 - ratio * ratio, 0.0)
        out = np.log(2.0 * rho) + np.log(np.maximum(coeff, np.finfo(float).tiny)) + 2.0 * _log_sinh(u)
        out[coeff <= 0.0] = -np.inf
        return out
    return np.log(2.0 * rho) + 2.0 * _log_cosh_plus_ratio_sinh(u, ratio)


class HighDimTheory:
    """Numerical evaluator for the high-dimensional formulas in the paper."""

    def __init__(self, law: SignalLaw, config: HighDimConfig):
        self.law = law
        self.config = config
        self.kappa = float(config.kappa)
        self.tau = float(config.tau)
        self.beta = float(min(1.0, self.tau))
        self.kappa_star = float(config.kappa_star)
        self.rho = float(config.rho)
        self.signal_weight = self.beta * self.kappa_star
        self.atom_zero_weight = max(0.0, 1.0 - self.signal_weight)
        self.x_nodes, phi_weights = law.gauss_legendre(config.n_quad)
        self.phi_weights = phi_weights
        self.tilde_weights = self.signal_weight * phi_weights
        self.x_star = self.compute_x_star()
        self.paper_x_star = 0.0 if self.kappa >= self.signal_weight else self.x_star

    @property
    def selected_mass(self) -> float:
        return min(self.kappa, self.signal_weight)

    def compute_x_star(self) -> float:
        a, b = self.law.support
        if self.kappa >= self.signal_weight:
            return a
        target_tail = self.kappa / self.signal_weight
        return self.law.quantile(1.0 - target_tail)

    def tilde_integral(self, values: np.ndarray, value_at_zero: complex | float) -> complex:
        return self.atom_zero_weight * value_at_zero + np.sum(self.tilde_weights * values)

    def tail_mask(self) -> np.ndarray:
        threshold = self.law.support[0] if self.kappa >= self.signal_weight else self.x_star
        return self.x_nodes > threshold

    def atom_masses(self, branch: str) -> Dict[str, float]:
        """Structural atoms of the plotted spectral measures.

        The density routines use these masses to remove Dirac peaks before
        plotting the regular part, which is compared with finite histograms
        without their exact zero modes and frozen 2*rho modes.
        """
        branch = branch.upper()
        selected = min(self.kappa, self.signal_weight)
        if branch == "U":
            zero = max(0.0, 1.0 - selected / self.tau)
            frozen = 0.0
            total = 1.0
        elif branch == "V":
            zero = max(0.0, 1.0 - self.kappa)
            frozen = max(0.0, self.kappa - self.signal_weight)
            total = 1.0
        elif branch == "Z":
            zero = max(0.0, 1.0 - selected / self.beta)
            frozen = 0.0
            total = 1.0
        else:
            raise ValueError("branch must be 'U', 'V', or 'Z'.")
        continuous = max(0.0, total - zero - frozen)
        return {
            "zero": float(zero),
            "frozen_2rho": float(frozen),
            "continuous": float(continuous),
        }

    # ------------------------------------------------------------------
    # One-point Stieltjes transforms and densities
    # ------------------------------------------------------------------

    def _self_consistency(self, g: complex, t: float, w: complex, branch: str) -> Tuple[complex, complex]:
        q, r = stable_q_r(self.x_nodes, t, self.rho, branch)
        q0, r0 = stable_q_r(np.array([0.0]), t, self.rho, branch)
        a = r - w * q
        a0 = complex(r0[0] - w * q0[0])
        den = 1.0 - self.kappa + self.kappa * g * a
        den0 = 1.0 - self.kappa + self.kappa * g * a0
        f = self.tilde_integral(1.0 / den, 1.0 / den0) - 1.0
        df = self.tilde_integral(-self.kappa * a / (den * den), -self.kappa * a0 / (den0 * den0))
        return complex(f), complex(df)

    def _initial_g_starts(self, t: float, w: complex, branch: str) -> list[complex]:
        """Build starts spanning the physical and long-time small-g scales."""
        q, r = stable_q_r(self.x_nodes, t, self.rho, branch)
        q0, r0 = stable_q_r(np.array([0.0]), t, self.rho, branch)
        a = r - w * q
        a_abs = np.abs(np.concatenate([np.asarray(a, dtype=np.complex128), np.array([r0[0] - w * q0[0]])]))
        finite = a_abs[np.isfinite(a_abs) & (a_abs > 0.0)]
        starts = [
            complex(1.0 / max(1e-8, 1.0 - self.kappa)),
            complex(1.0),
            complex(0.1),
            -1.0 / w,
            -1.0 / (self.kappa * w),
            -0.2 / (self.kappa * w),
            complex(1.0 / (self.kappa * (1.0 + abs(w)))),
        ]
        if finite.size:
            # At long times the physical branch can be exponentially small.
            # These starts put Newton near the scale set by r_t - w q_t.
            scales = [
                1.0 / (self.kappa * max(float(np.nanmax(finite)), np.finfo(float).tiny)),
                1.0 / (self.kappa * max(float(np.nanmedian(finite)), np.finfo(float).tiny)),
            ]
            direction = -1.0 / w
            direction /= max(abs(direction), np.finfo(float).tiny)
            for scale in scales:
                starts.extend([scale * direction, 0.1 * scale * direction, 10.0 * scale * direction])
        return starts

    @staticmethod
    def _is_physical_g(g: complex, w: complex, *, tol: float = 1e-8) -> bool:
        if not (np.isfinite(g.real) and np.isfinite(g.imag)):
            return False
        if abs(w.imag) <= tol:
            return g.real >= -tol and abs(g.imag) <= 1e-5
        # For the Stieltjes convention m(z)=int (x-z)^-1 dmu(x), the selected
        # auxiliary branch is Herglotz as well: Im g has the sign of Im w.
        return g.imag * w.imag >= -tol

    def _newton_from_start(
        self,
        start: complex,
        t: float,
        w: complex,
        branch: str,
        *,
        tol: float,
        max_iter: int,
    ) -> Tuple[complex, float, bool]:
        g = complex(start)
        best_g = g
        best_res = np.inf
        converged = False
        for _ in range(max_iter):
            f, df = self._self_consistency(g, t, w, branch)
            res = abs(f)
            if res < best_res:
                best_g, best_res = g, res
            if res < tol:
                converged = True
                break
            if abs(df) < 1e-14:
                break
            step = f / df
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.1, 0.03, 0.01):
                candidate = g - scale * step
                if not (np.isfinite(candidate.real) and np.isfinite(candidate.imag)):
                    continue
                f_candidate, _ = self._self_consistency(candidate, t, w, branch)
                if abs(f_candidate) <= (1.0 - 0.05 * scale) * res or abs(f_candidate) < res:
                    g = candidate
                    accepted = True
                    break
            if not accepted:
                g = g - 0.1 * step
        return best_g, best_res, converged

    def _choose_g_candidate(
        self,
        candidates: list[Tuple[complex, float]],
        w: complex,
        g0: Optional[complex],
    ) -> Optional[complex]:
        if not candidates:
            return None
        physical = [(g, res) for g, res in candidates if self._is_physical_g(g, w)]
        if not physical:
            # Never return a root with the wrong imaginary sign. The caller then
            # tries other solvers and raises an error if they also fail.
            return None
        pool = physical
        if g0 is not None and np.isfinite(g0.real) and np.isfinite(g0.imag):
            pool.sort(key=lambda item: (abs(item[0] - g0), item[1]))
        else:
            pool.sort(key=lambda item: (item[1], abs(item[0])))
        return pool[0][0]

    def solve_g(
        self,
        t: float,
        w: complex,
        branch: str,
        *,
        g0: Optional[complex] = None,
        tol: float = 1e-10,
        max_iter: int = 80,
    ) -> complex:
        """Solve the scalar self-consistent equation at one w.

        `g0`, if given, is tried first, then several other starting points.
        """
        if g0 is not None and np.isfinite(g0.real) and np.isfinite(g0.imag):
            g, res, converged = self._newton_from_start(
                complex(g0),
                t,
                w,
                branch,
                tol=tol,
                max_iter=max_iter,
            )
            if converged and self._is_physical_g(g, w):
                return g

        starts: list[complex] = []
        if g0 is not None and np.isfinite(g0.real) and np.isfinite(g0.imag):
            starts.append(complex(g0))
        starts.extend(self._initial_g_starts(t, w, branch))

        best_g = starts[0]
        best_res = np.inf
        candidates: list[Tuple[complex, float]] = []

        for start in starts:
            g, res, converged = self._newton_from_start(
                complex(start),
                t,
                w,
                branch,
                tol=tol,
                max_iter=max_iter,
            )
            if res < best_res:
                best_g, best_res = g, res
            if converged or res < max(tol, 1e-9):
                candidates.append((g, res))

        chosen = self._choose_g_candidate(candidates, w, g0)
        if chosen is not None:
            return chosen

        def fun(v: np.ndarray) -> np.ndarray:
            f, _ = self._self_consistency(complex(v[0], v[1]), t, w, branch)
            return np.array([f.real, f.imag], dtype=float)

        for start in starts:
            sol = opt.root(fun, np.array([start.real, start.imag], dtype=float), method="hybr", tol=tol)
            if sol.success:
                g = complex(sol.x[0], sol.x[1])
                f, _ = self._self_consistency(g, t, w, branch)
                res = abs(f)
                if res < best_res:
                    best_g, best_res = g, res
                if res < 1e-7:
                    candidates.append((g, res))

        chosen = self._choose_g_candidate(candidates, w, g0)
        if chosen is not None:
            return chosen

        if best_res < 1e-7 and self._is_physical_g(best_g, w, tol=1e-6):
            return best_g

        raise RuntimeError(
            f"Could not solve self-consistent equation for branch={branch}, t={t}, w={w}; "
            f"best residual={best_res:.3e}."
        )

    def _continuation_seed(
        self,
        t: float,
        target_w: complex,
        branch: str,
        *,
        steps: int = 48,
        verbose: bool = False,
        progress_every: int = 12,
    ) -> complex:
        """Follow the selected branch from the negative real axis to target_w."""
        left = -max(1.0, self.law.support[1] + 2.0 * self.rho + abs(target_w.real))
        anchor = complex(left, target_w.imag)
        if verbose:
            print(
                f"[density] branch {branch}, t={t:g}: continuation seed "
                f"from Re(w)={anchor.real:.3g} to {target_w.real:.3g} "
                f"({int(max(2, steps))} steps)",
                flush=True,
            )
        g = self.solve_g(t, anchor, branch)
        path = np.linspace(0.0, 1.0, int(max(2, steps)))[1:]
        for step_index, s in enumerate(path, start=1):
            w = anchor + s * (target_w - anchor)
            g = self.solve_g(t, complex(w), branch, g0=g)
            if verbose and (step_index % progress_every == 0 or step_index == len(path)):
                print(
                    f"[density] branch {branch}, t={t:g}: seed step "
                    f"{step_index}/{len(path)}",
                    flush=True,
                )
        return g

    def stieltjes(
        self,
        branch: str,
        t: float,
        w: complex,
        *,
        g0: Optional[complex] = None,
        return_g: bool = False,
    ) -> complex | Tuple[complex, complex]:
        branch = branch.upper()
        if branch not in {"U", "V"}:
            raise ValueError("branch must be 'U' or 'V'.")
        g = self.solve_g(t, w, branch, g0=g0)
        q, r = stable_q_r(self.x_nodes, t, self.rho, branch)
        q0, r0 = stable_q_r(np.array([0.0]), t, self.rho, branch)
        den = 1.0 - self.kappa + self.kappa * g * (r - w * q)
        den0 = 1.0 - self.kappa + self.kappa * g * (r0[0] - w * q0[0])
        integral = self.tilde_integral(q / den, q0[0] / den0)
        if branch == "U":
            m = -(1.0 - self.kappa / self.tau) / w + (self.kappa / self.tau) * g * integral
        else:
            m = -(1.0 - self.kappa) / w + self.kappa * g * integral
        if return_g:
            return complex(m), g
        return complex(m)

    def density(
        self,
        branch: str,
        energies: Sequence[float],
        t: float,
        *,
        eta: float = 2e-3,
        clip_negative: bool = True,
        remove_atoms: bool = True,
        normalize_continuous: bool = True,
        verbose: bool = False,
        progress_every: int = 100,
    ) -> Dict[str, np.ndarray]:
        branch = branch.upper()
        energies = np.asarray(energies, dtype=float)
        density = np.empty_like(energies)
        density_unnormalized = np.empty_like(energies)
        g_values = np.empty_like(energies, dtype=np.complex128)
        residuals = np.empty_like(energies, dtype=float)
        atoms = self.atom_masses(branch)
        mass = max(atoms["continuous"], np.finfo(float).tiny)
        g0: Optional[complex] = None
        if energies.size:
            if verbose:
                print(
                    f"[density] branch {branch}, t={t:g}: starting "
                    f"{energies.size} energies, eta={eta:g}",
                    flush=True,
                )
            g0 = self._continuation_seed(
                t,
                complex(float(energies[0]), float(eta)),
                branch,
                verbose=verbose,
                progress_every=max(1, progress_every // 8),
            )
        for i, e in enumerate(energies):
            w = complex(float(e), float(eta))
            m, g0 = self.stieltjes(branch, t, w, g0=g0, return_g=True)  # type: ignore[assignment]
            if remove_atoms:
                if atoms["zero"] > 0.0:
                    m -= atoms["zero"] / (0.0 - w)
                if branch == "V" and atoms["frozen_2rho"] > 0.0:
                    m -= atoms["frozen_2rho"] / (2.0 * self.rho - w)
            dens = np.imag(m) / np.pi
            if clip_negative:
                dens = max(0.0, dens)
            density_unnormalized[i] = dens
            density[i] = dens / mass if normalize_continuous and remove_atoms else dens
            g_values[i] = g0
            residuals[i] = abs(self._self_consistency(g0, t, w, branch)[0])
            if verbose and (
                i == 0 or (i + 1) % progress_every == 0 or i + 1 == energies.size
            ):
                print(
                    f"[density] branch {branch}, t={t:g}: energy "
                    f"{i + 1}/{energies.size}, residual={residuals[i]:.2e}",
                    flush=True,
                )
        return {
            "energies": energies,
            "density": density,
            "density_unnormalized": density_unnormalized,
            "g": g_values,
            "residual": residuals,
            "atom_zero": np.array(atoms["zero"]),
            "atom_2rho": np.array(atoms["frozen_2rho"]),
            "continuous_mass": np.array(atoms["continuous"]),
        }

    def density_grid(
        self,
        energies: Sequence[float],
        times: Sequence[float],
        *,
        branches: Iterable[str] = ("U", "V"),
        eta: float = 2e-3,
        include_prediction_from_u: bool = True,
        verbose: bool = False,
        progress_every: int = 100,
    ) -> Dict[str, Any]:
        energies = np.asarray(energies, dtype=float)
        times = np.asarray(times, dtype=float)
        out: Dict[str, Any] = {
            "energies": energies,
            "times": times,
            "eta": float(eta),
            "kappa": self.kappa,
            "tau": self.tau,
            "beta": self.beta,
            "kappa_star": self.kappa_star,
            "rho": self.rho,
            "x_star": self.paper_x_star,
        }
        for branch in branches:
            key = branch.upper()
            arr = np.empty((len(times), len(energies)), dtype=float)
            arr_raw = np.empty_like(arr)
            residual = np.empty_like(arr)
            for j, t in enumerate(times):
                if verbose:
                    print(
                        f"[density_grid] branch {key}: time {j + 1}/{len(times)} "
                        f"(t={float(t):g})",
                        flush=True,
                    )
                result = self.density(
                    key,
                    energies,
                    float(t),
                    eta=eta,
                    verbose=verbose,
                    progress_every=progress_every,
                )
                arr[j] = result["density"]
                arr_raw[j] = result["density_unnormalized"]
                residual[j] = result["residual"]
            out[f"density_{key}"] = arr
            out[f"density_{key}_unnormalized"] = arr_raw
            out[f"residual_{key}"] = residual
            out[f"atom_zero_{key}"] = self.atom_masses(key)["zero"]
            out[f"atom_2rho_{key}"] = self.atom_masses(key)["frozen_2rho"]
            out[f"continuous_mass_{key}"] = self.atom_masses(key)["continuous"]

        if include_prediction_from_u and "density_U_unnormalized" in out:
            density_z_raw = prediction_density_from_u(
                energies,
                energies,
                out["density_U_unnormalized"],
                tau=self.tau,
                beta=self.beta,
                rho=self.rho,
            )
            mass_z = max(self.atom_masses("Z")["continuous"], np.finfo(float).tiny)
            out["density_Z_unnormalized"] = density_z_raw
            out["density_Z"] = density_z_raw / mass_z
            out["atom_zero_Z"] = self.atom_masses("Z")["zero"]
            out["continuous_mass_Z"] = self.atom_masses("Z")["continuous"]
        return out

    # ------------------------------------------------------------------
    # Range projector distances
    # ------------------------------------------------------------------

    @staticmethod
    def _inv_one_minus_k_plus_logz(one_minus_k: float, log_z: np.ndarray) -> np.ndarray:
        log_z = np.asarray(log_z, dtype=float)
        out = np.empty_like(log_z)
        if one_minus_k == 0.0:
            return np.exp(-np.clip(log_z, -745.0, 745.0))
        large = log_z > 50.0
        out[large] = np.exp(-np.minimum(log_z[large], 745.0))
        z = np.exp(np.clip(log_z[~large], -745.0, 50.0))
        out[~large] = 1.0 / (one_minus_k + z)
        return out

    def _solve_log_positive_parameter(self, log_values: np.ndarray, log_value_at_zero: float) -> float:
        one_minus_k = 1.0 - self.kappa

        def equation(log_lam: float) -> float:
            vals = self._inv_one_minus_k_plus_logz(one_minus_k, np.log(self.kappa) + log_lam + log_values)
            if np.isneginf(log_value_at_zero):
                val0 = 1.0 / one_minus_k
            else:
                val0 = float(
                    self._inv_one_minus_k_plus_logz(
                        one_minus_k,
                        np.array([np.log(self.kappa) + log_lam + log_value_at_zero]),
                    )[0]
                )
            return float(np.real(self.tilde_integral(vals, val0)) - 1.0)

        lo, hi = -80.0, 80.0
        flo, fhi = equation(lo), equation(hi)
        while flo < 0.0 and lo > -5000.0:
            lo -= 40.0
            flo = equation(lo)
        while fhi > 0.0 and hi < 5000.0:
            hi += 40.0
            fhi = equation(hi)
        if flo * fhi > 0.0:
            raise RuntimeError(
                "Could not bracket the positive self-consistent parameter "
                f"(f({lo})={flo}, f({hi})={fhi})."
            )
        return float(opt.brentq(equation, lo, hi, xtol=1e-13, rtol=1e-11, maxiter=300))

    def projector_delta(
        self,
        branch: str,
        t: float,
        *,
        rho: Optional[float] = None,
        zero_rho: bool = False,
    ) -> float:
        branch = branch.upper()
        if branch not in {"U", "V"}:
            raise ValueError("branch must be 'U' or 'V'.")
        if branch == "U" and float(t) <= 0.0:
            return float(self.selected_mass)
        if branch == "U" and self.kappa > self.signal_weight:
            return 0.0

        rho_eff = self.rho if rho is None else float(rho)
        log_values = stable_log_r(self.x_nodes, t, rho_eff, branch, zero_rho=zero_rho)
        log_value0 = float(stable_log_r(np.array([0.0]), t, rho_eff, branch, zero_rho=zero_rho)[0])
        log_lam = self._solve_log_positive_parameter(log_values, log_value0)
        inv_den = self._inv_one_minus_k_plus_logz(1.0 - self.kappa, np.log(self.kappa) + log_lam + log_values)
        mask = self.tail_mask()
        return float(2.0 * (1.0 - self.kappa) * np.sum(self.tilde_weights[mask] * inv_den[mask]))

    def projector_curves(
        self,
        times: Sequence[float],
        *,
        rhos: Optional[Sequence[float]] = None,
        include_zero_rho: bool = True,
    ) -> Dict[str, Any]:
        times = np.asarray(times, dtype=float)
        out: Dict[str, Any] = {
            "times": times,
            "kappa": self.kappa,
            "tau": self.tau,
            "beta": self.beta,
            "kappa_star": self.kappa_star,
            "rho": self.rho,
            "x_star": self.paper_x_star,
        }

        if include_zero_rho:
            for branch in ("U", "V"):
                out[f"delta0_{branch}"] = np.array(
                    [self.projector_delta(branch, float(t), zero_rho=True) for t in times],
                    dtype=float,
                )

        if rhos is not None:
            rhos_arr = np.asarray(rhos, dtype=float)
            out["rhos"] = rhos_arr
            for branch in ("U", "V"):
                curves = np.empty((len(rhos_arr), len(times)), dtype=float)
                for i, rho in enumerate(rhos_arr):
                    for j, t in enumerate(times):
                        curves[i, j] = self.projector_delta(branch, float(t), rho=float(rho))
                out[f"delta_{branch}"] = curves
        return out

    # ------------------------------------------------------------------
    # Closed-form theorem curves for the small-initialization figures
    # ------------------------------------------------------------------

    def logtime_spectral_front(self, s_values: Sequence[float], x_grid: Sequence[float]) -> Dict[str, np.ndarray]:
        """Small-rho limits on the timescale t = s log(1/rho).

        "loss_limit" is normalized by min(p, d), as in Theorem 13. Compare
        it with the simulator's "loss" (divided by min(p, d)), not "loss_per_d".
        The two agree only when tau >= 1.
        """
        s_values = np.asarray(s_values, dtype=float)
        x_grid = np.asarray(x_grid, dtype=float)
        density_z = np.zeros((len(s_values), len(x_grid)), dtype=float)
        atom_zero = np.zeros(len(s_values), dtype=float)
        learned_mass = np.zeros(len(s_values), dtype=float)
        wasserstein = np.zeros(len(s_values), dtype=float)
        threshold = np.full(len(s_values), np.inf, dtype=float)

        phi = _pdf_values(self.law.pdf, x_grid)
        for i, s in enumerate(s_values):
            if s <= 0:
                atom_zero[i] = 1.0
                wasserstein[i] = self.kappa_star * self.law.partial_second_moment(np.inf)
                continue
            xs = 1.0 / (2.0 * s)
            xs_clipped = max(xs, self.paper_x_star)
            threshold[i] = xs
            tail = self.law.tail(xs_clipped)
            learned_mass[i] = self.kappa_star * tail
            atom_zero[i] = 1.0 - learned_mass[i]
            density_z[i] = self.kappa_star * phi * (x_grid > xs_clipped)
            wasserstein[i] = self.kappa_star * self.law.partial_second_moment(xs_clipped)

        return {
            "s": s_values,
            "x": x_grid,
            "density_Z": density_z,
            "atom_zero": atom_zero,
            "learned_mass": learned_mass,
            "M_C": self.beta * learned_mass,
            "wasserstein2": wasserstein,
            "loss_limit": wasserstein,
            "threshold": threshold,
            "x_star": np.array(self.paper_x_star),
        }

    def projector_asymptotic_constants(self) -> Dict[str, float]:
        x0 = self.law.support[0]
        out = {"x0": float(x0), "x_star": float(self.paper_x_star)}
        if self.kappa < self.signal_weight:
            out["under_t_delta_limit"] = float(self.signal_weight * self.law.pdf_scalar(self.x_star) * np.log(2.0))
        return out


def prediction_density_from_u(
    z_grid: Sequence[float],
    u_grid: Sequence[float],
    density_u: np.ndarray,
    *,
    tau: float,
    beta: float,
    rho: float,
) -> np.ndarray:
    """Approximate the regular density of mu_t^Z from a grid for mu_t^U.

    This applies the pushforward theta_rho(lambda)=sqrt(lambda(lambda+2rho))
    to the absolutely continuous part represented on `u_grid`.  Atoms at zero
    are intentionally not drawn into the returned density.
    """
    z_grid = np.asarray(z_grid, dtype=float)
    u_grid = np.asarray(u_grid, dtype=float)
    dens = np.asarray(density_u, dtype=float)
    squeeze = False
    if dens.ndim == 1:
        dens = dens[None, :]
        squeeze = True
    out = np.zeros((dens.shape[0], len(z_grid)), dtype=float)
    inv = np.sqrt(rho * rho + z_grid * z_grid) - rho
    jac = np.divide(z_grid, np.sqrt(rho * rho + z_grid * z_grid), out=np.zeros_like(z_grid), where=z_grid > 0)
    for i in range(dens.shape[0]):
        interp = np.interp(inv, u_grid, dens[i], left=0.0, right=0.0)
        out[i] = (tau / beta) * interp * jac
    return out[0] if squeeze else out


def make_theory(
    law: SignalLaw,
    *,
    kappa: float = 0.35,
    tau: float = 1.0,
    kappa_star: float = 1.0,
    rho: float = 1e-2,
    n_quad: int = 700,
) -> HighDimTheory:
    return HighDimTheory(law, HighDimConfig(kappa=kappa, tau=tau, kappa_star=kappa_star, rho=rho, n_quad=n_quad))
