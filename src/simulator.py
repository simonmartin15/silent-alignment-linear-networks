"""
Finite-dimensional gradient-descent simulations for the linear-network figures.

The simulator intentionally stores more than any single figure needs:
losses, spectra of U U^T and V V^T, singular values of Z=U V^T,
range-projector distances, top-eigenspace distances, and imbalance errors.
`PrescribedSpectrumSimulator` uses given singular values of C instead of
sampled ones (Figure 6 and validation 6).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import json
import math

import numpy as np
import torch

from .laws import DEFAULT_SUPPORT

ArrayLike = np.ndarray


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower().replace("torch.", "")
    if key in {"float32", "single", "fp32"}:
        return torch.float32
    if key in {"float64", "double", "fp64"}:
        return torch.float64
    raise ValueError("dtype must be 'float32' or 'float64'.")


def _dtype_name(dtype: str | torch.dtype) -> str:
    return "float64" if _torch_dtype(dtype) == torch.float64 else "float32"


def _as_pdf_values(pdf: Callable[[ArrayLike], ArrayLike], x: np.ndarray) -> np.ndarray:
    y = pdf(x)
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    y = np.asarray(y, dtype=float)
    if y.shape == ():
        y = np.full_like(x, float(y))
    return y


class DensitySampler1D:
    """Inverse-CDF sampler for a compactly supported one-dimensional density."""

    def __init__(
        self,
        pdf: Callable[[ArrayLike], ArrayLike],
        support: Tuple[float, float],
        *,
        grid_size: int = 20000,
        normalize: bool = True,
        eps: float = 1e-14,
    ):
        a, b = map(float, support)
        if not (np.isfinite(a) and np.isfinite(b) and 0.0 <= a < b):
            raise ValueError("support must be a finite nonnegative interval (a,b).")
        self.pdf = pdf
        self.support = (a, b)
        self.grid_size = int(grid_size)
        self.eps = float(eps)

        x = np.linspace(a, b, self.grid_size)
        y = np.maximum(_as_pdf_values(pdf, x), 0.0)
        increments = 0.5 * (y[:-1] + y[1:]) * np.diff(x)
        cdf = np.zeros_like(x)
        cdf[1:] = np.cumsum(increments)
        total = cdf[-1]
        if not np.isfinite(total) or total <= eps:
            raise ValueError("density has near-zero mass on the chosen support.")
        if normalize:
            cdf /= total
        elif abs(total - 1.0) > 5e-4:
            raise ValueError(f"density should integrate to one; got {total:.8g}.")
        cdf[0] = 0.0
        cdf[-1] = 1.0
        self.x_grid = x
        self.cdf_grid = np.maximum.accumulate(cdf)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        u = rng.random(int(n))
        return np.interp(u, self.cdf_grid, self.x_grid)


def sample_stiefel(
    n: int,
    m: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if m > n:
        raise ValueError("Stiefel sampling requires m <= n.")
    a = torch.randn((n, m), generator=generator, device=device, dtype=dtype)
    q, r = torch.linalg.qr(a, mode="reduced")
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1.0
    return q * signs.unsqueeze(0)


def projector_from_basis(q: torch.Tensor) -> torch.Tensor:
    if q.numel() == 0 or q.shape[-1] == 0:
        return torch.zeros((q.shape[0], q.shape[0]), device=q.device, dtype=q.dtype)
    return q @ q.mT


def range_projector(a: torch.Tensor, *, rtol: float = 1e-7) -> torch.Tensor:
    if a.numel() == 0:
        return torch.zeros((a.shape[0], a.shape[0]), device=a.device, dtype=a.dtype)
    u, s, _ = torch.linalg.svd(a, full_matrices=False)
    if s.numel() == 0 or float(s[0].detach().cpu()) == 0.0:
        rank = 0
    else:
        rank = int((s > rtol * s[0]).sum().item())
    return projector_from_basis(u[:, :rank])


def top_projector_from_factor(a: torch.Tensor, k: int) -> torch.Tensor:
    k = int(k)
    if k <= 0:
        return torch.zeros((a.shape[0], a.shape[0]), device=a.device, dtype=a.dtype)
    u, _, _ = torch.linalg.svd(a, full_matrices=False)
    k = min(k, u.shape[1])
    return projector_from_basis(u[:, :k])


def frozen_projector_e0(C: torch.Tensor, R: torch.Tensor, *, tol: float = 1e-7) -> torch.Tensor:
    """Projector onto {0} x (Im(R) cap Ker(C)) in R^{p+d}."""
    p, d = C.shape
    m = R.shape[1]
    out = torch.zeros((d, d), device=C.device, dtype=C.dtype)
    cr = C @ R
    _, s, vh = torch.linalg.svd(cr, full_matrices=True)
    if s.numel() == 0:
        rank = 0
    else:
        threshold = tol * max(1.0, float(s.max().detach().cpu()))
        rank = int((s > threshold).sum().item())
    null_dim = m - rank
    if null_dim <= 0:
        return out
    coeff = vh.mT[:, rank:]
    bottom = R @ coeff
    bottom, _ = torch.linalg.qr(bottom, mode="reduced")
    return projector_from_basis(bottom)


@dataclass
class FiniteSimulationConfig:
    d: int
    kappa: float
    tau: float = 1.0
    kappa_star: float = 1.0
    rhos: Tuple[float, ...] = (1e-2,)
    eta: float = 1e-2
    t_max: float = 10.0
    num_save: int = 60
    save_times: Optional[Tuple[float, ...]] = None
    save_grid: str = "log"
    reps: int = 1
    top_eig_fracs: Tuple[float, ...] = (0.05, 0.10, 0.20)
    seed: int = 0
    dtype: str = "float32"
    device: Optional[str] = None
    random_teacher_basis: bool = False
    same_teacher_across_rho: bool = True
    same_init_across_rho: bool = True
    density_grid_size: int = 20000
    normalize_phi: bool = True
    store_uv_spectra: bool = True
    store_prediction_spectrum: bool = True
    store_uv_range_projectors: bool = True
    store_uv_top_projectors: bool = True
    store_prediction_projectors: bool = True
    store_prediction_top_projectors: bool = True
    store_imbalance: bool = True
    promote_projectors_to_float64: bool = True
    progress: bool = True

    def stores_projectors(self) -> bool:
        return (
            self.store_uv_range_projectors
            or self.store_uv_top_projectors
            or self.store_prediction_projectors
            or self.store_prediction_top_projectors
        )

    def __post_init__(self) -> None:
        if self.d <= 0:
            raise ValueError("d must be positive.")
        if not (0.0 < self.kappa <= 1.0):
            raise ValueError("kappa must lie in (0,1].")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive.")
        if not (0.0 < self.kappa_star <= 1.0):
            raise ValueError("kappa_star must lie in (0,1].")
        if self.eta <= 0.0 or self.t_max < 0.0:
            raise ValueError("Need eta > 0 and t_max >= 0.")
        self.rhos = tuple(float(r) for r in self.rhos)
        if any(r <= 0.0 for r in self.rhos):
            raise ValueError("all rhos must be positive.")
        self.top_eig_fracs = tuple(float(a) for a in self.top_eig_fracs)
        if self.save_times is not None:
            self.save_times = tuple(float(t) for t in self.save_times)
        self.save_grid = str(self.save_grid).lower()
        if self.save_grid not in {"log", "linear"}:
            raise ValueError("save_grid must be 'log' or 'linear'.")
        self.dtype = _dtype_name(self.dtype)
        if self.promote_projectors_to_float64 and self.stores_projectors():
            # Projector distances depend on SVD subspaces and are noticeably
            # more stable in double precision near small singular values.
            self.dtype = "float64"


@dataclass
class FiniteSimulationResult:
    config: Dict[str, Any]
    dimensions: Dict[str, int]
    rhos: np.ndarray
    times: np.ndarray
    top_eig_fracs: np.ndarray
    teacher_singular_values: np.ndarray
    loss: np.ndarray
    loss_per_d: np.ndarray
    eigvals_u: np.ndarray
    eigvals_v: np.ndarray
    singular_values_z: np.ndarray
    singular_values_z_inf: np.ndarray
    prediction_convergence_error: np.ndarray
    prediction_convergence_error_per_d: np.ndarray
    range_delta_u: np.ndarray
    range_delta_v: np.ndarray
    prediction_left_delta: np.ndarray
    prediction_right_delta: np.ndarray
    top_delta_u: np.ndarray
    top_delta_v: np.ndarray
    prediction_top_left_delta: np.ndarray
    prediction_top_right_delta: np.ndarray
    imbalance_error: np.ndarray

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config_json": np.array(json.dumps(self.config)),
            "dimensions_json": np.array(json.dumps(self.dimensions)),
            "rhos": self.rhos,
            "times": self.times,
            "top_eig_fracs": self.top_eig_fracs,
            "teacher_singular_values": self.teacher_singular_values,
            "loss": self.loss,
            "loss_per_d": self.loss_per_d,
            "eigvals_u": self.eigvals_u,
            "eigvals_v": self.eigvals_v,
            "singular_values_z": self.singular_values_z,
            "singular_values_z_inf": self.singular_values_z_inf,
            "prediction_convergence_error": self.prediction_convergence_error,
            "prediction_convergence_error_per_d": self.prediction_convergence_error_per_d,
            "range_delta_u": self.range_delta_u,
            "range_delta_v": self.range_delta_v,
            "prediction_left_delta": self.prediction_left_delta,
            "prediction_right_delta": self.prediction_right_delta,
            "top_delta_u": self.top_delta_u,
            "top_delta_v": self.top_delta_v,
            "prediction_top_left_delta": self.prediction_top_left_delta,
            "prediction_top_right_delta": self.prediction_top_right_delta,
            "imbalance_error": self.imbalance_error,
        }
        np.savez_compressed(path, **payload)


class FiniteLinearNetworkSimulator:
    """Explicit-Euler simulator for the gradient flow in the paper."""

    def __init__(
        self,
        config: FiniteSimulationConfig,
        phi_c: Callable[[ArrayLike], ArrayLike],
        support_phi: Tuple[float, float],
    ):
        self.config = config
        self.dtype = _torch_dtype(config.dtype)
        if config.device is None:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = config.device
        self.device = torch.device(device_name)

        self.d = int(config.d)
        self.p = int(round(config.tau * config.d))
        self.m = int(round(config.kappa * config.d))
        self.min_pd = min(self.p, self.d)
        self.beta = min(1.0, config.tau)
        self.m_star = min(int(round(config.kappa_star * self.min_pd)), self.min_pd)
        self.selected_rank = min(self.m, self.m_star)
        self.n_steps = int(round(config.t_max / config.eta))
        self.times, self.save_steps = self._make_save_grid()
        self.sampler = DensitySampler1D(
            phi_c,
            support_phi,
            grid_size=config.density_grid_size,
            normalize=config.normalize_phi,
        )

    def _make_save_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        if cfg.save_times is None:
            if cfg.num_save <= 1:
                times = np.array([0.0, cfg.t_max], dtype=float)
            elif cfg.save_grid == "log":
                raw = np.logspace(0.0, np.log10(max(1, self.n_steps)), cfg.num_save) - 1.0
                steps = np.unique(np.clip(np.rint(raw).astype(int), 0, self.n_steps))
                steps = np.unique(np.concatenate([np.array([0, self.n_steps], dtype=int), steps]))
                times = steps.astype(float) * cfg.eta
            else:
                times = np.linspace(0.0, cfg.t_max, cfg.num_save)
        else:
            times = np.array(cfg.save_times, dtype=float)
            if np.any(times < -1e-12) or np.any(times > cfg.t_max + cfg.eta):
                raise ValueError("save_times must lie in [0,t_max].")
        steps = np.unique(np.clip(np.rint(times / cfg.eta).astype(int), 0, self.n_steps))
        times = steps.astype(float) * cfg.eta
        return times, steps

    def _rng(self, offset: int) -> np.random.Generator:
        return np.random.default_rng(self.config.seed + offset)

    def _torch_generator(self, offset: int) -> torch.Generator:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(self.config.seed + offset))
        return gen

    def _sample_teacher(self, rep: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        rng = self._rng(10000 + rep)
        values = np.sort(self.sampler.sample(self.m_star, rng))[::-1].copy()
        s = torch.as_tensor(values, device=self.device, dtype=self.dtype)
        if self.config.random_teacher_basis:
            gen_l = self._torch_generator(20000 + rep)
            gen_r = self._torch_generator(30000 + rep)
            left = sample_stiefel(self.p, self.m_star, generator=gen_l, device=self.device, dtype=self.dtype)
            right = sample_stiefel(self.d, self.m_star, generator=gen_r, device=self.device, dtype=self.dtype)
            C = (left * s.unsqueeze(0)) @ right.mT
        else:
            C = torch.zeros((self.p, self.d), device=self.device, dtype=self.dtype)
            idx = torch.arange(self.m_star, device=self.device)
            C[idx, idx] = s
            left = torch.zeros((self.p, self.m_star), device=self.device, dtype=self.dtype)
            right = torch.zeros((self.d, self.m_star), device=self.device, dtype=self.dtype)
            left[idx, idx] = 1.0
            right[idx, idx] = 1.0
        return C, left, right, values

    def _sample_init(self, rep: int, rho_index: int) -> torch.Tensor:
        if self.config.same_init_across_rho:
            offset = 40000 + rep
        else:
            offset = 40000 + 1000 * rho_index + rep
        return sample_stiefel(self.d, self.m, generator=self._torch_generator(offset), device=self.device, dtype=self.dtype)

    def _teacher_projectors(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        R: torch.Tensor,
        C: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        rank = self.selected_rank
        p_signal_u = projector_from_basis(left[:, :rank])
        p_signal_v = projector_from_basis(right[:, :rank])
        p_inf_u = p_signal_u
        p_inf_v = p_signal_v
        if self.m > self.m_star:
            p_inf_v = p_inf_v + frozen_projector_e0(C, R)
        top_u: Dict[int, torch.Tensor] = {}
        top_v: Dict[int, torch.Tensor] = {}
        for frac in self.config.top_eig_fracs:
            k = int(math.floor(frac * self.d))
            if 0 < k <= self.m_star and k <= self.p and k <= self.d:
                top_u[k] = projector_from_basis(left[:, :k])
                top_v[k] = projector_from_basis(right[:, :k])
        return p_inf_u, p_inf_v, p_signal_u, p_signal_v, top_u, top_v

    def _full_eigvals_from_factor(self, a: torch.Tensor, ambient_dim: int) -> np.ndarray:
        vals = torch.linalg.svdvals(a).square()
        vals, _ = torch.sort(vals, descending=True)
        out = torch.zeros(ambient_dim, device=self.device, dtype=self.dtype)
        n = min(ambient_dim, vals.numel())
        out[:n] = vals[:n]
        return out.detach().cpu().numpy()

    def _singular_values_z(self, z: torch.Tensor) -> np.ndarray:
        vals = torch.linalg.svdvals(z)
        vals, _ = torch.sort(vals, descending=True)
        out = torch.zeros(self.min_pd, device=self.device, dtype=self.dtype)
        n = min(self.min_pd, vals.numel())
        out[:n] = vals[:n]
        return out.detach().cpu().numpy()


    def _prediction_limit(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        singulars: np.ndarray,
    ) -> torch.Tensor:
        """Limit Z_inf = sum_{i <= min(m,m*)} sigma_i u_i v_i^T."""
        rank = self.selected_rank
        if rank <= 0:
            return torch.zeros((self.p, self.d), device=self.device, dtype=self.dtype)

        s = torch.as_tensor(
            singulars[:rank],
            device=self.device,
            dtype=self.dtype,
        )
        return (left[:, :rank] * s.unsqueeze(0)) @ right[:, :rank].mT


    def _z_svd_observables(
        self,
        z: torch.Tensor,
        p_signal_u: torch.Tensor,
        p_signal_v: torch.Tensor,
        top_teacher_u: Dict[int, torch.Tensor],
        top_teacher_v: Dict[int, torch.Tensor],
        *,
        need_spectrum: bool,
        need_range: bool,
        need_top: bool,
    ) -> Dict[str, Any]:
        singular_values = np.full(self.min_pd, np.nan, dtype=float)
        prediction_left_delta = np.nan
        prediction_right_delta = np.nan
        prediction_top_left = np.full(len(self.config.top_eig_fracs), np.nan, dtype=float)
        prediction_top_right = np.full(len(self.config.top_eig_fracs), np.nan, dtype=float)

        if not (need_spectrum or need_range or need_top):
            return {
                "singular_values_z": singular_values,
                "prediction_left_delta": prediction_left_delta,
                "prediction_right_delta": prediction_right_delta,
                "prediction_top_left_delta": prediction_top_left,
                "prediction_top_right_delta": prediction_top_right,
            }

        left_z, s_z, vh_z = torch.linalg.svd(z, full_matrices=False)
        right_z = vh_z.mT
        if need_spectrum:
            n = min(self.min_pd, s_z.numel())
            singular_values[:n] = s_z[:n].detach().cpu().numpy()

        has_nonzero_prediction = s_z.numel() > 0 and float(s_z[0].detach().cpu()) > 0.0
        if not has_nonzero_prediction:
            return {
                "singular_values_z": singular_values,
                "prediction_left_delta": prediction_left_delta,
                "prediction_right_delta": prediction_right_delta,
                "prediction_top_left_delta": prediction_top_left,
                "prediction_top_right_delta": prediction_top_right,
            }

        if need_range and self.selected_rank > 0:
            rank = min(self.selected_rank, left_z.shape[1], right_z.shape[1])
            pz_left = projector_from_basis(left_z[:, :rank])
            pz_right = projector_from_basis(right_z[:, :rank])
            prediction_left_delta = float((torch.sum((pz_left - p_signal_u) ** 2) / float(self.d)).detach().cpu())
            prediction_right_delta = float((torch.sum((pz_right - p_signal_v) ** 2) / float(self.d)).detach().cpu())

        if need_top:
            for j, frac in enumerate(self.config.top_eig_fracs):
                k = int(math.floor(frac * self.d))
                if k in top_teacher_u and k <= left_z.shape[1] and k <= right_z.shape[1]:
                    pz_left = projector_from_basis(left_z[:, :k])
                    pz_right = projector_from_basis(right_z[:, :k])
                    prediction_top_left[j] = float((torch.sum((pz_left - top_teacher_u[k]) ** 2) / max(1, k)).detach().cpu())
                    prediction_top_right[j] = float((torch.sum((pz_right - top_teacher_v[k]) ** 2) / max(1, k)).detach().cpu())

        return {
            "singular_values_z": singular_values,
            "prediction_left_delta": prediction_left_delta,
            "prediction_right_delta": prediction_right_delta,
            "prediction_top_left_delta": prediction_top_left,
            "prediction_top_right_delta": prediction_top_right,
        }

    def _save_observables(
        self,
        U: torch.Tensor,
        V: torch.Tensor,
        Z: torch.Tensor,
        C: torch.Tensor,
        Z_inf: torch.Tensor,
        rho: float,
        p_inf_u: torch.Tensor,
        p_inf_v: torch.Tensor,
        p_signal_u: torch.Tensor,
        p_signal_v: torch.Tensor,
        top_teacher_u: Dict[int, torch.Tensor],
        top_teacher_v: Dict[int, torch.Tensor],
    ) -> Dict[str, Any]:
        cfg = self.config

        # Distance to C: this is the empirical prediction loss proxy.
        diff_to_c = Z - C
        loss_min = torch.sum(diff_to_c * diff_to_c) / float(self.min_pd)
        loss_d = torch.sum(diff_to_c * diff_to_c) / float(self.d)

        # Distance to the actual convergence point Z_inf.
        # This differs from distance to C in the underparameterized case.
        diff_to_inf = Z - Z_inf
        prediction_convergence_error = torch.sum(diff_to_inf * diff_to_inf) / float(self.min_pd)
        prediction_convergence_error_per_d = torch.sum(diff_to_inf * diff_to_inf) / float(self.d)

        eigvals_u = np.full(self.p, np.nan, dtype=float)
        eigvals_v = np.full(self.d, np.nan, dtype=float)
        if cfg.store_uv_spectra:
            eigvals_u = self._full_eigvals_from_factor(U, self.p)
            eigvals_v = self._full_eigvals_from_factor(V, self.d)

        delta_u = np.nan
        delta_v = np.nan
        if cfg.store_uv_range_projectors:
            proj_u = range_projector(U)
            proj_v = range_projector(V)
            delta_u = float((torch.sum((proj_u - p_inf_u) ** 2) / float(self.d)).detach().cpu())
            delta_v = float((torch.sum((proj_v - p_inf_v) ** 2) / float(self.d)).detach().cpu())

        imbalance_error = np.nan
        if cfg.store_imbalance:
            eye_m = torch.eye(self.m, device=self.device, dtype=self.dtype)
            imbalance = V.mT @ V - U.mT @ U - 2.0 * rho * eye_m
            denom = torch.linalg.norm(2.0 * rho * eye_m).clamp_min(torch.finfo(self.dtype).tiny)
            imbalance_error = float((torch.linalg.norm(imbalance) / denom).detach().cpu())

        top_u = np.full(len(cfg.top_eig_fracs), np.nan, dtype=float)
        top_v = np.full(len(cfg.top_eig_fracs), np.nan, dtype=float)
        if cfg.store_uv_top_projectors:
            for j, frac in enumerate(cfg.top_eig_fracs):
                k = int(math.floor(frac * self.d))
                if k in top_teacher_u:
                    pu = top_projector_from_factor(U, k)
                    pv = top_projector_from_factor(V, k)
                    top_u[j] = float((torch.sum((pu - top_teacher_u[k]) ** 2) / max(1, k)).detach().cpu())
                    top_v[j] = float((torch.sum((pv - top_teacher_v[k]) ** 2) / max(1, k)).detach().cpu())

        z_obs = self._z_svd_observables(
            Z,
            p_signal_u,
            p_signal_v,
            top_teacher_u,
            top_teacher_v,
            need_spectrum=cfg.store_prediction_spectrum,
            need_range=cfg.store_prediction_projectors,
            need_top=cfg.store_prediction_top_projectors,
        )

        return {
            "loss": float(loss_min.detach().cpu()),
            "loss_per_d": float(loss_d.detach().cpu()),
            "prediction_convergence_error": float(prediction_convergence_error.detach().cpu()),
            "prediction_convergence_error_per_d": float(prediction_convergence_error_per_d.detach().cpu()),
            "eigvals_u": eigvals_u,
            "eigvals_v": eigvals_v,
            "singular_values_z": z_obs["singular_values_z"],
            "range_delta_u": delta_u,
            "range_delta_v": delta_v,
            "prediction_left_delta": z_obs["prediction_left_delta"],
            "prediction_right_delta": z_obs["prediction_right_delta"],
            "top_delta_u": top_u,
            "top_delta_v": top_v,
            "prediction_top_left_delta": z_obs["prediction_top_left_delta"],
            "prediction_top_right_delta": z_obs["prediction_top_right_delta"],
            "imbalance_error": imbalance_error,
        }

    def run(self) -> FiniteSimulationResult:
        cfg = self.config
        n_rho = len(cfg.rhos)
        n_time = len(self.times)
        n_alpha = len(cfg.top_eig_fracs)

        teacher_singular_values = np.empty((cfg.reps, self.m_star), dtype=float)
        loss = np.empty((n_rho, n_time, cfg.reps), dtype=float)
        loss_per_d = np.empty_like(loss)
        eigvals_u = np.empty((n_rho, n_time, cfg.reps, self.p), dtype=float)
        eigvals_v = np.empty((n_rho, n_time, cfg.reps, self.d), dtype=float)
        singular_values_z = np.empty((n_rho, n_time, cfg.reps, self.min_pd), dtype=float)
        singular_values_z_inf = np.zeros((cfg.reps, self.min_pd), dtype=float)
        prediction_convergence_error = np.empty_like(loss)
        prediction_convergence_error_per_d = np.empty_like(loss)
        range_delta_u = np.empty_like(loss)
        range_delta_v = np.empty_like(loss)
        prediction_left_delta = np.empty_like(loss)
        prediction_right_delta = np.empty_like(loss)
        top_delta_u = np.empty((n_rho, n_time, n_alpha, cfg.reps), dtype=float)
        top_delta_v = np.empty_like(top_delta_u)
        prediction_top_left_delta = np.empty_like(top_delta_u)
        prediction_top_right_delta = np.empty_like(top_delta_u)
        imbalance_error = np.empty_like(loss)
        need_teacher_projectors = cfg.stores_projectors()

        for rep in range(cfg.reps):
            if cfg.same_teacher_across_rho:
                C_base, left_base, right_base, singulars = self._sample_teacher(rep)
            else:
                C_base = left_base = right_base = None  # type: ignore[assignment]
                singulars = np.zeros(self.m_star)
            teacher_singular_values[rep] = singulars
            singular_values_z_inf[rep, : self.selected_rank] = singulars[: self.selected_rank]

            for rho_index, rho in enumerate(cfg.rhos):
                if not cfg.same_teacher_across_rho:
                    C_base, left_base, right_base, singulars = self._sample_teacher(1000 * rho_index + rep)
                    teacher_singular_values[rep] = singulars
                    singular_values_z_inf[rep, :] = 0.0
                    singular_values_z_inf[rep, : self.selected_rank] = singulars[: self.selected_rank]


                assert C_base is not None and left_base is not None and right_base is not None
                C = C_base
                Z_inf = self._prediction_limit(left_base, right_base, singulars)
                R = self._sample_init(rep, rho_index)
                U = torch.zeros((self.p, self.m), device=self.device, dtype=self.dtype)
                V = math.sqrt(2.0 * rho) * R

                if need_teacher_projectors:
                    p_inf_u, p_inf_v, p_signal_u, p_signal_v, top_teacher_u, top_teacher_v = self._teacher_projectors(left_base, right_base, R, C)
                else:
                    empty_projector = torch.empty((0, 0), device=self.device, dtype=self.dtype)
                    p_inf_u = p_inf_v = p_signal_u = p_signal_v = empty_projector
                    top_teacher_u = {}
                    top_teacher_v = {}
                save_lookup = {int(step): i for i, step in enumerate(self.save_steps)}

                for step in range(self.n_steps + 1):
                    Z = None
                    if step in save_lookup:
                        i_time = save_lookup[step]
                        Z = U @ V.mT
                        obs = self._save_observables(
                            U,
                            V,
                            Z,
                            C,
                            Z_inf,
                            rho,
                            p_inf_u,
                            p_inf_v,
                            p_signal_u,
                            p_signal_v,
                            top_teacher_u,
                            top_teacher_v,
                        )
                        loss[rho_index, i_time, rep] = obs["loss"]
                        loss_per_d[rho_index, i_time, rep] = obs["loss_per_d"]
                        prediction_convergence_error[rho_index, i_time, rep] = obs["prediction_convergence_error"]
                        prediction_convergence_error_per_d[rho_index, i_time, rep] = obs["prediction_convergence_error_per_d"]
                        eigvals_u[rho_index, i_time, rep] = obs["eigvals_u"]
                        eigvals_v[rho_index, i_time, rep] = obs["eigvals_v"]
                        singular_values_z[rho_index, i_time, rep] = obs["singular_values_z"]
                        range_delta_u[rho_index, i_time, rep] = obs["range_delta_u"]
                        range_delta_v[rho_index, i_time, rep] = obs["range_delta_v"]
                        prediction_left_delta[rho_index, i_time, rep] = obs["prediction_left_delta"]
                        prediction_right_delta[rho_index, i_time, rep] = obs["prediction_right_delta"]
                        top_delta_u[rho_index, i_time, :, rep] = obs["top_delta_u"]
                        top_delta_v[rho_index, i_time, :, rep] = obs["top_delta_v"]
                        prediction_top_left_delta[rho_index, i_time, :, rep] = obs["prediction_top_left_delta"]
                        prediction_top_right_delta[rho_index, i_time, :, rep] = obs["prediction_top_right_delta"]
                        imbalance_error[rho_index, i_time, rep] = obs["imbalance_error"]

                    if step == self.n_steps:
                        break

                    if Z is None:
                        Z = U @ V.mT
                    U_new = U + cfg.eta * ((C - Z) @ V)
                    V_new = V + cfg.eta * ((C.mT - Z.mT) @ U)
                    U, V = U_new, V_new

                    if cfg.progress and (step + 1) % max(1, self.n_steps // 20) == 0:
                        print(
                            f"\rrep {rep + 1}/{cfg.reps}, rho {rho_index + 1}/{n_rho}, "
                            f"step {step + 1}/{self.n_steps}",
                            end="",
                            flush=True,
                        )
                if cfg.progress:
                    print()

        dims = {
            "d": self.d,
            "p": self.p,
            "m": self.m,
            "min_pd": self.min_pd,
            "m_star": self.m_star,
            "selected_rank": self.selected_rank,
        }
        return FiniteSimulationResult(
            config=asdict(cfg),
            dimensions=dims,
            rhos=np.asarray(cfg.rhos, dtype=float),
            times=self.times,
            top_eig_fracs=np.asarray(cfg.top_eig_fracs, dtype=float),
            teacher_singular_values=teacher_singular_values,
            loss=loss,
            loss_per_d=loss_per_d,
            eigvals_u=eigvals_u,
            eigvals_v=eigvals_v,
            singular_values_z=singular_values_z,
            singular_values_z_inf=singular_values_z_inf,
            prediction_convergence_error=prediction_convergence_error,
            prediction_convergence_error_per_d=prediction_convergence_error_per_d,
            range_delta_u=range_delta_u,
            range_delta_v=range_delta_v,
            prediction_left_delta=prediction_left_delta,
            prediction_right_delta=prediction_right_delta,
            top_delta_u=top_delta_u,
            top_delta_v=top_delta_v,
            prediction_top_left_delta=prediction_top_left_delta,
            prediction_top_right_delta=prediction_top_right_delta,
            imbalance_error=imbalance_error,
        )


def run_simulation(
    config: FiniteSimulationConfig,
    phi_c: Callable[[ArrayLike], ArrayLike],
    support_phi: Tuple[float, float] = DEFAULT_SUPPORT,
) -> FiniteSimulationResult:
    return FiniteLinearNetworkSimulator(config, phi_c, support_phi).run()


class PrescribedSpectrumSimulator(FiniteLinearNetworkSimulator):
    """Finite simulator with deterministic teacher singular values."""

    def __init__(
        self,
        config: FiniteSimulationConfig,
        singular_values: np.ndarray,
        density_pdf: Callable[[ArrayLike], np.ndarray],
    ) -> None:
        super().__init__(config, density_pdf, DEFAULT_SUPPORT)
        values = np.asarray(singular_values, dtype=float)
        if values.shape != (self.m_star,):
            raise ValueError(f"Expected {self.m_star} teacher singular values; got {values.shape}.")
        if np.any(np.diff(values) > 0.0) or np.any(values <= 0.0):
            raise ValueError("Teacher singular values must be positive and decreasing.")
        self.prescribed_singular_values = values.copy()

    def _sample_teacher(self, rep: int):
        del rep
        values = self.prescribed_singular_values.copy()
        singular_values = torch.as_tensor(values, device=self.device, dtype=self.dtype)
        C = torch.zeros((self.p, self.d), device=self.device, dtype=self.dtype)
        left = torch.zeros((self.p, self.m_star), device=self.device, dtype=self.dtype)
        right = torch.zeros((self.d, self.m_star), device=self.device, dtype=self.dtype)
        indices = torch.arange(self.m_star, device=self.device)
        C[indices, indices] = singular_values
        left[indices, indices] = 1.0
        right[indices, indices] = 1.0
        return C, left, right, values
