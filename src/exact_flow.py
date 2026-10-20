"""
Reference solutions of the gradient flow, used by validation/validation01_exact_flow.py.

With whitened inputs (Sigma = I), the flow of the paper reads
    dU/dt = (C - U V^T) V,        dV/dt = (C - U V^T)^T U.
This module gives two references for it, in float64 on the CPU:

- `rk4_flow`: classical fourth-order Runge-Kutta. With a step of order 1e-3 its
  error is far below the first-order error of the explicit Euler simulator.
- `closed_form_prediction`: the closed form of Proposition 1,
      W W^T = e^{tA} W0 (I + W0^T F(t) W0)^{-1} W0^T e^{tA},  F(t) = int_0^t e^{2sA} ds,
  with W = (U; V) and A = [[-rho I, C], [C^T, rho I]]. It is exact, but the
  m x m system becomes ill-conditioned once rho e^{2 t sigma_max} is large, so it
  is only evaluated while its condition number stays below `cond_max`.

All linear algebra uses PyTorch, like the simulator, so that NumPy and PyTorch
do not load two OpenMP runtimes in one process. Inputs may be NumPy arrays or
tensors on any device; outputs are float64 CPU tensors (plain floats for scalars).
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch


DTYPE = torch.float64
DEVICE = torch.device("cpu")


def as_tensor(a) -> torch.Tensor:
    """float64 CPU tensor holding `a` (NumPy array, tensor or nested sequence)."""
    return torch.as_tensor(a).to(device=DEVICE, dtype=DTYPE)


def gradient_field(U: torch.Tensor, V: torch.Tensor, C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-hand side of the gradient flow for Sigma = I."""
    E = C - U @ V.mT
    return E @ V, E.mT @ U


def rk4_step(U: torch.Tensor, V: torch.Tensor, C: torch.Tensor, h: float) -> Tuple[torch.Tensor, torch.Tensor]:
    k1u, k1v = gradient_field(U, V, C)
    k2u, k2v = gradient_field(U + 0.5 * h * k1u, V + 0.5 * h * k1v, C)
    k3u, k3v = gradient_field(U + 0.5 * h * k2u, V + 0.5 * h * k2v, C)
    k4u, k4v = gradient_field(U + h * k3u, V + h * k3v, C)
    U_new = U + (h / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
    V_new = V + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return U_new, V_new


def rk4_flow(
    C,
    U0,
    V0,
    save_steps: Sequence[int],
    step: float,
    substeps: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrate with RK4 of step `step / substeps` and return U, V at times `save_steps * step`.

    `save_steps` are the integer step indices of the simulator, so the RK4 states
    are taken exactly at the simulator's save times. The outputs have shapes
    (len(save_steps), p, m) and (len(save_steps), d, m).
    """
    steps = [int(s) for s in save_steps]
    if any(b < a for a, b in zip(steps, steps[1:])) or (steps and steps[0] < 0):
        raise ValueError("save_steps must be nonnegative and sorted.")
    h = float(step) / int(substeps)
    C = as_tensor(C)
    U = as_tensor(U0).clone()
    V = as_tensor(V0).clone()
    out_u = torch.empty((len(steps), *U.shape), device=DEVICE, dtype=DTYPE)
    out_v = torch.empty((len(steps), *V.shape), device=DEVICE, dtype=DTYPE)
    current = 0
    for i, target in enumerate(steps):
        for _ in range((target - current) * int(substeps)):
            U, V = rk4_step(U, V, C, h)
        current = target
        out_u[i] = U
        out_v[i] = V
    return out_u, out_v


def oja_matrix(C: torch.Tensor, rho: float) -> torch.Tensor:
    """The matrix A_rho of Proposition 1."""
    p, d = C.shape
    A = torch.zeros((p + d, p + d), device=DEVICE, dtype=DTYPE)
    A[:p, :p] = -rho * torch.eye(p, device=DEVICE, dtype=DTYPE)
    A[:p, p:] = C
    A[p:, :p] = C.mT
    A[p:, p:] = rho * torch.eye(d, device=DEVICE, dtype=DTYPE)
    return A


def closed_form_prediction(
    C,
    R,
    rho: float,
    times: Sequence[float],
    *,
    cond_max: float = 1e10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prediction Z(t) = U(t) V(t)^T from the closed form, for U0 = 0 and V0 = sqrt(2 rho) R.

    Returns (Z, cond): Z has shape (len(times), p, d) and is NaN at the times where
    the condition number `cond` of the m x m system exceeds `cond_max` (cond is
    inf where the system is not even finite).
    """
    C = as_tensor(C)
    R = as_tensor(R)
    rho = float(rho)
    p, d = C.shape
    m = R.shape[1]
    lam, Q = torch.linalg.eigh(oja_matrix(C, rho))
    W0 = torch.cat([torch.zeros((p, m), device=DEVICE, dtype=DTYPE), math.sqrt(2.0 * rho) * R], dim=0)
    B = Q.mT @ W0
    times = [float(t) for t in times]
    Z = torch.full((len(times), p, d), float("nan"), device=DEVICE, dtype=DTYPE)
    cond = torch.full((len(times),), float("inf"), device=DEVICE, dtype=DTYPE)
    small = lam.abs() < 1e-14
    safe_lam = torch.where(small, torch.ones_like(lam), lam)
    eye = torch.eye(m, device=DEVICE, dtype=DTYPE)
    for i, t in enumerate(times):
        # F(t) in the eigenbasis of A: (e^{2 t lambda} - 1) / (2 lambda), and t when lambda = 0.
        f = torch.where(small, torch.full_like(lam, t), torch.expm1(2.0 * t * lam) / (2.0 * safe_lam))
        M = eye + B.mT @ (f[:, None] * B)
        if not bool(torch.isfinite(M).all()):
            continue
        cond[i] = torch.linalg.cond(M)
        if not math.isfinite(float(cond[i])) or float(cond[i]) > cond_max:
            continue
        G = torch.exp(t * lam)[:, None] * B
        WWt = Q @ (G @ torch.linalg.solve(M, G.mT)) @ Q.mT
        Z[i] = WWt[:p, p:]
    return Z, cond


def relative_frobenius(A, B) -> float:
    """||A - B||_F / ||B||_F, NaN when B = 0 or when either matrix is not finite."""
    A = as_tensor(A)
    B = as_tensor(B)
    if not (bool(torch.isfinite(A).all()) and bool(torch.isfinite(B).all())):
        return float("nan")
    norm_b = float(torch.linalg.norm(B))
    if norm_b == 0.0:
        return float("nan")
    return float(torch.linalg.norm(A - B)) / norm_b
