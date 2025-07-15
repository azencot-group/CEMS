"""cems_sampling.py –cems_methodSampling utilities for CEMS
=================================================
This module contains the math used by CEMS data augmentation:

* **Intrinsic‑dimension aware tangent sampling** via `get_batch_cems`.
"""
from __future__ import annotations

# ‑‑‑ Standard Library ‑‑‑
import math
from typing import List, Optional, Tuple

# ‑‑‑ Third‑party ‑‑‑
import numpy as np
import torch
from intrinsic_dimension import intrinsic_dimension  # type: ignore


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _adjust_dims(
        x: torch.Tensor,
        y: torch.Tensor,
        xk: Optional[torch.Tensor] = None,
        yk: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
    """Flatten *x* and ensure *y* is 2‑D, then concatenate.*

    Parameters
    ----------
    x, y : torch.Tensor
        Anchor sample and label batch, shaped *(b, …)* and *(b,)* or *(b, 1)*.
    xk, yk : torch.Tensor | None, optional
        Neighbourhood tensors for sampling (same leading dims as *x* / *y*).

    Returns
    -------
    x_flat : torch.Tensor
        Batch flattened to *(b, m)*.
    zi : torch.Tensor
        Concatenation of *x_flat* and *y* → *(b, m + 1)*.
    zk : torch.Tensor | None
        Same concatenation for the neighbour batch.
    m : int
        Number of flattened feature dimensions in *x*.
    """
    if x.ndim > 2:  # collapse everything but batch
        x = x.reshape(x.shape[0], -1)
    if y.ndim == 1:
        y = y.reshape(y.shape[0], 1)

    m = x.shape[-1]
    zi = torch.cat((x, y), dim=-1)

    zk: Optional[torch.Tensor] = None
    if xk is not None and yk is not None:
        if xk.ndim > 3:
            xk = xk.reshape(xk.shape[0], xk.shape[1], -1)
        zk = torch.cat((xk, yk), dim=-1)

    return x, zi, zk, m


# ---------------------------------------------------------------------------
# Core CEMS routines
# ---------------------------------------------------------------------------

def get_batch_cems(
        args,
        x: torch.Tensor,
        y: torch.Tensor,
        xk: Optional[torch.Tensor] = None,
        yk: Optional[torch.Tensor] = None,
        *,
        scaler=None,
        latent: bool = False,
):
    """Generate a CEMS sampled batch.

    This function wraps the full CEMS pipeline:
        1. Flatten tensors & concat features/labels.
        2. Estimate (global or per‑batch) intrinsic dimension *d*.
        3. Compute basis, gradient, Hessian at each anchor sample.
        4. Sample in the estimated tangent bundle.
        5. Reshape everything back to the original input shapes.
    """
    # Store original shapes for later reshape
    x_shape, y_shape = x.shape, y.shape

    x_flat, zi, zk, m = _adjust_dims(x, y, xk, yk)

    d = args.id
    if latent:
        with torch.no_grad():
            d = intrinsic_dimension(zi)

    # Sanity‑check *d* for numerical stability
    if d < 1 or d >= zi.shape[-1] or d >= zi.shape[-2]:
        d = min(args.id, zi.shape[-1] - 1, zi.shape[-2] - 1)

    basis, grad, hess, u, u_prev_d, x_mean = _estimate_grad_hessian(args, zi, zk, d)
    z_sampled = _sample_tangent(args, zi, u, u_prev_d, x_mean, basis, grad, hess)

    x_new, y_new = z_sampled[..., :m], z_sampled[..., m:]
    x_new = x_new.reshape(x_shape)
    y_new = y_new.reshape(y_shape)

    return x_new, y_new


# ---------------------------------------------------------------------------
# Linear‑algebra helpers
# ---------------------------------------------------------------------------

def _get_projection(args, x: torch.Tensor, xk: Optional[torch.Tensor]):
    """Project *x* (and optionally *xk*) onto a local basis via SVD."""
    x_c = x.transpose(-2, -1)
    x_c_mean: Optional[torch.Tensor] = None

    if args.cems_method == 1:
        x_c_mean = torch.mean(x_c, -1)
        x_c = x_c - x_c_mean.unsqueeze(-1)
    else:
        assert xk is not None, "`xk` required when cems_method ≠ 1"
        xk_t = xk.transpose(-1, -2)
        x = x.unsqueeze(-1)
        x_c = xk_t - x

    basis,_,_ = torch.linalg.svd(x_c - x_c_mean.unsqueeze(-1) if x_c_mean is not None else x_c, full_matrices=False,
                             driver="gesvd")

    u = basis.transpose(-2, -1) @ x_c
    u_prev = u.transpose(-2, -1)

    if args.cems_method == 1:
        # Build pairwise differences
        u_t = u.transpose(-1, -2)
        u = (u_t.unsqueeze(1) - u_t).transpose(-1, -2)
        n = x.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
        u = -u.transpose(-1, -2)[mask].reshape((u.shape[0], u.shape[2] - 1, u.shape[1])).transpose(-1, -2)
    elif args.cems_method == 2:
        u = u.unsqueeze(0)

    return basis, u, u_prev, x_c_mean


def _estimate_grad_hessian(
        args,
        x: torch.Tensor,
        xk: Optional[torch.Tensor],
        d: int,
):
    """Local gradient & Hessian estimation via ridge regression."""
    tidx = torch.triu_indices(d, d)
    ones_mult = torch.ones((d, d), device=x.device)
    ones_mult.fill_diagonal_(0.5)

    basis, u, u_prev, x_mean = _get_projection(args, x, xk)
    u_d = u[:, :d]
    f = u[:, d:].transpose(-2, -1)  # residuals

    uu = torch.einsum('bki,bkj->bkij', u_d.transpose(-2, -1), u_d.transpose(-2, -1))
    uu = uu * ones_mult
    uu = uu[:, :, tidx[0], tidx[1]].transpose(-2, -1)

    psi = torch.cat((u_d, uu), dim=1).transpose(-2, -1)

    lam = torch.linalg.norm(psi, dim=(-1, -2)).mean()
    b = _solve_ridge_regression(psi, f, lam=lam).transpose(-2, -1)

    gradient = b[..., :d]
    hessian = torch.zeros((u.shape[0], b.shape[1], d, d), dtype=b.dtype, device=b.device)
    hessian[..., tidx[0], tidx[1]] = b[..., d:]
    hessian[..., tidx[1], tidx[0]] = b[..., d:]

    return basis, gradient, hessian, u_d, u_prev, x_mean


def _sample_tangent(
        args,
        x: torch.Tensor,
        u_k_d: torch.Tensor,
        u_prev: torch.Tensor,
        x_mean: Optional[torch.Tensor],
        basis: torch.Tensor,
        grad: torch.Tensor,
        hess: torch.Tensor,
):
    """Sample a point in the tangent bundle and map it back to the manifold."""
    d = grad.shape[-1]
    nu = torch.distributions.Normal(0, args.sigma).sample((x.shape[0], d, 1)).to(x.device)

    # First‑order term
    f_nu = (grad @ nu).squeeze(-1)

    # Second‑order term
    nu_ex = nu.unsqueeze(1)
    f_nu += 0.5 * (nu_ex.transpose(-1, -2) @ hess @ nu_ex).squeeze((-1, -2))

    x_zero = nu.squeeze(-1)
    x_new_local = torch.cat((x_zero, f_nu), dim=-1)

    if args.cems_method == 1:
        x_new_local += u_prev

    x_cems = (basis @ x_new_local.unsqueeze(-1)).squeeze(-1)
    x_cems += (x_mean if args.cems_method == 1 else x)

    return x_cems



def _solve_ridge_regression(a: torch.Tensor, b: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Solve `(A^T A + λI)X = A^T B` for *X* (ridge/Tikhonov)."""
    n = a.shape[-1]
    eye = torch.eye(n, device=a.device, dtype=a.dtype)
    a_t = a.transpose(-2, -1)
    a_reg = a_t @ a + lam * eye

    return torch.linalg.inv(a_reg) @ a_t @ b


