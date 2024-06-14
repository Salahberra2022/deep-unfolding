# Copyright (c) 2023-2024 Salah Berra and contributors
# Distributed under the the GNU General Public License (See accompanying file
# LICENSE or copy at https://www.gnu.org/licenses/)

"""Utility functions."""

from __future__ import annotations

import math

import numpy as np
import torch
from numpy.typing import NDArray

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU, if not CPU
"""The device where training will take place."""

print(f"Code run on : {device}")


def generate_A_H_sol(
    n: int = 300,
    m: int = 600,
    seed: int = 12,
    bs: int = 10,
    device: torch.device = device,
) -> tuple[NDArray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate matrices $A$, $H$, and $W$, as well as the solution and $y$.

    Args:
      n: Number of rows.
      m: Number of columns.
      seed: Seed for random number generation.
      bs: Batch size.
      device: Device to run the computations on.

    Returns:
      A tuple with the following contents:
        - Matrix $A$ (square matrix) of shape (`n`, `n`)
        - Matrix $H$ (random matrix) of shape (`n`, `m`)
        - Matrix $W$ with diagonal eigenvalues of $A$ of shape (`n`, `n`)
        - Solution tensor of shape (`bs`, `n`)
        - Tensor $y$ resulting from `solution @ H` of shape (`bs`, `m`)
    """
    np.random.seed(seed=seed)
    h = np.random.normal(0, 1.0 / math.sqrt(n), (n, m))
    a = np.dot(h, h.T)
    eig = np.linalg.eig(a)[0]  # Eigenvalues

    wt = torch.Tensor(np.diag(eig)).to(device)  # Define the appropriate 'device'
    ht = torch.from_numpy(h).float().to(device)  # Define the appropriate 'device'

    print(
        f"""
    - Condition number of A: {np.max(eig) / np.min(eig)}
    - Min eigenvalue of A: {np.min(eig)}
    - Max eigenvalue of A: {np.max(eig)}"""
    )

    solution = torch.normal(torch.zeros(bs, n), 1.0).to(device).detach()
    y = solution @ ht.detach()

    return a, ht, wt, solution, y


def decompose_matrix(
    a: NDArray | torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Decompose a given matrix into its diagonal, lower triangular, upper
    triangular components and their inverses.

    Args:
      a: Input square matrix to decompose.

    Returns:
      A tuple with the following contents:
        - $A$: Original matrix converted to a torch tensor.
        - $D$: Diagonal matrix of $A$.
        - $L$: Lower triangular matrix of $A$.
        - $U$: Upper triangular matrix of $A$.
        - $D^{-1}$: Inverse of the diagonal matrix $D$.
        - $M^{-1}$: Inverse of the matrix $D + L$.
    """
    # Decomposed matrix calculations
    d = np.diag(np.diag(a))  # Diagonal matrix
    l = np.tril(a, -1)  # Lower triangular matrix
    u = np.triu(a, 1)  # Upper triangular matrix
    d_inv = np.linalg.inv(d)  # Inverse of the diagonal matrix
    m_inv = np.linalg.inv(d + l)  # Inverse of the matrix (D + L)

    # Convert to Torch tensors and move to device
    at = torch.Tensor(a).to(device)
    dt = torch.Tensor(d).to(device)
    lt = torch.Tensor(l).to(device)
    ut = torch.Tensor(u).to(device)
    dt_inv = torch.Tensor(d_inv).to(device)
    mt_inv = torch.Tensor(m_inv).to(device)

    return at, dt, lt, ut, dt_inv, mt_inv
