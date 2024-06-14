# Copyright (c) 2023-2024 Salah Berra and contributors
# Distributed under the the GNU General Public License (See accompanying file
# LICENSE or copy at https://www.gnu.org/licenses/)

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
    H: NDArray = np.random.normal(0, 1.0 / math.sqrt(n), (n, m))
    A = np.dot(H, H.T)
    eig = np.linalg.eig(A)[0]  # Eigenvalues

    Wt = torch.Tensor(np.diag(eig)).to(device)  # Define the appropriate 'device'
    Ht = torch.from_numpy(H).float().to(device)  # Define the appropriate 'device'

    print(
        f"""
    - Condition number of A: {np.max(eig) / np.min(eig)}
    - Min eigenvalue of A: {np.min(eig)}
    - Max eigenvalue of A: {np.max(eig)}"""
    )

    solution = torch.normal(torch.zeros(bs, n), 1.0).to(device).detach()
    y = solution @ Ht.detach()

    return A, Ht, Wt, solution, y


def decompose_matrix(
    A: NDArray | torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Decompose a given matrix into its diagonal, lower triangular, upper
    triangular components and their inverses.

    Args:
      A: Input square matrix to decompose.

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
    D = np.diag(np.diag(A))  # Diagonal matrix
    L = np.tril(A, -1)  # Lower triangular matrix
    U = np.triu(A, 1)  # Upper triangular matrix
    Dinv = np.linalg.inv(D)  # Inverse of the diagonal matrix
    invM = np.linalg.inv(D + L)  # Inverse of the matrix (D + L)

    # Convert to Torch tensors and move to device
    At = torch.Tensor(A).to(device)
    Dt = torch.Tensor(D).to(device)
    Lt = torch.Tensor(L).to(device)
    Ut = torch.Tensor(U).to(device)
    Dtinv = torch.Tensor(Dinv).to(device)
    Mtinv = torch.Tensor(invM).to(device)

    return At, Dt, Lt, Ut, Dtinv, Mtinv
