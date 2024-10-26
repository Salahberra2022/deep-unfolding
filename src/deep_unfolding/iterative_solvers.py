# Copyright (c) 2023-2024 Salah Berra and contributors
# Distributed under the the GNU General Public License (See accompanying file
# LICENSE or copy at https://www.gnu.org/licenses/)

"""Conventional iterative methods."""

from abc import ABC, abstractmethod

import torch
from numpy.typing import NDArray
from torch import Tensor

from .utils import _decompose_matrix, _device


class IterativeModel(ABC):
    """Base model class for matrix decomposition and initialization."""

    _n: int
    """Dimension of the solution."""

    _H: Tensor
    """Random matrix $H$."""

    _bs: int
    """Batch size."""

    _y: Tensor
    """Solution tensor."""

    _A: Tensor
    """Original matrix converted to a torch tensor."""

    _D: Tensor
    """Diagonal matrix of $A$."""

    _L: Tensor
    """Lower triangular matrix of $A$."""

    _U: Tensor
    """Upper triangular matrix of $A$."""

    _Dinv: Tensor
    """Inverse of the diagonal matrix $D$."""

    _Minv: Tensor
    """Inverse of the matrix $D + L$."""

    _device: torch.device
    """Device where to run the model."""

    _solved: bool
    """Flag indicating whether the problem has been solved yet."""

    _s_hats: list[Tensor]
    """Solutions obtained through the several solver iterations."""

    @property
    def s_hat(self) -> Tensor:
        """Final solution, obtained after all the solver iterations."""
        if not self._solved:
            raise RuntimeError("Problem has not been solved yet!")
        return self._s_hats[-1]

    def __init__(
        self,
        n: int,
        a: NDArray,
        h: Tensor,
        bs: int,
        y: Tensor,
        device: torch.device = _device,
    ):
        """Initialize the base_model with the given parameters and decompose matrix $A$.

        Args:

          n: Dimension of the solution.
          a: Input square matrix to decompose.
          h: Random matrix $H$.
          bs: Batch size.
          y: Solution tensor.
          device: Device to run the model on ('cpu' or 'cuda').
        """
        self._n = n
        self._H = h
        self._bs = bs
        self._y = y
        self._device = device
        self._solved = False

        self._s_hats = []

        self._A, self._D, self._L, self._U, self._Dinv, self._Minv = _decompose_matrix(
            a, device
        )

    @abstractmethod
    def _iterate(self, num_itr: int, yMF: Tensor, s: Tensor) -> None:
        """Performs the low-level solver iterations according to the concrete model.

        Args:
          num_itr: The number of iterations to perform.
          yMF: The `y` matrix multiplied by the `H` matrix transpose.
          s: Initial solution.
        """
        pass

    def solve(
        self,
        iters: int = 25,
    ) -> list[Tensor]:
        """Solve the linear problem.

        Args:
          iters: Number of iterations to perform.

        Returns:
          The solution tensors through the several iterations.
        """
        if self._solved:
            raise RuntimeError("Problem has already been solved!")

        self._solved = True

        s = torch.zeros(self._bs, self._n).to(self._device)
        self._s_hats.append(s)

        yMF = torch.matmul(self._y, self._H.T)

        # Generate batch initial solution vector
        s = torch.matmul(yMF, self._Dinv)

        # Delegate the actual iterations to the concrete solver method
        self._iterate(iters, yMF, s)

        return self._s_hats

    def _evaluate(
        self,
        iter: int,
        solution: Tensor,
        device: torch.device = _device,
    ) -> float:
        """Low-level evaluation of the method's solution.

        Args:
          iter: Evaluate the solution at this iteration.
          solution: The actual solution to the problem.

        Returns:
          The MSE between the real solution and the solution found by the method
            after `iter` iterations.
        """
        return (
            torch.norm(solution.to(device) - self._s_hats[iter].to(device)) ** 2
        ).item() / (self._n * self._bs)

    def evaluate_final(
        self,
        solution: Tensor,
    ) -> float:
        """Evaluate the final solution found by the method.

        Args:
          solution: The actual solution to the problem.

        Returns:
          The MSE between the real solution and the final solution found by the
            method.
        """
        if not self._solved:
            raise RuntimeError("Problem has not been solved yet!")

        return self._evaluate(-1, solution, self._device)

    def evaluate_all(
        self,
        solution: Tensor,
    ) -> list[float]:
        """Evaluate solutions found by the method during all iterations.

        Args:
          solution: The actual solution to the problem.

        Returns:
          A list containing the MSE between the real solution and the solutions
            found by the method during the iterative process.
        """
        if not self._solved:
            raise RuntimeError("Problem has not been solved yet!")

        return [self._evaluate(i, solution, self._device) for i in range(len(self._s_hats))]


class GaussSeidel(IterativeModel):
    """The Gauss-Seidel algorithm for solving a linear system."""

    def __init__(
        self,
        n: int,
        a: NDArray,
        h: Tensor,
        bs: int,
        y: Tensor,
        device: torch.device = _device,
    ):
        """Initialize the Gauss-Seidel solver.

        Args:
          n: Dimension of the solution.
          a: Input square matrix to decompose.
          h: Random matrix H.
          bs: Batch size.
          y: Solution tensor.
          device: Device to run the model on ('cpu' or 'cuda').
        """
        super().__init__(n, a, h, bs, y, device)

    def _iterate(self, num_itr: int, yMF: Tensor, s: Tensor) -> None:
        for _ in range(num_itr):
            temp = -torch.matmul(s, self._U) + yMF
            s = torch.matmul(temp, self._Minv)
            self._s_hats.append(s)


class Richardson(IterativeModel):
    """The Richardson iteration algorithm for solving a linear system."""

    omega: Tensor
    """TODO explain what omega is."""

    def __init__(
        self,
        n: int,
        a: NDArray,
        h: Tensor,
        bs: int,
        y: Tensor,
        omega: float = 0.25,
        device: torch.device = _device,
    ):
        """Initialize the Richardson iteration solver.

        Args:
          n: Dimension of the solution.
          a: Input square matrix to decompose.
          h: Random matrix $H$.
          bs: Batch size.
          y: Solution tensor.
          omega: TODO explain.
          device: Device where to run the model.
        """
        super().__init__(n, a, h, bs, y, device)

        self.omega = torch.tensor(omega)

    def _iterate(self, num_itr: int, yMF: Tensor, s: Tensor) -> None:

        for _ in range(num_itr):
            s = s + torch.mul(self.omega, (yMF - torch.matmul(s, self._A)))
            self._s_hats.append(s)


class Jacobi(IterativeModel):
    """The Jacobi iteration algorithm for solving a linear system."""

    omega: Tensor
    """Relaxation parameter for Jacobi iterations."""

    def __init__(
        self,
        n: int,
        a: NDArray,
        h: Tensor,
        bs: int,
        y: Tensor,
        omega: float = 0.2,
        device: torch.device = _device,
    ):
        """Initialize the Jacobi iteration solver.

        Args:
          n: Dimension of the solution.
          a: Input square matrix to decompose.
          h: Random matrix $H$.
          bs: Batch size.
          y: Solution tensor.
          omega: Relaxation parameter for Jacobi iterations.
          device: Device where to run the model.
        """
        super().__init__(n, a, h, bs, y, device)
        self.omega = torch.tensor(omega)

    def _iterate(self, num_itr: int, yMF: Tensor, s: Tensor) -> None:

        for _ in range(num_itr):
            temp = torch.matmul(self._Dinv, (self._D - self._A))
            s = torch.matmul(s, temp) + torch.matmul(yMF, self._Dinv)
            self._s_hats.append(s)


class SOR(IterativeModel):
    """The Successive Over-Relaxation (SOR) algorithm for solving a linear system."""

    omega: Tensor
    """Relaxation parameter for SOR iterations."""

    def __init__(
        self,
        n: int,
        a: NDArray,
        h: Tensor,
        bs: int,
        y: Tensor,
        omega: float = 1.8,
        device: torch.device = _device,
    ):
        """Initialize the SOR solver.

        Args:
          n: Dimension of the solution.
          A: Input square matrix to decompose.
          H: Random matrix $H$.
          bs: Batch size.
          y: Solution tensor.
          omega: Relaxation parameter for SOR iterations.
          device: Device where to run the model.
        """
        super().__init__(n, a, h, bs, y, device)
        self.omega = torch.tensor(omega)

    def _iterate(self, num_itr: int, yMF: Tensor, s: Tensor) -> None:

        inv_omega = torch.div(1, self.omega)
        m_inv_sor = torch.linalg.inv(self._D - torch.mul(inv_omega, self._L))

        for _ in range(num_itr):
            temp = torch.mul((inv_omega - 1), self._D) + torch.mul(inv_omega, self._U)
            s = torch.matmul(s, torch.matmul(m_inv_sor, temp)) + torch.matmul(
                yMF, m_inv_sor
            )
            self._s_hats.append(s)


class SORCheby(IterativeModel):
    """The SOR-Chebyshev algorithm for solving a linear system."""

    omega: Tensor
    """Relaxation parameter for SOR iterations."""

    omegaa: Tensor
    """Acceleration parameter for SOR-Chebyshev iterations."""

    gamma: Tensor
    """Damping factor for SOR-Chebyshev iterations."""

    def __init__(
        self,
        n: int,
        a: NDArray,
        h: Tensor,
        bs: int,
        y: Tensor,
        omega: float = 1.8,
        omegaa: float = 0.8,
        gamma: float = 0.8,
        device: torch.device = _device,
    ):
        """Initialize the SOR-Chebyshev solver.

        Args:
          n: Dimension of the solution.
          a: Input square matrix to decompose.
          h: Random matrix $H$.
          bs: Batch size.
          y: Solution tensor.
          omega: Relaxation parameter for SOR iterations.
          omegaa: Acceleration parameter for SOR-Chebyshev iterations.
          gamma: Damping factor for SOR-Chebyshev iterations.
          device: Device where to run the model.
        """
        super().__init__(n, a, h, bs, y, device)
        self.omega = torch.tensor(omega)
        self.omegaa = torch.tensor(omegaa)
        self.gamma = torch.tensor(gamma)

    def _iterate(self, num_itr: int, yMF: Tensor, s: Tensor) -> None:

        inv_omega = torch.div(1, self.omega)
        m_inv_sor = torch.linalg.inv(self._D - torch.mul(inv_omega, self._L))

        s_present = s
        s_old = torch.zeros(s_present.shape).to(self._device)

        for _ in range(num_itr):
            temp = torch.mul((inv_omega - 1), self._D) + torch.mul(inv_omega, self._U)
            s = torch.matmul(s, torch.matmul(m_inv_sor, temp)) + torch.matmul(
                yMF, m_inv_sor
            )

            s_new = (
                self.omegaa * (self.gamma * (s - s_present) + (s_present - s_old))
                + s_old
            )
            s_old = s
            s_present = s_new

            self._s_hats.append(s_new)


class AOR(IterativeModel):
    """The Accelerated Over-Relaxation (AOR) algorithm for solving a linear system."""

    omega: Tensor
    """Relaxation parameter for AOR iterations."""

    r: Tensor
    """Relaxation parameter."""

    def __init__(
        self,
        n: int,
        a: NDArray,
        h: Tensor,
        bs: int,
        y: Tensor,
        omega: float = 0.3,
        r: float = 0.2,
        device: torch.device = _device,
    ):
        """Initialize the AOR solver.

        Args:
          n: Dimension of the solution.
          a: Input square matrix to decompose.
          h: Random matrix $H$.
          bs: Batch size.
          y: Solution tensor.
          omega: Relaxation parameter for AOR iterations.
          r: Relaxation parameter.
          device: Device where to run the model.
        """
        super().__init__(n, a, h, bs, y, device)
        self.omega = torch.tensor(omega)
        self.r = torch.tensor(r)

    def _iterate(self, num_itr: int, yMF: Tensor, s: Tensor) -> None:

        m = self._D - torch.mul(self.r, self._L)
        m_inv_aor = torch.linalg.inv(m)

        # ! bug : difference between n and self.n (some ambiguity)
        n = (
            torch.mul((1 - self.omega), self._D)
            + torch.mul((self.omega - self.r), self._L)
            + torch.mul(self.omega, self._U)
        )

        for _ in range(num_itr):
            s = torch.matmul(s, torch.matmul(m_inv_aor, n)) + torch.mul(
                self.omega, torch.matmul(yMF, m_inv_aor)
            )
            self._s_hats.append(s)


class AORCheby(IterativeModel):
    """The AOR with Chebyshev acceleration algorithm for solving a linear system."""

    omega: Tensor
    """Relaxation parameter for AOR iterations."""

    r: Tensor
    """Relaxation parameter."""

    def __init__(
        self,
        n: int,
        a: NDArray,
        h: Tensor,
        bs: int,
        y: Tensor,
        omega: float = 0.1,
        r: float = 0.1,
        device: torch.device = _device,
    ):
        """Initialize the AOR-Chebyshev solver.

        Args:
          n: Dimension of the solution.
          a: Input square matrix to decompose.
          h: Random matrix $H$.
          bs: Batch size.
          y: Solution tensor.
          omega: Relaxation parameter for AOR iterations.
          r: Relaxation parameter.
          device: Device where to run the model.
        """
        super().__init__(n, a, h, bs, y, device)
        self.omega = torch.tensor(omega)
        self.r = torch.tensor(r)

    def _iterate(self, num_itr: int, yMF: Tensor, s: Tensor) -> None:

        y0 = s

        m = self._D - torch.mul(self.r, self._L)
        m_inv = torch.linalg.inv(m)
        n = (
            torch.mul((1 - self.omega), self._D)
            + torch.mul((self.omega - self.r), self._L)
            + torch.mul(self.omega, self._U)
        )
        temp = torch.matmul(m_inv, n)

        rho = torch.tensor(0.1)
        mu0 = torch.tensor(1)
        mu1 = rho
        xhat1 = torch.matmul(s, temp) + self.omega * torch.matmul(yMF, m_inv)
        y1 = xhat1
        y = y1

        for _ in range(num_itr):
            f = 2 / (rho * mu1)
            j = 1 / mu0
            c = f - j
            mu = 1 / c
            a = (2 * mu) / (rho * mu1)
            y = (
                torch.matmul((y1 * a), torch.matmul(m_inv, n))
                - (((mu / mu0)) * y0)
                + (a * torch.matmul(yMF, m_inv))
            )
            y0 = y1
            y1 = y
            mu0 = mu1
            mu1 = mu

            self._s_hats.append(y)
