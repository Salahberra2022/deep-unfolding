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

    n: int
    """Dimension of the solution."""

    H: Tensor
    """Random matrix $H$."""

    bs: int
    """Batch size."""

    y: Tensor
    """Solution tensor."""

    A: Tensor
    """Original matrix converted to a torch tensor."""

    D: Tensor
    """Diagonal matrix of $A$."""

    L: Tensor
    """Lower triangular matrix of $A$."""

    U: Tensor
    """Upper triangular matrix of $A$."""

    Dinv: Tensor
    """Inverse of the diagonal matrix $D$."""

    Minv: Tensor
    """Inverse of the matrix $D + L$."""

    device: torch.device
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

          a: Input square matrix to decompose.
          h: Random matrix $H$.
          bs: Batch size.
          y: Solution tensor.
          device: Device to run the model on ('cpu' or 'cuda').
        """
        self.n = n
        self.H = h
        self.bs = bs
        self.y = y
        self.device = device
        self._solved = False

        self._s_hats = []

        self.A, self.D, self.L, self.U, self.Dinv, self.Minv = _decompose_matrix(
            a, device
        )

    @abstractmethod
    def _iterate(self, num_itr: int, yMF: Tensor, s: Tensor) -> None:
        """Iteration method, to be implemented in subclasses."""
        pass

    def solve(
        self,
        total_itr: int = 25,
    ) -> list[Tensor]:
        """Perform iterations using the provided model and calculate the error norm at each iteration.

        Args:
          total_itr: Total number of iterations to perform.
          device: Device to run the model on ('cpu' or 'cuda').

        Returns:
          A tuple with the following contents:
            - List of tensors representing the solution estimates at each iteration.
        """
        if self._solved:
            raise RuntimeError("Problem has already been solved!")

        self._solved = True

        s = torch.zeros(self.bs, self.n).to(self.device)
        self._s_hats.append(s)

        yMF = torch.matmul(self.y, self.H.T)  # Assuming H is defined
        s = torch.matmul(yMF, self.Dinv)  # Generate batch initial solution vector

        self._iterate(total_itr, yMF, s)

        return self._s_hats

    def _evaluate(
        self,
        iter: int,
        solution: Tensor,
        device: torch.device = _device,
    ) -> float:
        """Evaluate function

        Args:
          TODO.

        Returns:
          TODO
        """
        return (
            torch.norm(solution.to(device) - self._s_hats[iter].to(device)) ** 2
        ).item() / (self.n * self.bs)

    def evaluate_final(
        self,
        solution: Tensor,
        device: torch.device = _device,
    ) -> float:
        """Evaluate function

        Args:
            num_itr (int, optional): Number of iterations choose. Defaults to 10.
            solution (Tensor, optional): The solution of the linear problem. Defaults to None.
            device (torch.device, optional): The device. Defaults to _device.

        Returns:
            The error between the exact solution and the proposed solution
        """
        if not self._solved:
            raise RuntimeError("Problem has not been solved yet!")

        return self._evaluate(-1, solution, device)

    def evaluate_all(
        self,
        solution: Tensor,
        device: torch.device = _device,
    ) -> list[float]:
        """Evaluate function

        Args:
            num_itr (int, optional): Number of iterations choose. Defaults to 10.
            solution (Tensor, optional): The solution of the linear problem. Defaults to None.
            device (torch.device, optional): The device. Defaults to _device.

        Returns:
            The error between the exact solution and the proposed solution
        """
        if not self._solved:
            raise RuntimeError("Problem has not been solved yet!")

        return [self._evaluate(i, solution, device) for i in range(len(self._s_hats))]


class GaussSeidel(IterativeModel):
    """Class implementing the Gauss-Seidel algorithm for solving a linear system."""

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
        """Performs the Gauss-Seidel iterations and returns the final solution
          and trajectory of solutions.

        Args:
          num_itr: The number of iterations to perform.
        """
        for _ in range(num_itr):
            temp = -torch.matmul(s, self.U) + yMF
            s = torch.matmul(temp, self.Minv)
            self._s_hats.append(s)


class Richardson(IterativeModel):
    """Class implementing the Richardson iteration algorithm for solving a linear system."""

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
        """Performs the Richardson iterations and returns the final solution and
          trajectory of solutions.

        Args:
          num_itr: The number of iterations to perform.
        """

        for _ in range(num_itr):
            s = s + torch.mul(self.omega, (yMF - torch.matmul(s, self.A)))
            self._s_hats.append(s)


class Jacobi(IterativeModel):
    """Class implementing the Jacobi iteration algorithm for solving a linear system."""

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
        """Performs the Jacobi iterations and returns the final solution and
          trajectory of solutions.

        Args:
          num_itr: The number of iterations to perform.
        """

        for _ in range(num_itr):
            temp = torch.matmul(self.Dinv, (self.D - self.A))
            s = torch.matmul(s, temp) + torch.matmul(yMF, self.Dinv)
            self._s_hats.append(s)


class SOR(IterativeModel):
    """Class implementing the Successive Over-Relaxation (SOR) algorithm for
    solving a linear system."""

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
        """Performs the SOR iterations and returns the final solution and
          trajectory of solutions.

        Args:
          num_itr: The number of iterations to perform.
        """

        inv_omega = torch.div(1, self.omega)
        m_inv_sor = torch.linalg.inv(self.D - torch.mul(inv_omega, self.L))

        for _ in range(num_itr):
            temp = torch.mul((inv_omega - 1), self.D) + torch.mul(inv_omega, self.U)
            s = torch.matmul(s, torch.matmul(m_inv_sor, temp)) + torch.matmul(
                yMF, m_inv_sor
            )
            self._s_hats.append(s)


class SORCheby(IterativeModel):
    """Class implementing the SOR-Chebyshev algorithm for solving a linear system."""

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
        """Performs the SOR-Chebyshev iterations and returns the final solution
          and trajectory of solutions.

        Args:
          num_itr: The number of iterations to perform.
        """

        inv_omega = torch.div(1, self.omega)
        m_inv_sor = torch.linalg.inv(self.D - torch.mul(inv_omega, self.L))

        s_present = s
        s_old = torch.zeros(s_present.shape).to(self.device)

        for _ in range(num_itr):
            temp = torch.mul((inv_omega - 1), self.D) + torch.mul(inv_omega, self.U)
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
    """Class implementing the Accelerated Over-Relaxation (AOR) algorithm for
    solving a linear system."""

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
        """Performs the AOR iterations and returns the final solution and
          trajectory of solutions.

        Args:
          num_itr: The number of iterations to perform.
        """

        m = self.D - torch.mul(self.r, self.L)
        m_inv_aor = torch.linalg.inv(m)

        # ! bug : difference between n and self.n (some ambiguity)
        n = (
            torch.mul((1 - self.omega), self.D)
            + torch.mul((self.omega - self.r), self.L)
            + torch.mul(self.omega, self.U)
        )

        for _ in range(num_itr):
            s = torch.matmul(s, torch.matmul(m_inv_aor, n)) + torch.mul(
                self.omega, torch.matmul(yMF, m_inv_aor)
            )
            self._s_hats.append(s)


class AORCheby(IterativeModel):
    """Class implementing the Accelerated Over-Relaxation (AOR) with Chebyshev
    acceleration algorithm for solving a linear system."""

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
        """Performs the AOR-Chebyshev iterations and returns the final solution
          and trajectory of solutions.

        Args:
          num_itr: The number of iterations to perform.
        """

        y0 = s

        m = self.D - torch.mul(self.r, self.L)
        m_inv = torch.linalg.inv(m)
        n = (
            torch.mul((1 - self.omega), self.D)
            + torch.mul((self.omega - self.r), self.L)
            + torch.mul(self.omega, self.U)
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
