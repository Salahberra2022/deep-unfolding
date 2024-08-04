# Copyright (c) 2023-2024 Salah Berra and contributors
# Distributed under the the GNU General Public License (See accompanying file
# LICENSE or copy at https://www.gnu.org/licenses/)

import numpy as np
import pytest
import torch

from torch import Tensor
from numpy.typing import NDArray

from deep_unfolding import (
    AOR,
    GaussSeidel,
    Richardson,
    SOR,
    AORCheby,
    IterativeModel,
    Jacobi,
    SORCheby,
    _decompose_matrix,
    _device,
)


@pytest.fixture
def generate_matrices():
    n = 300
    m = 600
    bs = 10
    A = np.random.rand(n, n)
    H = torch.from_numpy(np.random.rand(n, m)).float().to(_device)
    y = torch.from_numpy(np.random.rand(bs, m)).float().to(_device)
    return n, A, H, bs, y


@pytest.fixture
def generate_solution():
    bs = 10
    n = 300
    return torch.from_numpy(np.random.rand(bs, n)).float().to(_device)


def pytest_generate_tests(metafunc):
    """Special PyTest function for dynamically creating parameterized fixtures."""

    if "iterative_model" in metafunc.fixturenames:
        models_to_test = []
        n, A, H, bs, y = generate_matrices

        # Gauss-Seidel
        models_to_test.append(GaussSeidel(n, A, H, bs, y, _device))

        # Richardson model
        omega = 0.25
        models_to_test.append(Richardson(n, A, H, bs, y, omega, _device))

        # Jacobi model
        omega = 0.2
        models_to_test.append(Jacobi(n, A, H, bs, y, omega, _device))

        # SOR model
        omega = 1.8
        models_to_test.append(SOR(n, A, H, bs, y, omega, _device))

        # SOR-Cheby model
        omega = 1.8
        omegaa = 0.8
        gamma = 0.8
        models_to_test.append(SORCheby(n, A, H, bs, y, omega, omegaa, gamma, _device))

        # AOR model
        omega = 0.3
        r = 0.2
        models_to_test.append(AOR(n, A, H, bs, y, omega, r, _device))

        # AOR-Cheby model
        omega = 0.1
        r = 0.1
        models_to_test.append(AORCheby(n, A, H, bs, y, omega, r, _device))

        # Generate a dynamic 'iterative_model' fixture with all the iterative
        # models to test
        metafunc.parametrize("iterative_model", models_to_test)



def test_model_initialization(iterative_model):
    """Test initialization of models defined in the iterative_models fixture."""

    assert iterative_model.n == n, "Attribute n should be initialized correctly"
    assert iterative_model.H.shape == H.shape, "Attribute H should be initialized correctly"
    assert iterative_model.bs == bs, "Attribute bs should be initialized correctly"
    assert iterative_model.y.shape == y.shape, "Attribute y should be initialized correctly"

    A_torch, D, L, U, Dinv, Minv = _decompose_matrix(A, _device)
    assert torch.allclose(
        iterative_model.A, A_torch
    ), "Attribute A should match the decomposed matrix"
    assert torch.allclose(iterative_model.D, D), "Attribute D should match the decomposed matrix"
    assert torch.allclose(iterative_model.L, L), "Attribute L should match the decomposed matrix"
    assert torch.allclose(iterative_model.U, U), "Attribute U should match the decomposed matrix"
    assert torch.allclose(
        iterative_model.Dinv, Dinv
    ), "Attribute Dinv should match the decomposed matrix"
    assert torch.allclose(
        iterative_model.Minv, Minv
    ), "Attribute Minv should match the decomposed matrix"



if __name__ == "__main__":
    pytest.main()