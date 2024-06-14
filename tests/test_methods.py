# Copyright (c) 2023-2024 Salah Berra and contributors
# Distributed under the the GNU General Public License (See accompanying file
# LICENSE or copy at https://www.gnu.org/licenses/)

import numpy as np
import pytest
import torch

from deep_unfolding import (
    AOR,
    AOR_CHEBY,
    GS,
    RI,
    SOR,
    SOR_CHEBY,
    BaseModel,
    Jacobi,
    decompose_matrix,
    device,
    model_iterations,
)


@pytest.fixture
def generate_matrices():
    n = 300
    m = 600
    bs = 10
    A = np.random.rand(n, n)
    H = torch.from_numpy(np.random.rand(n, m)).float().to(device)
    y = torch.from_numpy(np.random.rand(bs, m)).float().to(device)
    return n, A, H, bs, y


@pytest.fixture
def generate_solution():
    bs = 10
    n = 300
    return torch.from_numpy(np.random.rand(bs, n)).float().to(device)


def test_model_iterations(generate_matrices, generate_solution):
    n, _, _, bs, _ = generate_matrices
    solution = generate_solution

    class DummyModel:
        def iterate(self, i):
            return solution, []

    model = DummyModel()
    s_hats, norm_list_model = model_iterations(model, solution, n, total_itr=5, bs=bs)

    assert len(s_hats) == 6, "s_hats should contain total_itr + 1 elements"
    assert (
        len(norm_list_model) == 6
    ), "norm_list_model should contain total_itr + 1 elements"
    for norm in norm_list_model:
        assert isinstance(
            norm, float
        ), "Each element in norm_list_model should be a float"


def test_base_model_initialization(generate_matrices):
    n, A, H, bs, y = generate_matrices
    model = BaseModel(n, A, H, bs, y)

    assert model.n == n, "Attribute n should be initialized correctly"
    assert model.H.shape == H.shape, "Attribute H should be initialized correctly"
    assert model.bs == bs, "Attribute bs should be initialized correctly"
    assert model.y.shape == y.shape, "Attribute y should be initialized correctly"

    A_torch, D, L, U, Dinv, Minv = decompose_matrix(A)
    assert torch.allclose(
        model.A, A_torch
    ), "Attribute A should match the decomposed matrix"
    assert torch.allclose(model.D, D), "Attribute D should match the decomposed matrix"
    assert torch.allclose(model.L, L), "Attribute L should match the decomposed matrix"
    assert torch.allclose(model.U, U), "Attribute U should match the decomposed matrix"
    assert torch.allclose(
        model.Dinv, Dinv
    ), "Attribute Dinv should match the decomposed matrix"
    assert torch.allclose(
        model.Minv, Minv
    ), "Attribute Minv should match the decomposed matrix"


def test_GS_initialization(generate_matrices):
    n, A, H, bs, y = generate_matrices
    gs_model = GS(n, A, H, bs, y)

    assert gs_model.n == n, "Attribute n should be initialized correctly in GS model"
    assert (
        gs_model.H.shape == H.shape
    ), "Attribute H should be initialized correctly in GS model"
    assert gs_model.bs == bs, "Attribute bs should be initialized correctly in GS model"
    assert (
        gs_model.y.shape == y.shape
    ), "Attribute y should be initialized correctly in GS model"

    A_torch, D, L, U, Dinv, Minv = decompose_matrix(A)
    assert torch.allclose(
        gs_model.A, A_torch
    ), "Attribute A should match the decomposed matrix in GS model"
    assert torch.allclose(
        gs_model.D, D
    ), "Attribute D should match the decomposed matrix in GS model"
    assert torch.allclose(
        gs_model.L, L
    ), "Attribute L should match the decomposed matrix in GS model"
    assert torch.allclose(
        gs_model.U, U
    ), "Attribute U should match the decomposed matrix in GS model"
    assert torch.allclose(
        gs_model.Dinv, Dinv
    ), "Attribute Dinv should match the decomposed matrix in GS model"
    assert torch.allclose(
        gs_model.Minv, Minv
    ), "Attribute Minv should match the decomposed matrix in GS model"


def test_GS_iterate(generate_matrices):
    n, A, H, bs, y = generate_matrices
    gs_model = GS(n, A, H, bs, y)

    s, traj = gs_model.iterate(num_itr=5)

    assert len(traj) == 6, "Trajectory should contain num_itr + 1 elements"
    assert traj[0].shape == (
        bs,
        n,
    ), "Each element in the trajectory should have shape (bs, n)"
    assert s.shape == (bs, n), "Final solution tensor should have shape (bs, n)"


def test_RI_initialization(generate_matrices):
    n, A, H, bs, y = generate_matrices
    ri_model = RI(n, A, H, bs, y)

    assert ri_model.n == n, "Attribute n should be initialized correctly in RI model"
    assert (
        ri_model.H.shape == H.shape
    ), "Attribute H should be initialized correctly in RI model"
    assert ri_model.bs == bs, "Attribute bs should be initialized correctly in RI model"
    assert (
        ri_model.y.shape == y.shape
    ), "Attribute y should be initialized correctly in RI model"

    A_torch, D, L, U, Dinv, Minv = decompose_matrix(A)
    assert torch.allclose(
        ri_model.A, A_torch
    ), "Attribute A should match the decomposed matrix in RI model"
    assert torch.allclose(
        ri_model.D, D
    ), "Attribute D should match the decomposed matrix in RI model"
    assert torch.allclose(
        ri_model.L, L
    ), "Attribute L should match the decomposed matrix in RI model"
    assert torch.allclose(
        ri_model.U, U
    ), "Attribute U should match the decomposed matrix in RI model"
    assert torch.allclose(
        ri_model.Dinv, Dinv
    ), "Attribute Dinv should match the decomposed matrix in RI model"
    assert torch.allclose(
        ri_model.Minv, Minv
    ), "Attribute Minv should match the decomposed matrix in RI model"


def test_RI_iteration(generate_matrices):
    n, A, H, bs, y = generate_matrices
    ri_model = RI(n, A, H, bs, y)

    s, traj = ri_model.iterate(num_itr=5)
    assert len(traj) == 6, "Trajectory should contain num_itr + 1 elements"
    assert s.shape == (bs, n), "Final solution tensor should have the correct shape"


def test_Jacobi_initialization(generate_matrices):
    n, A, H, bs, y = generate_matrices
    jacobi_model = Jacobi(n, A, H, bs, y, omega=0.2)

    assert (
        jacobi_model.n == n
    ), "Attribute n should be initialized correctly in Jacobi model"
    assert (
        jacobi_model.H.shape == H.shape
    ), "Attribute H should be initialized correctly in Jacobi model"
    assert (
        jacobi_model.bs == bs
    ), "Attribute bs should be initialized correctly in Jacobi model"
    assert (
        jacobi_model.y.shape == y.shape
    ), "Attribute y should be initialized correctly in Jacobi model"
    assert jacobi_model.omega == torch.tensor(
        0.2
    ), "Attribute omega should be initialized correctly in Jacobi model"

    A_torch, D, L, U, Dinv, Minv = decompose_matrix(A)
    assert torch.allclose(
        jacobi_model.A, A_torch
    ), "Attribute A should match the decomposed matrix in Jacobi model"
    assert torch.allclose(
        jacobi_model.D, D
    ), "Attribute D should match the decomposed matrix in Jacobi model"
    assert torch.allclose(
        jacobi_model.L, L
    ), "Attribute L should match the decomposed matrix in Jacobi model"
    assert torch.allclose(
        jacobi_model.U, U
    ), "Attribute U should match the decomposed matrix in Jacobi model"
    assert torch.allclose(
        jacobi_model.Dinv, Dinv
    ), "Attribute Dinv should match the decomposed matrix in Jacobi model"
    assert torch.allclose(
        jacobi_model.Minv, Minv
    ), "Attribute Minv should match the decomposed matrix in Jacobi model"


def test_Jacobi_iteration(generate_matrices):
    n, A, H, bs, y = generate_matrices
    jacobi_model = Jacobi(n, A, H, bs, y, omega=0.2)

    s, traj = jacobi_model.iterate(num_itr=5)
    assert len(traj) == 6, "Trajectory should contain num_itr + 1 elements"
    assert s.shape == (bs, n), "Final solution tensor should have the correct shape"


def test_SOR_initialization(generate_matrices):
    n, A, H, bs, y = generate_matrices
    sor_model = SOR(n, A, H, bs, y, omega=1.8)

    assert sor_model.n == n, "Attribute n should be initialized correctly in SOR model"
    assert (
        sor_model.H.shape == H.shape
    ), "Attribute H should be initialized correctly in SOR model"
    assert (
        sor_model.bs == bs
    ), "Attribute bs should be initialized correctly in SOR model"
    assert (
        sor_model.y.shape == y.shape
    ), "Attribute y should be initialized correctly in SOR model"
    assert sor_model.omega == torch.tensor(
        1.8
    ), "Attribute omega should be initialized correctly in SOR model"

    A_torch, D, L, U, Dinv, Minv = decompose_matrix(A)
    assert torch.allclose(
        sor_model.A, A_torch
    ), "Attribute A should match the decomposed matrix in SOR model"
    assert torch.allclose(
        sor_model.D, D
    ), "Attribute D should match the decomposed matrix in SOR model"
    assert torch.allclose(
        sor_model.L, L
    ), "Attribute L should match the decomposed matrix in SOR model"
    assert torch.allclose(
        sor_model.U, U
    ), "Attribute U should match the decomposed matrix in SOR model"
    assert torch.allclose(
        sor_model.Dinv, Dinv
    ), "Attribute Dinv should match the decomposed matrix in SOR model"
    assert torch.allclose(
        sor_model.Minv, Minv
    ), "Attribute Minv should match the decomposed matrix in SOR model"


def test_SOR_iteration(generate_matrices):
    n, A, H, bs, y = generate_matrices
    sor_model = SOR(n, A, H, bs, y, omega=1.8)

    s, traj = sor_model.iterate(num_itr=5)
    assert len(traj) == 6, "Trajectory should contain num_itr + 1 elements"
    assert s.shape == (bs, n), "Final solution tensor should have the correct shape"


def test_SOR_CHEBY_initialization(generate_matrices):
    n, A, H, bs, y = generate_matrices
    sor_cheby_model = SOR_CHEBY(n, A, H, bs, y, omega=1.8, omegaa=0.8, gamma=0.8)

    assert (
        sor_cheby_model.n == n
    ), "Attribute n should be initialized correctly in SOR_CHEBY model"
    assert (
        sor_cheby_model.H.shape == H.shape
    ), "Attribute H should be initialized correctly in SOR_CHEBY model"
    assert (
        sor_cheby_model.bs == bs
    ), "Attribute bs should be initialized correctly in SOR_CHEBY model"
    assert (
        sor_cheby_model.y.shape == y.shape
    ), "Attribute y should be initialized correctly in SOR_CHEBY model"
    assert sor_cheby_model.omega == torch.tensor(
        1.8
    ), "Attribute omega should be initialized correctly in SOR_CHEBY model"
    assert sor_cheby_model.omegaa == torch.tensor(
        0.8
    ), "Attribute omegaa should be initialized correctly in SOR_CHEBY model"
    assert sor_cheby_model.gamma == torch.tensor(
        0.8
    ), "Attribute gamma should be initialized correctly in SOR_CHEBY model"

    A_torch, D, L, U, Dinv, Minv = decompose_matrix(A)
    assert torch.allclose(
        sor_cheby_model.A, A_torch
    ), "Attribute A should match the decomposed matrix in SOR_CHEBY model"
    assert torch.allclose(
        sor_cheby_model.D, D
    ), "Attribute D should match the decomposed matrix in SOR_CHEBY model"
    assert torch.allclose(
        sor_cheby_model.L, L
    ), "Attribute L should match the decomposed matrix in SOR_CHEBY model"
    assert torch.allclose(
        sor_cheby_model.U, U
    ), "Attribute U should match the decomposed matrix in SOR_CHEBY model"
    assert torch.allclose(
        sor_cheby_model.Dinv, Dinv
    ), "Attribute Dinv should match the decomposed matrix in SOR_CHEBY model"
    assert torch.allclose(
        sor_cheby_model.Minv, Minv
    ), "Attribute Minv should match the decomposed matrix in SOR_CHEBY model"


def test_SOR_CHEBY_iteration(generate_matrices):
    n, A, H, bs, y = generate_matrices
    sor_cheby_model = SOR_CHEBY(n, A, H, bs, y, omega=1.8, omegaa=0.8, gamma=0.8)

    s, traj = sor_cheby_model.iterate(num_itr=5)
    assert len(traj) == 6, "Trajectory should contain num_itr + 1 elements"
    assert s.shape == (bs, n), "Final solution tensor should have the correct shape"


def test_AOR_initialization(generate_matrices):
    n, A, H, bs, y = generate_matrices
    aor_model = AOR(n, A, H, bs, y, omega=0.3, r=0.2)

    assert aor_model.n == n, "Attribute n should be initialized correctly in AOR model"
    assert (
        aor_model.H.shape == H.shape
    ), "Attribute H should be initialized correctly in AOR model"
    assert (
        aor_model.bs == bs
    ), "Attribute bs should be initialized correctly in AOR model"
    assert (
        aor_model.y.shape == y.shape
    ), "Attribute y should be initialized correctly in AOR model"
    assert aor_model.omega == torch.tensor(
        0.3
    ), "Attribute omega should be initialized correctly in AOR model"
    assert aor_model.r == torch.tensor(
        0.2
    ), "Attribute r should be initialized correctly in AOR model"

    A_torch, D, L, U, Dinv, Minv = decompose_matrix(A)
    assert torch.allclose(
        aor_model.A, A_torch
    ), "Attribute A should match the decomposed matrix in AOR model"
    assert torch.allclose(
        aor_model.D, D
    ), "Attribute D should match the decomposed matrix in AOR model"
    assert torch.allclose(
        aor_model.L, L
    ), "Attribute L should match the decomposed matrix in AOR model"
    assert torch.allclose(
        aor_model.U, U
    ), "Attribute U should match the decomposed matrix in AOR model"
    assert torch.allclose(
        aor_model.Dinv, Dinv
    ), "Attribute Dinv should match the decomposed matrix in AOR model"
    assert torch.allclose(
        aor_model.Minv, Minv
    ), "Attribute Minv should match the decomposed matrix in AOR model"


def test_AOR_iteration(generate_matrices):
    n, A, H, bs, y = generate_matrices
    aor_model = AOR(n, A, H, bs, y, omega=0.3, r=0.2)

    s, traj = aor_model.iterate(num_itr=5)
    assert len(traj) == 6, "Trajectory should contain num_itr + 1 elements"
    assert s.shape == (bs, n), "Final solution tensor should have the correct shape"


def test_AOR_CHEBY_initialization(generate_matrices):
    n, A, H, bs, y = generate_matrices
    aor_cheby_model = AOR_CHEBY(n, A, H, bs, y, omega=0.1, r=0.1)

    assert (
        aor_cheby_model.n == n
    ), "Attribute n should be initialized correctly in AOR_CHEBY model"
    assert (
        aor_cheby_model.H.shape == H.shape
    ), "Attribute H should be initialized correctly in AOR_CHEBY model"
    assert (
        aor_cheby_model.bs == bs
    ), "Attribute bs should be initialized correctly in AOR_CHEBY model"
    assert (
        aor_cheby_model.y.shape == y.shape
    ), "Attribute y should be initialized correctly in AOR_CHEBY model"
    assert aor_cheby_model.omega == torch.tensor(
        0.1
    ), "Attribute omega should be initialized correctly in AOR_CHEBY model"
    assert aor_cheby_model.r == torch.tensor(
        0.1
    ), "Attribute r should be initialized correctly in AOR_CHEBY model"

    A_torch, D, L, U, Dinv, Minv = decompose_matrix(A)
    assert torch.allclose(
        aor_cheby_model.A, A_torch
    ), "Attribute A should match the decomposed matrix in AOR_CHEBY model"
    assert torch.allclose(
        aor_cheby_model.D, D
    ), "Attribute D should match the decomposed matrix in AOR_CHEBY model"
    assert torch.allclose(
        aor_cheby_model.L, L
    ), "Attribute L should match the decomposed matrix in AOR_CHEBY model"
    assert torch.allclose(
        aor_cheby_model.U, U
    ), "Attribute U should match the decomposed matrix in AOR_CHEBY model"
    assert torch.allclose(
        aor_cheby_model.Dinv, Dinv
    ), "Attribute Dinv should match the decomposed matrix in AOR_CHEBY model"
    assert torch.allclose(
        aor_cheby_model.Minv, Minv
    ), "Attribute Minv should match the decomposed matrix in AOR_CHEBY model"


def test_AOR_CHEBY_iteration(generate_matrices):
    n, A, H, bs, y = generate_matrices
    aor_cheby_model = AOR_CHEBY(n, A, H, bs, y, omega=0.1, r=0.1)

    s, traj = aor_cheby_model.iterate(num_itr=5)
    assert len(traj) == 6, "Trajectory should contain num_itr + 1 elements"
    assert s.shape == (bs, n), "Final solution tensor should have the correct shape"


if __name__ == "__main__":
    pytest.main()
