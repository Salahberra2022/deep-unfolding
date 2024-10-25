# Copyright (c) 2023-2024 Salah Berra and contributors
# Distributed under the the GNU General Public License (See accompanying file
# LICENSE or copy at https://www.gnu.org/licenses/)

import pytest
import torch
import torch.nn as nn

from deep_unfolding import (AORNet, RichardsonNet, SORChebyNet,  # train_model,
                            SORNet, _device, evaluate, gen_linear)


# Fixture pour générer les matrices et tensors nécessaires
@pytest.fixture
def generate_matrices():
    A, H, _, solution, y = gen_linear()
    n, m = A.shape[0], H.shape[1]
    bs = solution.shape[0]
    return A, H, y, n, m, bs, solution


# test SORNet
def test_SORNet_initialization(generate_matrices):
    A, H, y, n, m, bs, solution = generate_matrices
    model = SORNet(A, H, bs, y, device=_device)

    assert model.A.shape == (n, n), "Attribute A should have the correct shape"
    assert model.H.shape == (n, m), "Attribute H should have the correct shape"
    assert model.D.shape == (n, n), "Attribute D should have the correct shape"
    assert model.L.shape == (n, n), "Attribute L should have the correct shape"
    assert model.U.shape == (n, n), "Attribute U should have the correct shape"
    assert model.Dinv.shape == (n, n), "Attribute Dinv should have the correct shape"
    assert model.bs == bs, "Attribute bs should be initialized correctly"
    assert model.y.shape == (bs, m), "Attribute y should have the correct shape"
    assert torch.is_tensor(model.inv_omega), "Attribute inv_omega should be a tensor"


def test_SORNet_forward(generate_matrices):
    A, H, y, n, m, bs, solution = generate_matrices
    model = SORNet(A, H, bs, y, device=_device)

    s, traj = model.forward(num_itr=3)

    assert s.shape == (bs, n), "Output tensor should have the correct shape"
    assert len(traj) == 4, "Trajectory should contain num_itr + 1 elements"
    assert all(
        t.shape == (bs, n) for t in traj
    ), "Each element in the trajectory should have the correct shape"


# Test SOR_ChebyNet
def test_SOR_CHEBY_Net_initialization(generate_matrices):
    num_itr = 25
    A, H, y, n, m, bs, solution = generate_matrices
    model = SORChebyNet(num_itr, A, H, bs, y, device=_device)

    assert model.A.shape == (n, n), "Attribute A should have the correct shape"
    assert model.H.shape == (n, m), "Attribute H should have the correct shape"
    assert model.D.shape == (n, n), "Attribute D should have the correct shape"
    assert model.L.shape == (n, n), "Attribute L should have the correct shape"
    assert model.U.shape == (n, n), "Attribute U should have the correct shape"
    assert model.Dinv.shape == (n, n), "Attribute Dinv should have the correct shape"
    assert model.bs == bs, "Attribute bs should be initialized correctly"
    assert model.y.shape == (bs, m), "Attribute y should have the correct shape"
    assert torch.is_tensor(model.inv_omega), "Attribute inv_omega should be a tensor"
    assert model.gamma.shape == (
        num_itr,
    ), "Attribute gamma should have the correct shape"
    assert model.omega.shape == (
        num_itr,
    ), "Attribute omega should have the correct shape"


#### Test du forward pass


def test_SOR_CHEBY_Net_forward(generate_matrices):
    num_itr = 3
    A, H, y, n, m, bs, solution = generate_matrices
    model = SORChebyNet(num_itr, A, H, bs, y, device=_device)

    s, traj = model.forward(num_itr=num_itr)

    assert s.shape == (bs, n), "Output tensor should have the correct shape"
    assert len(traj) == num_itr + 1, "Trajectory should contain num_itr + 1 elements"
    assert all(
        t.shape == (bs, n) for t in traj
    ), "Each element in the trajectory should have the correct shape"


# Test AORNet
def test_AORNet_initialization(generate_matrices):
    A, H, y, n, m, bs, solution = generate_matrices
    model = AORNet(A, H, bs, y, device=_device)

    assert model.A.shape == (n, n), "Attribute A should have the correct shape"
    assert model.H.shape == (n, m), "Attribute H should have the correct shape"
    assert model.D.shape == (n, n), "Attribute D should have the correct shape"
    assert model.L.shape == (n, n), "Attribute L should have the correct shape"
    assert model.U.shape == (n, n), "Attribute U should have the correct shape"
    assert model.Dinv.shape == (n, n), "Attribute Dinv should have the correct shape"
    assert model.bs == bs, "Attribute bs should be initialized correctly"
    assert model.y.shape == (bs, m), "Attribute y should have the correct shape"
    assert torch.is_tensor(model.r), "Attribute r should be a tensor"
    assert torch.is_tensor(model.omega), "Attribute omega should be a tensor"


def test_AORNet_forward(generate_matrices):
    A, H, y, n, m, bs, solution = generate_matrices
    model = AORNet(A, H, bs, y, device=_device)

    s, traj = model.forward(num_itr=3)

    assert s.shape == (bs, n), "Output tensor should have the correct shape"
    assert len(traj) == 4, "Trajectory should contain num_itr + 1 elements"
    assert all(
        t.shape == (bs, n) for t in traj
    ), "Each element in the trajectory should have the correct shape"


# Test RINet
def test_RINet_initialization(generate_matrices):
    A, H, y, n, m, bs, solution = generate_matrices
    model = RichardsonNet(A, H, bs, y, device=_device)

    assert model.A.shape == (n, n), "Attribute A should have the correct shape"
    assert model.H.shape == (n, m), "Attribute H should have the correct shape"
    assert model.D.shape == (n, n), "Attribute D should have the correct shape"
    assert model.L.shape == (n, n), "Attribute L should have the correct shape"
    assert model.U.shape == (n, n), "Attribute U should have the correct shape"
    assert model.Dinv.shape == (n, n), "Attribute Dinv should have the correct shape"
    assert model.bs == bs, "Attribute bs should be initialized correctly"
    assert model.y.shape == (bs, m), "Attribute y should have the correct shape"
    assert torch.is_tensor(model.inv_omega), "Attribute inv_omega should be a tensor"


def test_RINet_forward(generate_matrices):
    A, H, y, n, m, bs, solution = generate_matrices
    model = RichardsonNet(A, H, bs, y, device=_device)

    s, traj = model.forward(num_itr=3)

    assert s.shape == (bs, n), "Output tensor should have the correct shape"
    assert len(traj) == 4, "Trajectory should contain num_itr + 1 elements"
    assert all(
        t.shape == (bs, n) for t in traj
    ), "Each element in the trajectory should have the correct shape"


############################## TEST FUNCTIONS ##############################
# test functions just with SORNet is sufficient
# def test_train_model(generate_matrices):
#     A, H, y, n, m, bs, solution = generate_matrices
#     model = SORNet(A, H, bs, y, device=_device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     loss_func = nn.MSELoss()

#     trained_model, loss_gen = train_model(
#         model, optimizer, loss_func, solution, total_itr=3, num_batch=5
#     )

#     assert isinstance(trained_model, SORNet), "Returned model should be of type SORNet"
#     assert len(loss_gen) == 3, "Length of loss_gen should be equal to total_itr"


def test_evaluate_model(generate_matrices):
    A, H, y, n, m, bs, solution = generate_matrices
    model = SORNet(A, H, bs, y, device=_device)

    norm_list = evaluate_model(model, solution, n, bs=bs, total_itr=3, device=_device)

    assert len(norm_list) == 4, "Length of norm_list should be equal to total_itr + 1"
    assert all(
        isinstance(err, float) for err in norm_list
    ), "All elements in norm_list should be floats"


if __name__ == "__main__":
    pytest.main()
