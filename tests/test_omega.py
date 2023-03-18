import torch
import pytest

from methods import SOR

@pytest.fixture
def setup():
    bs, n = 16, 10
    A = torch.randn(bs, n, n)
    b = torch.randn(bs, n)
    x_true, _ = torch.solve(b, A)
    return A, b, x_true

def test_omega(setup):
    A, b, x_true = setup
    num_itr = 10
    omegas = torch.linspace(1.0, 2.0, 11)
    errors = []

    for omega in omegas:
        sor_model = SOR(num_itr=num_itr)
        x_pred, _ = sor_model.forward(num_itr=num_itr, bs=A.shape[0], y=b, omega=omega)
        error = torch.norm(x_pred - x_true, dim=1)
        errors.append(error.mean().item())

    # Check that error decreases as omega increases (up to a certain point)
    assert errors[0] > errors[-1]
    assert errors[-1] < errors[len(omegas)//2]