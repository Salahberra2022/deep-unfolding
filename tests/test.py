import pytest
import numpy as np
import torch
from your_code_file import GS, RI, Jacobi, SOR, SOR_CHEBY, AOR, AOR_CHEBY

@pytest.fixture
def setup():
    # Set up the necessary variables and models
    itr = 25
    bs = 10
    n = 300
    m = 600
    device = torch.device('cpu')

    # Generate A and H
    np.random.seed(seed=12)
    H = np.random.normal(0, 1.0 / np.sqrt(n), (n, m))
    A = np.dot(H, H.T)
    eig = np.linalg.eig(A)[0]

    # Convert to Torch tensors and move to device
    W = torch.Tensor(np.diag(eig)).to(device)
    H = torch.from_numpy(H).float().to(device)
    A = torch.Tensor(A).to(device)

    # Generate y and solution tensors
    solution = torch.normal(0.0 * torch.ones(bs, n), 1.0).to(device).detach()
    y = solution @ H.detach()

    return A, H, y, solution, itr, bs

def test_gs(setup):
    A, H, y, solution, itr, bs = setup
    model = GS(itr)
    s_hat, _ = model(itr, bs, y)
    error = torch.norm(solution - s_hat) / (n * bs)
    assert error < 1e-6

def test_ri(setup):
    A, H, y, solution, itr, bs = setup
    model = RI(itr)
    s_hat, _ = model(itr, bs, y)
    error = torch.norm(solution - s_hat) / (n * bs)
    assert error < 1e-6

def test_jacobi(setup):
    A, H, y, solution, itr, bs = setup
    model = Jacobi(itr)
    s_hat, _ = model(itr, bs, y)
    error = torch.norm(solution - s_hat) / (n * bs)
    assert error < 1e-6

def test_sor(setup):
    A, H, y, solution, itr, bs = setup
    model = SOR(itr)
    s_hat, _ = model(itr, bs, y)
    error = torch.norm(solution - s_hat) / (n * bs)
    assert error < 1e-6

def test_sor_cheby(setup):
    A, H, y, solution, itr, bs = setup
    model = SOR_CHEBY(itr)
    s_hat, _ = model(itr, bs, y)
    error = torch.norm(solution - s_hat) / (n * bs)
    assert error < 1e-6

def test_aor(setup):
    A, H, y, solution, itr, bs = setup
    model = AOR(itr)
    s_hat, _ = model(itr, bs, y)
    error = torch.norm(solution - s_hat) / (n * bs)
    assert error < 1e-6

def test_aor_cheby(setup):
    A, H, y, solution, itr, bs = setup
    model = AOR_CHEBY(itr)
    s_hat, _ = model(itr, bs, y)
    error = torch.norm(solution - s_hat) / (n * bs)
    assert error < 1e-6
