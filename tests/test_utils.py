from unfolding_linear.utils import generate_A_H_sol, decompose_matrix
import pytest
import torch
import numpy as np

@pytest.fixture
def generate_data():
    n = 300
    m = 600
    seed = 12
    bs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # You can also test on GPU if available
    
    return generate_A_H_sol(n, m, seed, bs, device)

def test_A_shape(generate_data):
    A, _, _, _, _ = generate_data
    assert A.shape == (300, 300), "The matrix A must be of form (n, n)"

def test_H_shape(generate_data):
    _, H, _, _, _ = generate_data
    assert H.shape == (300, 600), "The matrix H must be of the form (n, m)"

def test_W_shape(generate_data):
    _, _, W, _, _ = generate_data
    assert W.shape == (300, 300), "The matrix W must be of the form (n, n)"

def test_solution_shape(generate_data):
    _, _, _, solution, _ = generate_data
    assert solution.shape == (10, 300), "The solution tensor must be of the form (bs, n)"

def test_y_shape(generate_data):
    _, _, _, _, y = generate_data
    assert y.shape == (10, 600), "The tensor y must be of form (bs, m)"

def test_A_symmetry(generate_data):
    A, _, _, _, _ = generate_data
    assert np.allclose(A, A.T), "The matrix A must be symmetrical"

def test_A_positive_definite(generate_data):
    A, _, _, _, _ = generate_data
    eigvals = np.linalg.eigvals(A)
    assert np.all(eigvals > 0), "All eigenvalues of A must be positive"

def test_y_calculation(generate_data):
    _, H, _, solution, y = generate_data
    y_calculated = solution @ H
    assert torch.allclose(y, y_calculated), "The y tensor must be equal to solution @ H"

@pytest.fixture
def decompose_data():
    A = np.random.rand(5, 5)
    return decompose_matrix(A)

def test_decompose_matrix_shapes(decompose_data):
    A, D, L, U, Dinv, Minv = decompose_data
    assert A.shape == (5, 5), "Matrix A should be of shape (5, 5)"
    assert D.shape == (5, 5), "Matrix D should be of shape (5, 5)"
    assert L.shape == (5, 5), "Matrix L should be of shape (5, 5)"
    assert U.shape == (5, 5), "Matrix U should be of shape (5, 5)"
    assert Dinv.shape == (5, 5), "Matrix Dinv should be of shape (5, 5)"
    assert Minv.shape == (5, 5), "Matrix Minv should be of shape (5, 5)"

def test_diagonal_matrix(decompose_data):
    _, D, _, _, _, _ = decompose_data
    assert torch.allclose(D, torch.diag(torch.diag(D))), "Matrix D should be diagonal"

def test_lower_triangular_matrix(decompose_data):
    _, _, L, _, _, _ = decompose_data
    assert torch.allclose(L, torch.tril(L)), "Matrix L should be lower triangular"

def test_upper_triangular_matrix(decompose_data):
    _, _, _, U, _, _ = decompose_data
    assert torch.allclose(U, torch.triu(U)), "Matrix U should be upper triangular"

def test_inverse_diagonal(decompose_data):
    _, D, _, _, Dinv, _ = decompose_data
    identity = torch.eye(D.shape[0]).to(D.device)
    assert torch.allclose(D @ Dinv, identity, atol=1e-6), "Dinv should be the inverse of D"

def test_inverse_D_plus_L(decompose_data):
    _, D, L, _, _, Minv = decompose_data
    identity = torch.eye(D.shape[0]).to(D.device)
    assert torch.allclose((D + L) @ Minv, identity, atol=1e-6), "Minv should be the inverse of D + L"

if __name__ == "__main__":
    pytest.main()
