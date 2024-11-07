# Copyright (c) 2023-2024 Salah Berra and contributors
# Distributed under the the GNU General Public License (See accompanying file
# LICENSE or copy at https://www.gnu.org/licenses/)

import numpy as np
import pytest
import torch

from deep_unfolding import (AOR, SOR, AORCheby, GaussSeidel, Jacobi,
                            Richardson, SORCheby)
from deep_unfolding.utils import _decompose_matrix

# ############################################################ #
# ############### Parameters to use in tests ################# #
# Create more combinations by adding values to the lists below #
# ############################################################ #

_seeds = [123]
_ns = [300]
_ms = [600]
_bss = [10]
_devices = ["cpu"]
_ri_omegas = [0.25]
_jac_omegas = [0.2]
_sor_omegas = [1.8]
_sorcheb_omegas = [1.8]
_sorcheb_omegaas = [0.8]
_sorcheb_gammas = [0.8]
_aor_omegas = [0.3]
_aor_rs = [0.2]
_aorcheb_omegas = [0.1]
_aorcheb_rs = [0.1]

# ########################################################################### #
# ################# Common data fixtures for all models ##################### #
# By parameterizing fixtures we create combinations of all parameters to test #
# ########################################################################### #


@pytest.fixture(params=_seeds)
def seed_for_tests(request):
    """Seeds to test."""
    return request.param


@pytest.fixture(params=_ns)
def n_to_test(request):
    """Values of n to test."""
    return request.param


@pytest.fixture(params=_ms)
def m_to_test(request):
    """Values of m to test."""
    return request.param


@pytest.fixture(params=_bss)
def bs_to_test(request):
    """Values of bs to test."""
    return request.param


@pytest.fixture(params=_devices)
def device_to_test(request):
    """Devices to test."""
    return request.param


@pytest.fixture()
def rng_for_tests(seed_for_tests):
    """Provides a PRNG for reproducible tests."""
    return np.random.default_rng(seed_for_tests)


@pytest.fixture()
def common_data_to_test(
    rng_for_tests, n_to_test, m_to_test, bs_to_test, device_to_test
):
    """Collate common data for all models."""
    A = rng_for_tests.random((n_to_test, n_to_test))
    H = (
        torch.from_numpy(rng_for_tests.random((n_to_test, m_to_test)))
        .float()
        .to(device_to_test)
    )
    y = (
        torch.from_numpy(rng_for_tests.random((bs_to_test, m_to_test)))
        .float()
        .to(device_to_test)
    )
    return A, H, bs_to_test, y, device_to_test


# ######################################### #
# ### Data fixtures for specific models ### #
# ######################################### #


@pytest.fixture(params=_ri_omegas)
def ri_omega_to_test(request):
    return request.param


@pytest.fixture(params=_jac_omegas)
def jac_omega_to_test(request):
    return request.param


@pytest.fixture(params=_sor_omegas)
def sor_omega_to_test(request):
    return request.param


@pytest.fixture(params=_sorcheb_omegas)
def sorcheb_omega_to_test(request):
    return request.param


@pytest.fixture(params=_sorcheb_omegaas)
def sorcheb_omegaa_to_test(request):
    return request.param


@pytest.fixture(params=_sorcheb_gammas)
def sorcheb_gamma_to_test(request):
    return request.param


@pytest.fixture(params=_aor_omegas)
def aor_omega_to_test(request):
    return request.param


@pytest.fixture(params=_aor_rs)
def aor_r_to_test(request):
    return request.param


@pytest.fixture(params=_aorcheb_omegas)
def aorcheb_omega_to_test(request):
    return request.param


@pytest.fixture(params=_aorcheb_rs)
def aorcheb_r_to_test(request):
    return request.param


# ################################# #
# #### Model creation fixtures #### #
# ################################# #


@pytest.fixture()
def gs_model(common_data_to_test):
    """Create and return a Gauss-Seidel model."""
    A, H, bs, y, device = common_data_to_test
    return GaussSeidel(A, H, bs, y, device)


@pytest.fixture()
def ri_model(common_data_to_test, ri_omega_to_test):
    """Create and return a Richardson model."""
    A, H, bs, y, device = common_data_to_test
    return Richardson(A, H, bs, y, ri_omega_to_test, device)


@pytest.fixture()
def jac_model(common_data_to_test, jac_omega_to_test):
    """Create and return a Jacobi model."""
    A, H, bs, y, device = common_data_to_test
    return Jacobi(A, H, bs, y, jac_omega_to_test, device)


@pytest.fixture()
def sor_model(common_data_to_test, sor_omega_to_test):
    """Create and return a SOR model."""
    A, H, bs, y, device = common_data_to_test
    return SOR(A, H, bs, y, sor_omega_to_test, device)


@pytest.fixture()
def sorcheb_model(
    common_data_to_test,
    sorcheb_omega_to_test,
    sorcheb_omegaa_to_test,
    sorcheb_gamma_to_test,
):
    """Create and return a SOR-Chebyshev model."""
    A, H, bs, y, device = common_data_to_test
    return SORCheby(
        A,
        H,
        bs,
        y,
        sorcheb_omega_to_test,
        sorcheb_omegaa_to_test,
        sorcheb_gamma_to_test,
        device,
    )


@pytest.fixture()
def aor_model(common_data_to_test, aor_omega_to_test, aor_r_to_test):
    """Create and return an AOR model."""
    A, H, bs, y, device = common_data_to_test
    return AOR(A, H, bs, y, aor_omega_to_test, aor_r_to_test, device)


@pytest.fixture()
def aorcheb_model(common_data_to_test, aorcheb_omega_to_test, aorcheb_r_to_test):
    """Create and return an AOR model."""
    A, H, bs, y, device = common_data_to_test
    return AORCheby(A, H, bs, y, aorcheb_omega_to_test, aorcheb_r_to_test, device)


# ############################ #
# ### Initialization tests ### #
# ############################ #


def common_initialization_tests(itmodel, common_data_to_test):
    """Helper function to perform common initialization checks."""
    A, H, bs, y, device = common_data_to_test

    assert itmodel._H.shape == H.shape, "Attribute H should be initialized correctly"
    assert itmodel._bs == bs, "Attribute bs should be initialized correctly"
    assert itmodel._y.shape == y.shape, "Attribute y should be initialized correctly"
    assert itmodel._device == device, "Device should be initialized correctly"

    A_torch, D, L, U, Dinv, Minv = _decompose_matrix(A, device)
    assert torch.allclose(
        itmodel._A, A_torch
    ), "Attribute A should match the decomposed matrix"
    assert torch.allclose(
        itmodel._D, D
    ), "Attribute D should match the decomposed matrix"
    assert torch.allclose(
        itmodel._L, L
    ), "Attribute L should match the decomposed matrix"
    assert torch.allclose(
        itmodel._U, U
    ), "Attribute U should match the decomposed matrix"
    assert torch.allclose(
        itmodel._Dinv, Dinv
    ), "Attribute Dinv should match the decomposed matrix"
    assert torch.allclose(
        itmodel._Minv, Minv
    ), "Attribute Minv should match the decomposed matrix"


def test_gs_initialization(gs_model, common_data_to_test):
    """Test initialization of the Gauss-Seidel model."""
    common_initialization_tests(gs_model, common_data_to_test)


def test_ri_initialization(ri_model, common_data_to_test, ri_omega_to_test):
    """Test initialization of the Richardson model."""
    common_initialization_tests(ri_model, common_data_to_test)
    assert (
        ri_model.omega == ri_omega_to_test
    ), "Richardson omega should be initialized correctly"


def test_jac_initialization(jac_model, common_data_to_test, jac_omega_to_test):
    """Test initialization of the Jacobi model."""
    common_initialization_tests(jac_model, common_data_to_test)
    assert (
        jac_model.omega == jac_omega_to_test
    ), "Jacobi omega should be initialized correctly"


def test_sor_initialization(sor_model, common_data_to_test, sor_omega_to_test):
    """Test initialization of the SOR model."""
    common_initialization_tests(sor_model, common_data_to_test)
    assert (
        sor_model.omega == sor_omega_to_test
    ), "SOR omega should be initialized correctly"


def test_sorcheb_initialization(
    sorcheb_model,
    common_data_to_test,
    sorcheb_omega_to_test,
    sorcheb_omegaa_to_test,
    sorcheb_gamma_to_test,
):
    """Test initialization of the SOR-Chebyshev model."""
    common_initialization_tests(sorcheb_model, common_data_to_test)
    assert (
        sorcheb_model.omega == sorcheb_omega_to_test
    ), "SOR-Cheby omega should be initialized correctly"
    assert (
        sorcheb_model.omegaa == sorcheb_omegaa_to_test
    ), "SOR-Cheby omegaa should be initialized correctly"
    assert (
        sorcheb_model.gamma == sorcheb_gamma_to_test
    ), "SOR-Cheby gamma should be initialized correctly"


def test_aor_initialization(
    aor_model, common_data_to_test, aor_omega_to_test, aor_r_to_test
):
    """Test initialization of the AOR model."""
    common_initialization_tests(aor_model, common_data_to_test)
    assert (
        aor_model.omega == aor_omega_to_test
    ), "AOR omega should be initialized correctly"
    assert aor_model.r == aor_r_to_test, "AOR r should be initialized correctly"


def test_aorcheb_initialization(
    aorcheb_model, common_data_to_test, aorcheb_omega_to_test, aorcheb_r_to_test
):
    """Test initialization of the AOR-Chebyshev model."""
    common_initialization_tests(aorcheb_model, common_data_to_test)
    assert (
        aorcheb_model.omega == aorcheb_omega_to_test
    ), "AOR-Cheby omega should be initialized correctly"
    assert (
        aorcheb_model.r == aorcheb_r_to_test
    ), "AOR-Cheby r should be initialized correctly"


# ######################## #
# ### _iterate() tests ### #
# ######################## #

# Values to test for num_itr
_num_itr_to_test = [5, 10]


def common_iterate_tests(itmodel, common_data_to_test, num_itr):
    """Common code for testing _iterate() for each model."""
    A, _, bs, _, _ = common_data_to_test

    yMF = torch.matmul(itmodel._y, itmodel._H.T).to(itmodel._device)
    s = torch.matmul(yMF, itmodel._Dinv)
    itmodel._iterate(num_itr, yMF, s)

    assert len(itmodel._s_hats) == num_itr, "Trajectory should contain num_itr elements"
    assert itmodel._s_hats[0].shape == (
        bs,
        A.shape[0],
    ), "Each element in the trajectory should have shape (bs, n)"

    # Since we called _iterate() and not solved, the instance should have
    # _solved set to False
    assert not itmodel._solved, "Model instance should not yet be solved"

    # Trying to get the final solution at this time should yield a RuntimeError
    with pytest.raises(RuntimeError, match="Problem has not been solved yet!"):
        _ = itmodel.s_hat.shape

    # Mark it as solved manually (client code should never do this directly)
    itmodel._solved = True

    # Check that it worked
    assert itmodel._solved, "Model instance should have been solved"

    # Now we can check the final solution (after we have set the _solved flag to True)
    assert itmodel.s_hat.shape == (
        bs,
        A.shape[0],
    ), "Final solution tensor should have shape (bs, n)"


@pytest.mark.parametrize("num_itr", _num_itr_to_test)
def test_gs_iterate(gs_model, common_data_to_test, num_itr):
    """Test iteration of the Gauss-Seidel model."""
    common_iterate_tests(gs_model, common_data_to_test, num_itr)


@pytest.mark.parametrize("num_itr", _num_itr_to_test)
def test_ri_iterate(ri_model, common_data_to_test, num_itr):
    """Test iteration of the Richardson model."""
    common_iterate_tests(ri_model, common_data_to_test, num_itr)


@pytest.mark.parametrize("num_itr", _num_itr_to_test)
def test_jac_iterate(jac_model, common_data_to_test, num_itr):
    """Test iteration of the Jacobi model."""
    common_iterate_tests(jac_model, common_data_to_test, num_itr)


@pytest.mark.parametrize("num_itr", _num_itr_to_test)
def test_sor_iterate(sor_model, common_data_to_test, num_itr):
    """Test iteration of the SOR model."""
    common_iterate_tests(sor_model, common_data_to_test, num_itr)


@pytest.mark.parametrize("num_itr", _num_itr_to_test)
def test_sorcheb_iterate(sorcheb_model, common_data_to_test, num_itr):
    """Test iteration of the SOR-Chebyshev model."""
    common_iterate_tests(sorcheb_model, common_data_to_test, num_itr)


@pytest.mark.parametrize("num_itr", _num_itr_to_test)
def test_aor_iterate(aor_model, common_data_to_test, num_itr):
    """Test iteration of the AOR model."""
    common_iterate_tests(aor_model, common_data_to_test, num_itr)


@pytest.mark.parametrize("num_itr", _num_itr_to_test)
def test_aorcheb_iterate(aorcheb_model, common_data_to_test, num_itr):
    """Test iteration of the AOR-Chebyshev model."""
    common_iterate_tests(aorcheb_model, common_data_to_test, num_itr)


# ##################### #
# ### solve() tests ### #
# ##################### #


# Values to test for total_itr
_total_itr = [5, 10]


def common_solve_tests(itmodel, total_itr):
    """Common code for testing solve() for each model."""

    s_hats = itmodel.solve(total_itr)

    assert len(s_hats) == total_itr + 1, "s_hats should contain total_itr + 1 elements"


@pytest.mark.parametrize("total_itr", _total_itr)
def test_gs_solve(gs_model, total_itr):
    """Test solve() for the Gauss-Seidel model."""
    common_solve_tests(gs_model, total_itr)


@pytest.mark.parametrize("total_itr", _total_itr)
def test_ri_solve(ri_model, total_itr):
    """Test solve() for the Richardson model."""
    common_solve_tests(ri_model, total_itr)


@pytest.mark.parametrize("total_itr", _total_itr)
def test_jac_solve(jac_model, total_itr):
    """Test solve() for the Jacobi model."""
    common_solve_tests(jac_model, total_itr)


@pytest.mark.parametrize("total_itr", _total_itr)
def test_sor_solve(sor_model, total_itr):
    """Test solve() for the SOR model."""
    common_solve_tests(sor_model, total_itr)


@pytest.mark.parametrize("total_itr", _total_itr)
def test_sorcheb_solve(sorcheb_model, total_itr):
    """Test solve() for the SOR-Chebyshev model."""
    common_solve_tests(sorcheb_model, total_itr)


@pytest.mark.parametrize("total_itr", _total_itr)
def test_aor_solve(aor_model, total_itr):
    """Test solve() for the AOR model."""
    common_solve_tests(aor_model, total_itr)


@pytest.mark.parametrize("total_itr", _total_itr)
def test_aorcheb_solve(aorcheb_model, total_itr):
    """Test solve() for the AOR-Chebyshev model."""
    common_solve_tests(aorcheb_model, total_itr)


# ####################################### #
# ### Run this test module separately ### #
# ####################################### #

if __name__ == "__main__":
    pytest.main()
