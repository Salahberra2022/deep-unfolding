# Copyright (c) 2023-2024 Salah Berra and contributors
# Distributed under the the GNU General Public License (See accompanying file
# LICENSE or copy at https://www.gnu.org/licenses/)

import numpy as np
import pytest
import torch

from deep_unfolding import (
    AOR,
    SOR,
    AORCheby,
    GaussSeidel,
    Jacobi,
    Richardson,
    SORCheby,
    _decompose_matrix,
    _device,
)

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


@pytest.fixture(scope="module", params=_seeds)
def seed_to_test(request):
    """Seeds to test."""
    return request.param


@pytest.fixture(scope="module", params=_ns)
def n_to_test(request):
    """Values of n to test."""
    return request.param


@pytest.fixture(scope="module", params=_ms)
def m_to_test(request):
    """Values of m to test."""
    return request.param


@pytest.fixture(scope="module", params=_bss)
def bs_to_test(request):
    """Values of bs to test."""
    return request.param


@pytest.fixture(scope="module", params=_devices)
def device_to_test(request):
    """Devices to test."""
    return request.param


@pytest.fixture(scope="module")
def common_data_to_test(seed_to_test, n_to_test, m_to_test, bs_to_test, device_to_test):
    """Collate common data for all models."""
    rng = np.random.default_rng(seed_to_test)
    A = rng.random((n_to_test, n_to_test))
    H = torch.from_numpy(rng.random((n_to_test, m_to_test))).float().to(_device)
    y = torch.from_numpy(rng.random((bs_to_test, m_to_test))).float().to(_device)
    return n_to_test, A, H, bs_to_test, y, device_to_test


# ######################################### #
# ### Data fixtures for specific models ### #
# ######################################### #


@pytest.fixture(scope="module", params=_ri_omegas)
def ri_omega_to_test(request):
    return request.param


@pytest.fixture(scope="module", params=_jac_omegas)
def jac_omega_to_test(request):
    return request.param


@pytest.fixture(scope="module", params=_sor_omegas)
def sor_omega_to_test(request):
    return request.param


@pytest.fixture(scope="module", params=_sorcheb_omegas)
def sorcheb_omega_to_test(request):
    return request.param


@pytest.fixture(scope="module", params=_sorcheb_omegaas)
def sorcheb_omegaa_to_test(request):
    return request.param


@pytest.fixture(scope="module", params=_sorcheb_gammas)
def sorcheb_gamma_to_test(request):
    return request.param


@pytest.fixture(scope="module", params=_aor_omegas)
def aor_omega_to_test(request):
    return request.param


@pytest.fixture(scope="module", params=_aor_rs)
def aor_r_to_test(request):
    return request.param


@pytest.fixture(scope="module", params=_aorcheb_omegas)
def aorcheb_omega_to_test(request):
    return request.param


@pytest.fixture(scope="module", params=_aorcheb_rs)
def aorcheb_r_to_test(request):
    return request.param


# ################################# #
# #### Model creation fixtures #### #
# ################################# #


@pytest.fixture(scope="module")
def gs_model(common_data_to_test):
    """Create and return a Gauss-Seidel model."""
    n, A, H, bs, y, device = common_data_to_test
    return GaussSeidel(n, A, H, bs, y, device)


@pytest.fixture(scope="module")
def ri_model(common_data_to_test, ri_omega_to_test):
    """Create and return a Richardson model."""
    n, A, H, bs, y, device = common_data_to_test
    return Richardson(n, A, H, bs, y, ri_omega_to_test, device)


@pytest.fixture(scope="module")
def jac_model(common_data_to_test, jac_omega_to_test):
    """Create and return a Jacobi model."""
    n, A, H, bs, y, device = common_data_to_test
    return Jacobi(n, A, H, bs, y, jac_omega_to_test, device)


@pytest.fixture(scope="module")
def sor_model(common_data_to_test, sor_omega_to_test):
    """Create and return a SOR model."""
    n, A, H, bs, y, device = common_data_to_test
    return SOR(n, A, H, bs, y, sor_omega_to_test, device)


@pytest.fixture(scope="module")
def sorcheb_model(
    common_data_to_test,
    sorcheb_omega_to_test,
    sorcheb_omegaa_to_test,
    sorcheb_gamma_to_test,
):
    """Create and return a SOR-Chebyshev model."""
    n, A, H, bs, y, device = common_data_to_test
    return SORCheby(
        n,
        A,
        H,
        bs,
        y,
        sorcheb_omega_to_test,
        sorcheb_omegaa_to_test,
        sorcheb_gamma_to_test,
        device,
    )


@pytest.fixture(scope="module")
def aor_model(common_data_to_test, aor_omega_to_test, aor_r_to_test):
    """Create and return an AOR model."""
    n, A, H, bs, y, device = common_data_to_test
    return AOR(n, A, H, bs, y, aor_omega_to_test, aor_r_to_test, device)


@pytest.fixture(scope="module")
def aorcheb_model(common_data_to_test, aorcheb_omega_to_test, aorcheb_r_to_test):
    """Create and return an AOR model."""
    n, A, H, bs, y, device = common_data_to_test
    return AORCheby(n, A, H, bs, y, aorcheb_omega_to_test, aorcheb_r_to_test, device)


# ######################## #
# ### Helper functions ### #
# ######################## #


def common_initialization_tests(itmodel, common_data_to_test):
    """Perform common initialization checks."""
    n, A, H, bs, y, device = common_data_to_test

    assert itmodel.n == n, "Attribute n should be initialized correctly"
    assert itmodel.H.shape == H.shape, "Attribute H should be initialized correctly"
    assert itmodel.bs == bs, "Attribute bs should be initialized correctly"
    assert itmodel.y.shape == y.shape, "Attribute y should be initialized correctly"
    assert itmodel.device == device, "Device should be initialized correctly"

    A_torch, D, L, U, Dinv, Minv = _decompose_matrix(A, device)
    assert torch.allclose(
        itmodel.A, A_torch
    ), "Attribute A should match the decomposed matrix"
    assert torch.allclose(
        itmodel.D, D
    ), "Attribute D should match the decomposed matrix"
    assert torch.allclose(
        itmodel.L, L
    ), "Attribute L should match the decomposed matrix"
    assert torch.allclose(
        itmodel.U, U
    ), "Attribute U should match the decomposed matrix"
    assert torch.allclose(
        itmodel.Dinv, Dinv
    ), "Attribute Dinv should match the decomposed matrix"
    assert torch.allclose(
        itmodel.Minv, Minv
    ), "Attribute Minv should match the decomposed matrix"


# ############################ #
# ### Initialization tests ### #
# ############################ #


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


if __name__ == "__main__":
    pytest.main()
