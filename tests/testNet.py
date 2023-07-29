import pytest
import torch
from methods import itr_list, norm_list_GS, norm_list_RI, norm_list_Jacobi, norm_list_SOR, norm_list_AOR, norm_list_AOR_CHEBY, norm_list_SOR_CHEBY
from train_methods import itr_list, norm_list_SorNet, norm_list_Sor_Cheby_Net, norm_list_AorNet, norm_list_rinet
  

# Create test cases for SORNet
@pytest.fixture
def sor_net_model():
    return SORNet(itr=25)

def test_sor_net_forward(sor_net_model):
    bs = 10  # Batch size
    num_itr = 5  # Number of iterations
    y = torch.randn(bs, 300)  # Input tensor
    output, _ = sor_net_model.forward(num_itr, bs, y)
    assert output.shape == (bs, 300)  # Check the shape of the output

# Create test cases for SOR_CHEBY_Net
@pytest.fixture
def sor_cheby_net_model():
    return SOR_CHEBY_Net(itr=25)

def test_sor_cheby_net_forward(sor_cheby_net_model):
    bs = 10  # Batch size
    num_itr = 5  # Number of iterations
    y = torch.randn(bs, 300)  # Input tensor
    output, _ = sor_cheby_net_model.forward(num_itr, bs, y)
    assert output.shape == (bs, 300)  # Check the shape of the output

# Create test cases for AORNet
@pytest.fixture
def aor_net_model():
    return AORNet(itr=25)

def test_aor_net_forward(aor_net_model):
    bs = 10  # Batch size
    num_itr = 5  # Number of iterations
    y = torch.randn(bs, 300)  # Input tensor
    output, _ = aor_net_model.forward(num_itr, bs, y)
    assert output.shape == (bs, 300)  # Check the shape of the output

# Create test cases for RINet
@pytest.fixture
def ri_net_model():
    return RINet(itr=25)

def test_ri_net_forward(ri_net_model):
    bs = 10  # Batch size
    num_itr = 5  # Number of iterations
    y = torch.randn(bs, 300)  # Input tensor
    output, _ = ri_net_model.forward(num_itr, bs, y)
    assert output.shape == (bs, 300)  # Check the shape of the output

# Run the tests
if __name__ == '__main__':
    pytest.main()
