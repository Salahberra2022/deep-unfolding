import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from train_methods import SORNet, SOR_CHEBY_Net, AORNet, RINet

# Model parameters and other settings (remain the same as in the previous code)

# ... (The rest of the code remains the same)

# Test functions
def test_SORNet_parameters():
    # Test parameters of SORNet
    num_itr = 25
    bs = 10
    y = torch.ones(bs, m).to(device)

    model_SorNet = SORNet(num_itr).to(device)
    assert model_SorNet.inv_omega.shape == torch.Size([1])
    assert model_SorNet.inv_omega[0] == init_val_SORNet

def test_SOR_CHEBY_Net_parameters():
    # Test parameters of SOR_CHEBY_Net
    num_itr = 25
    bs = 10
    y = torch.ones(bs, m).to(device)

    model_Sor_Cheby_Net = SOR_CHEBY_Net(num_itr).to(device)
    assert model_Sor_Cheby_Net.gamma.shape == torch.Size([num_itr])
    assert model_Sor_Cheby_Net.gamma[0] == init_val_SOR_CHEBY_Net_omega
    assert model_Sor_Cheby_Net.omega.shape == torch.Size([num_itr])
    assert model_Sor_Cheby_Net.omega[0] == init_val_SOR_CHEBY_Net_gamma
    assert model_Sor_Cheby_Net.inv_omega.shape == torch.Size([1])
    assert model_Sor_Cheby_Net.inv_omega[0] == init_val_SOR_CHEBY_Net_alpha

def test_AORNet_parameters():
    # Test parameters of AORNet
    num_itr = 25
    bs = 10
    y = torch.ones(bs, m).to(device)

    model_AorNet = AORNet(num_itr).to(device)
    assert model_AorNet.r.shape == torch.Size([1])
    assert model_AorNet.r[0] == init_val_AORNet_r
    assert model_AorNet.omega.shape == torch.Size([1])
    assert model_AorNet.omega[0] == init_val_AORNet_omega

def test_RINet_parameters():
    # Test parameters of RINet
    num_itr = 25
    bs = 10
    y = torch.ones(bs, m).to(device)

    rinet_model = RINet(num_itr).to(device)
    assert rinet_model.inv_omega.shape == torch.Size([1])
    assert rinet_model.inv_omega[0] == init_val_RINet

# To run all the tests at once, you can call the following function
# This will execute all the individual test functions
def test_all_models_parameters():
    test_SORNet_parameters()
    test_SOR_CHEBY_Net_parameters()
    test_AORNet_parameters()
    test_RINet_parameters()

# Call the test function
test_all_models_parameters()