import pytest
import numpy as np
import math
from Iterative import gs_model, ri_model, Jacobi_model, sor_model, sor_cheby_model, aor_model, aor_cheby_model, solution
import torch

device = torch.device('cpu') # 'cpu' or 'cuda'
## model parameters 
itr = 25 # iteration steps $T$
num_itr = itr # max. iterations
n = 300 
m = 600 
##


## parameters for the evaluation of generalization error 
total_itr = 25 # total number of iterations (multiple number of "itr")
bs = 10 # number of samples

# Individual test functions

def test_gs_model():
    # define test parameters
    num_itr = 25 # iteration steps $T$
    bs = 10 # number of samples
    y = torch.ones(bs, m).to(device)

    # test GS
    s, traj = gs_model(num_itr, bs, y)
    assert s.shape == (bs, n)
    assert len(traj) == num_itr + 1

def test_ri_model():
    # define test parameters
    num_itr = 25 # iteration steps $T$
    bs = 10 # number of samples
    y = torch.ones(bs, m).to(device)

    # test RI
    s, traj = ri_model(num_itr, bs, y)
    assert s.shape == (bs, n)
    assert len(traj) == num_itr + 1

def test_Jacobi_model():
    # define test parameters
    num_itr = 25 # iteration steps $T$
    bs = 10 # number of samples
    y = torch.ones(bs, m).to(device)

    # test Jacobi
    s, traj = Jacobi_model(num_itr, bs, y)
    assert s.shape == (bs, n)
    assert len(traj) == num_itr + 1

def test_sor_model():
    # define test parameters
    num_itr = 25 # iteration steps $T$
    bs = 10 # number of samples
    y = torch.ones(bs, m).to(device)

    # test SOR
    s, traj = sor_model(num_itr, bs, y)
    assert s.shape == (bs, n)
    assert len(traj) == num_itr + 1