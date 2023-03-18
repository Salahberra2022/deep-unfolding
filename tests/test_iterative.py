import pytest
import numpy as np
import math
from Iterative import gs_model, ri_model, Jacobi_model, sor_model 
import torch

device = torch.device('cpu') # 'cpu' or 'cuda'
## model parameters 
itr = 25 # iteration steps $T$
num_itr = itr # max. iterations
n = 300 
m = 600 
##


## parameters for evauation of generalization error 
total_itr=25 # total number of iterations (multiple number of "itr")
bs = 10 # number of samples 




def test_code():
    # define test parameters
    itr = 25 # iteration steps $T$
    num_itr = itr # max. iterations
    y = torch.ones(bs, m).to(device)

    # create models
   

    # test GS
    s, traj = gs_model(num_itr, bs, y)
    assert s.shape == (bs, n)
    assert len(traj) == num_itr+1

    # test RI
    s, traj = ri_model(num_itr, bs, y)
    assert s.shape == (bs, n)
    assert len(traj) == num_itr+1

    # test Jacobi
    s, traj = Jacobi_model(num_itr, bs, y)
    assert s.shape == (bs, n)
    assert len(traj) == num_itr+1

    # test SOR
    s, traj = sor_model(num_itr, bs, y)
    assert s.shape == (bs, n)
    assert len(traj) == num_itr+1
    




