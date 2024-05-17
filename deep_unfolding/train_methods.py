# Copyright (c) 2022-2023 Salah Berra and contributors
# Distributed under the the GNU General Public License (See accompanying file LICENSE or copy
# at https://www.gnu.org/licenses/)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

device = torch.device('cpu') # 'cpu' or 'cuda'
# Model parameters
itr = 25  # Iteration steps T
total_itr = itr  # Max iterations
n = 300  # Number of rows # ? suppose to be a variable ?
m = 600  # Number of columns # ? suppose to be a variable ?

# Training parameters # ? suppose to be variables ?
bs = 200  # Mini-batch size
num_batch = 500  # Number of mini-batches
lr_adam = 0.002  # Learning rate of optimizer
init_val_SORNet = 1.1  # Initial value of omega for SORNet
init_val_SOR_CHEBY_Net_omega = 0.6  # Initial value of omega for SOR_CHEBY_Net
init_val_SOR_CHEBY_Net_gamma = 0.8  # Initial value of gamma for SOR_CHEBY_Net
init_val_SOR_CHEBY_Net_alpha = 0.9  # Initial value of alpha for SOR_CHEBY_Net
init_val_AORNet_r = 0.9  # Initial value of r for AORNet
init_val_AORNet_omega = 1.5  # Initial value of omega for AORNet
init_val_RINet = 0.1  # Initial value of omega for RINet

# Parameters for evaluation of generalization error
total_itr = 25  # Total number of iterations (multiple of "itr")
bs = 10000  # Number of samples

# Generate A and H
seed_ = 12
np.random.seed(seed=seed_)
H = np.random.normal(0, 1.0 / math.sqrt(n), (n, m))
A = np.dot(H, H.T)
eig = np.linalg.eig(A)
eig = eig[0]  # Eigenvalues

# Convert to Torch tensors and move to device
W = torch.Tensor(np.diag(eig)).to(device)
H = torch.from_numpy(H).float().to(device)

# Additional matrix calculations
D = np.diag(np.diag(A))
U = np.triu(A, 1)
L = np.tril(A, -1)
Dinv = np.linalg.inv(D)
invM = np.linalg.inv(D + L)
A = torch.Tensor(A).to(device)
D = torch.Tensor(D).to(device)
U = torch.Tensor(U).to(device)
L = torch.Tensor(L).to(device)
Dinv = torch.Tensor(Dinv).to(device)
invM = torch.Tensor(invM).to(device)

# Calculate condition number, minimum, and maximum eigenvalues of A
print("Condition number, min. and max. eigenvalues of A:")
print(np.max(eig) / np.min(eig), np.max(eig), np.min(eig))


## Deep unfolded SOR with a constant step size 
class SORNet(nn.Module):
    """Deep unfolded SOR with a constant step size."""

    def __init__(self, num_itr):
        """
        Initialize the SORNet model.

        Args:
            num_itr (int): Number of iterations.

        """
        super(SORNet, self).__init__()
        self.inv_omega = nn.Parameter(init_val_SORNet * torch.ones(1))

    def forward(self, num_itr, bs, y):
        """
        Perform forward pass of the SORNet model.

        Args:
            num_itr (int): Number of iterations.
            bs (int): Batch size.
            y (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
            list: List of intermediate results.

        """
        traj = []

        invM = torch.linalg.inv(torch.mul(self.inv_omega[0], D) + L)
        s = torch.zeros(bs, n).to(device)
        traj.append(s)
        yMF = torch.matmul(y, H.T)
        s = torch.matmul(yMF, Dinv)

        for i in range(num_itr):
            temp = torch.matmul(s, torch.mul((self.inv_omega[0] - 1), D) - U) + yMF
            s = torch.matmul(temp, invM)
            traj.append(s)

        return s, traj

# Model initialization
model_SorNet = SORNet(itr).to(device)
loss_func = nn.MSELoss()
opt1 = optim.Adam(model_SorNet.parameters(), lr=lr_adam)

class SOR_CHEBY_Net(nn.Module):
    """Deep unfolded SOR with Chebyshev acceleration."""

    def __init__(self, num_itr):
        """
        Initialize the SOR_CHEBY_Net model.

        Args:
            num_itr (int): Number of iterations.

        """
        super(SOR_CHEBY_Net, self).__init__()
        self.gamma = nn.Parameter(init_val_SOR_CHEBY_Net_omega * torch.ones(num_itr))
        self.omega = nn.Parameter(init_val_SOR_CHEBY_Net_gamma * torch.ones(num_itr))
        self.inv_omega = nn.Parameter(init_val_SOR_CHEBY_Net_alpha * torch.ones(1))

    def forward(self, num_itr, bs, y):
        """
        Perform forward pass of the SOR_CHEBY_Net model.

        Args:
            num_itr (int): Number of iterations.
            bs (int): Batch size.
            y (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
            list: List of intermediate results.

        """
        traj = []

        invM = torch.linalg.inv(torch.mul(self.inv_omega[0], D) + L)
        s = torch.zeros(bs, n).to(device)
        s_new = torch.zeros(bs, n).to(device)
        traj.append(s)
        yMF = torch.matmul(y, H.T)
        s = torch.matmul(yMF, Dinv)
        s_present = s
        s_old = torch.zeros(s_present.shape).to(device)

        for i in range(num_itr):
            temp = torch.matmul(s, torch.mul((self.inv_omega[0] - 1), D) - U) + yMF
            s = torch.matmul(temp, invM)

            s_new = self.omega[i] * (self.gamma[i] * (s - s_present) + (s_present - s_old)) + s_old
            s_old = s
            s_present = s_new

            traj.append(s_new)

        return s_new, traj

# Model initialization
model_Sor_Cheby_Net = SOR_CHEBY_Net(itr).to(device)
loss_func = nn.MSELoss()
opt2 = optim.Adam(model_Sor_Cheby_Net.parameters(), lr=lr_adam)

## naive AORNet with a constant step size 
class AORNet(nn.Module):
    """Deep unfolded AOR with a constant step size."""

    def __init__(self, num_itr):
        """
        Initialize the AORNet model.

        Args:
            num_itr (int): Number of iterations.

        """
        super(AORNet, self).__init__()
        self.r = nn.Parameter(init_val_AORNet_r * torch.ones(1))
        self.omega = nn.Parameter(init_val_AORNet_omega * torch.ones(1))

    def forward(self, num_itr, bs, y):
        """
        Perform forward pass of the AORNet model.

        Args:
            num_itr (int): Number of iterations.
            bs (int): Batch size.
            y (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
            list: List of intermediate results.

        """
        traj = []

        invM = torch.linalg.inv(L - torch.mul(self.r[0], D))
        N = torch.mul((1 - self.omega[0]), D) + torch.mul((self.omega[0] - self.r[0]), L) + torch.mul(self.omega[0], U)
        s = torch.zeros(bs, n).to(device)
        traj.append(s)
        yMF = torch.matmul(y, H.T)
        s = torch.matmul(yMF, Dinv)

        for i in range(num_itr):
            s = torch.matmul(s, torch.matmul(invM, N)) + torch.matmul(yMF, invM)

        return s, traj

# Model initialization
model_AorNet = AORNet(itr).to(device)
loss_func = nn.MSELoss()
opt3 = optim.Adam(model_AorNet.parameters(), lr=lr_adam)

## naive RI with a constant step size 
class RINet(nn.Module):
    """Deep unfolded Richardson iteration."""

    def __init__(self, num_itr):
        """
        Initialize the RINet model.

        Args:
            num_itr (int): Number of iterations.

        """
        super(RINet, self).__init__()
        self.inv_omega = nn.Parameter(init_val_RINet * torch.ones(1))

    def forward(self, num_itr, bs, y):
        """
        Perform forward pass of the RINet model.

        Args:
            num_itr (int): Number of iterations.
            bs (int): Batch size.
            y (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
            list: List of intermediate results.

        """
        traj = []

        s = torch.zeros(bs, n).to(device)
        traj.append(s)
        yMF = torch.matmul(y, H.T)
        s = torch.matmul(yMF, Dinv)

        for i in range(num_itr):
            s = s + torch.mul(self.inv_omega[0], (yMF - torch.matmul(s, A)))
            traj.append(s)

        return s, traj

# Model initialization
rinet_model = RINet(itr).to(device)
loss_func = nn.MSELoss()
opt4 = optim.Adam(rinet_model.parameters(), lr=lr_adam)

## training process of SORNet
# it takes about several minutes on Google Colaboratory

loss_gen=[]
for gen in range(itr):
    """
    Training process of SORNet.

    Args:
        gen (int): Generation number.

    """
    for i in range(num_batch):
        opt1.zero_grad()
        solution = torch.normal(0.0 * torch.ones(bs, n), 1.0).to(device)
        y = solution @ H
        x_hat, _ = model_SorNet(gen + 1, bs, y)
        loss = loss_func(x_hat, solution)
        loss.backward()
        opt1.step()
        
        if i % 200 == 0:
            print("generation:", gen + 1, " batch:", i, "\t MSE loss:", loss.item())

    loss_gen.append(loss.item())
## training process of SOR_CHEBY_Net
# it takes about several minutes on Google Colaboratory

loss_gen=[]
for gen in range(itr):
    """
    Training process of SOR_CHEBY_Net.

    Args:
        gen (int): Generation number.

    """
    for i in range(num_batch):
        opt2.zero_grad()
        solution = torch.normal(0.0 * torch.ones(bs, n), 1.0).to(device)
        y = solution @ H
        x_hat, _ = model_Sor_Cheby_Net(gen + 1, bs, y)
        loss = loss_func(x_hat, solution)
        loss.backward()
        opt2.step()
        
        if i % 200 == 0:
            print("generation:", gen + 1, " batch:", i, "\t MSE loss:", loss.item())

    loss_gen.append(loss.item())



loss_gen=[]
for gen in range(itr):
    """
    Training process of AorNet.

    Args:
        gen (int): Generation number.

    """
    for i in range(num_batch):
        opt3.zero_grad()
        solution = torch.normal(0.0 * torch.ones(bs, n), 1.0).to(device)
        y = solution @ H
        x_hat, _ = model_AorNet(gen + 1, bs, y)
        loss = loss_func(x_hat, solution)
        loss.backward()
        opt3.step()
        
        if i % 200 == 0:
            print("generation:", gen + 1, " batch:", i, "\t MSE loss:", loss.item())

    loss_gen.append(loss.item())


## training process of RINet
# it takes about several minutes on Google Colaboratory

loss_gen=[]
for gen in range(itr):
    """
    Training process of RINet.

    Args:
        gen (int): Generation number.

    """
    for i in range(num_batch):
        opt4.zero_grad()
        solution = torch.normal(0.0 * torch.ones(bs, n), 1.0).to(device)
        y = solution @ H
        x_hat, _ = rinet_model(gen + 1, bs, y)
        loss = loss_func(x_hat, solution)
        loss.backward()
        opt4.step()
        
        if i % 200 == 0:
            print("generation:", gen + 1, " batch:", i, "\t MSE loss:", loss.item())

    loss_gen.append(loss.item())
itr_list = []
#-----------------
## naive_SORNet
norm_list_SorNet = []

for i in range(total_itr + 1):
    """
    Calculate the mean squared error (MSE) between solution and s_hat using SORNet.

    Args:
        i (int): Current iteration.

    """
    s_hat, _ = model_SorNet(i, bs, y)
    err = (torch.norm(solution.to(device) - s_hat.to(device)) ** 2).item() / (n * bs)
    norm_list_SorNet.append(err)
    itr_list.append(i)

## naive_Sor_Cheby_Ne
norm_list_Sor_Cheby_Net = []
for i in range(total_itr + 1):
    """
    Calculate the mean squared error (MSE) between solution and s_hat using SOR_CHEBY_Net.

    Args:
        i (int): Current iteration.

    """
    s_hat, _ = model_Sor_Cheby_Net(i, bs, y)
    err = (torch.norm(solution.to(device) - s_hat.to(device)) ** 2).item() / (n * bs)
    norm_list_Sor_Cheby_Net.append(err)
    ## naive_AORNet
norm_list_AorNet = []

for i in range(total_itr + 1):
    """
    Calculate the mean squared error (MSE) between solution and s_hat using AorNet.

    Args:
        i (int): Current iteration.

    """
    s_hat, _ = model_AorNet(i, bs, y)
    err = (torch.norm(solution.to(device) - s_hat.to(device)) ** 2).item() / (n * bs)
    norm_list_AorNet.append(err)
## naive_RINet
norm_list_rinet = []
for i in range(total_itr + 1):
    """
    Calculate the mean squared error (MSE) between solution and s_hat using rinet_model.

    Args:
        i (int): Current iteration.

    """
    s_hat, _ = rinet_model(i, bs, y)
    err = (torch.norm(solution.to(device) - s_hat.to(device)) ** 2).item() / (n * bs)
    norm_list_rinet.append(err)