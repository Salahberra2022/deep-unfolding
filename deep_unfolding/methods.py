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
total_itr = itr  # Max iterations # ?
n = 300  # Size of matrix (rows)
m = 600  # Size of matrix (columns)

# Parameters for evaluation of generalization error
total_itr = 25  # Total number of iterations (multiple of "itr")
bs = 10  # Number of samples

# Generate A and H
seed_ = 12
np.random.seed(seed=seed_)
H = np.random.normal(0, 1.0 / math.sqrt(n), (n, m))
A = np.dot(H, H.T)
eig = np.linalg.eig(A)
eig = eig[0]  # Eigenvalues

# Convert to Torch tensors and move to device
W = torch.Tensor(np.diag(eig)).to(device)  # Define the appropriate 'device'
H = torch.from_numpy(H).float().to(device)  # Define the appropriate 'device'

# Decmposed matrix calculations
D = np.diag(np.diag(A)) # diagonal matrix
L = np.tril(A, -1) #  lower triangular matrix
U = np.triu(A, 1) # upper triangular matrix
Dinv = np.linalg.inv(D) # inversion diagonal matrix
invM = np.linalg.inv(D + L) #  inversion matrix M
# Convert to Torch tensors and move to device
A = torch.Tensor(A).to(device)  # Define the appropriate 'device'
D = torch.Tensor(D).to(device)  # Define the appropriate 'device'
L = torch.Tensor(L).to(device)  # Define the appropriate 'device'
U = torch.Tensor(U).to(device)  # Define the appropriate 'device'
Dinv = torch.Tensor(Dinv).to(device)  # Define the appropriate 'device'
invM = torch.Tensor(invM).to(device)  # Define the appropriate 'device'

# Print condition number, minimum, and maximum eigenvalues of A
print("Condition number, min. and max. eigenvalues of A:")
print(np.max(eig) / np.min(eig), np.max(eig), np.min(eig))


    ## naive GS with a constant step size 
class GS(nn.Module):
    """Class implementing the Gauss-Seidel algorithm for solving a linear system.

    Args:
        num_itr (int): The number of Gauss-Seidel iterations to perform.

    Attributes:
        num_itr (int): The number of Gauss-Seidel iterations to perform.

    Methods:
        forward(num_itr, bs, y): Performs the Gauss-Seidel iterations and returns the final solution.

    """

    def __init__(self, num_itr):
        """Initialize the Gauss-Seidel solver.

        Args:
            num_itr (int): The number of Gauss-Seidel iterations to perform.

        """
        super(GS, self).__init__()
    def forward(self, num_itr, bs, y):
        """Perform the Gauss-Seidel iterations and return the final solution.

        Args:
            num_itr (int): The number of iterations to perform.
            bs (int): The batch size.
            y (torch.Tensor): The input tensor of shape (bs, n).

        Returns:
            torch.Tensor: The final solution tensor of shape (bs, n).
            list: List containing the trajectory of solutions throughout the iterations.

        """
        traj = []
        s = torch.zeros(bs, n).to(device)
        traj.append(s)

        yMF = torch.matmul(y, H.T)  # Assuming H is defined
        s = torch.matmul(yMF, Dinv)  # Generate batch initial solution vector

        for i in range(num_itr):
            temp = -torch.matmul(s, U) + yMF
            s = torch.matmul(temp, invM)
            traj.append(s)

        return s, traj

gs_model = GS(itr).to(device)
## naive RI with a constant step size 
class RI(nn.Module):
    """Class implementing the Richardson iteration algorithm for solving a linear system.

    Args:
        num_itr (int): The number of Richardson iterations to perform.

    Attributes:
        num_itr (int): The number of Richardson iterations to perform.

    Methods:
        forward(num_itr, bs, y): Performs the Richardson iterations and returns the final solution.

    """

    def __init__(self, num_itr):
        """Initialize the Richardson iteration solver.

        Args:
            num_itr (int): The number of Richardson iterations to perform.

        """
        super(RI, self).__init__()
        

    def forward(self, num_itr, bs, y):
        """Perform the Richardson iterations and return the final solution.

        Args:
            num_itr (int): The number of iterations to perform.
            bs (int): The batch size.
            y (torch.Tensor): The input tensor of shape (bs, n).

        Returns:
            torch.Tensor: The final solution tensor of shape (bs, n).
            list: List containing the trajectory of solutions throughout the iterations.

        """
        traj = []
        omega = torch.tensor(0.25)
        s = torch.zeros(bs, n).to(device)
        traj.append(s)

        yMF = torch.matmul(y, H.T)  # Assuming H is defined
        s = torch.matmul(yMF, Dinv)  # Generate batch initial solution vector

        for i in range(num_itr):
            s = s + torch.mul(omega, (yMF - torch.matmul(s, A)))
            traj.append(s)

        return s, traj


ri_model = RI(itr).to(device)
## naive Jacobi with a constant step size 

class Jacobi(nn.Module):
    """Class implementing the Jacobi iteration algorithm for solving a linear system.

    Args:
        num_itr (int): The number of Jacobi iterations to perform.

    Attributes:
        num_itr (int): The number of Jacobi iterations to perform.

    Methods:
        forward(num_itr, bs, y): Performs the Jacobi iterations and returns the final solution.

    """

    def __init__(self, num_itr):
        """Initialize the Jacobi iteration solver.

        Args:
            num_itr (int): The number of Jacobi iterations to perform.

        """
        super(Jacobi, self).__init__()
     
    def forward(self, num_itr, bs, y):
        """Perform the Jacobi iterations and return the final solution.

        Args:
            num_itr (int): The number of iterations to perform.
            bs (int): The batch size.
            y (torch.Tensor): The input tensor of shape (bs, n).

        Returns:
            torch.Tensor: The final solution tensor of shape (bs, n).
            list: List containing the trajectory of solutions throughout the iterations.

        """
        traj = []
        omega = torch.tensor(0.2)
        s = torch.zeros(bs, n).to(device)
        traj.append(s)

        yMF = torch.matmul(y, H.T)  # Assuming H is defined
        s = torch.matmul(yMF, Dinv)  # Generate batch initial solution vector

        for i in range(num_itr):
            temp = torch.matmul(Dinv, (D - A))
            s = torch.matmul(s, temp) + torch.matmul(yMF, Dinv)
            traj.append(s)

        return s, traj


Jacobi_model = Jacobi(itr).to(device)
## naive SOR with a constant step size 
class SOR(nn.Module):
    """Class implementing the Successive Over-Relaxation (SOR) algorithm for solving a linear system.

    Args:
        num_itr (int): The number of SOR iterations to perform.

    Attributes:
        num_itr (int): The number of SOR iterations to perform.

    Methods:
        forward(num_itr, bs, y): Performs the SOR iterations and returns the final solution.

    """

    def __init__(self, num_itr):
        """Initialize the SOR solver.

        Args:
            num_itr (int): The number of SOR iterations to perform.

        """
        super(SOR, self).__init__()
    def forward(self, num_itr, bs, y):
        """Perform the SOR iterations and return the final solution.

        Args:
            num_itr (int): The number of iterations to perform.
            bs (int): The batch size.
            y (torch.Tensor): The input tensor of shape (bs, n).

        Returns:
            torch.Tensor: The final solution tensor of shape (bs, n).
            list: List containing the trajectory of solutions throughout the iterations.

        """
        traj = []
        n = y.size(1)
        device = y.device

        omega = torch.tensor(1.8)
        inv_omega = torch.div(1, omega)
        invM_sor = torch.linalg.inv(D - torch.mul(inv_omega, L))

        s = torch.zeros(bs, n).to(device)
        traj.append(s)

        yMF = torch.matmul(y, H.T)  # Assuming H is defined
        s = torch.matmul(yMF, Dinv)  # Generate batch initial solution vector

        for i in range(num_itr):
            temp = torch.mul((inv_omega - 1), D) + torch.mul(inv_omega, U)
            s = torch.matmul(s, torch.matmul(invM_sor, temp)) + torch.matmul(yMF, invM_sor)
            traj.append(s)

        return s, traj


sor_model = SOR(itr).to(device)

class SOR_CHEBY(nn.Module):
    """Class implementing the SOR-Chebyshev algorithm for solving a linear system.

    Args:
        num_itr (int): The number of SOR-Chebyshev iterations to perform.

    Attributes:
        num_itr (int): The number of SOR-Chebyshev iterations to perform.

    Methods:
        forward(num_itr, bs, y): Performs the SOR-Chebyshev iterations and returns the final solution.

    """

    def __init__(self, num_itr):
        """Initialize the SOR-Chebyshev solver.

        Args:
            num_itr (int): The number of SOR-Chebyshev iterations to perform.

        """
        super(SOR_CHEBY, self).__init__()
       

    def forward(self, num_itr, bs, y):
        """Perform the SOR-Chebyshev iterations and return the final solution.

        Args:
            num_itr (int): The number of iterations to perform.
            bs (int): The batch size.
            y (torch.Tensor): The input tensor of shape (bs, n).

        Returns:
            torch.Tensor: The final solution tensor of shape (bs, n).
            list: List containing the trajectory of solutions throughout the iterations.

        """
        traj = []
       
        omega = torch.tensor(1.8)
        inv_omega = torch.div(1, omega)
        invM_sor = torch.linalg.inv(D - torch.mul(inv_omega, L))

        s = torch.zeros(bs, n).to(device)
        s_new = torch.zeros(bs, n).to(device)
        traj.append(s)

        yMF = torch.matmul(y, H.T)  # Assuming H is defined
        s = torch.matmul(yMF, Dinv)  # Generate batch initial solution vector

        omegaa = torch.tensor(0.8)
        gamma = torch.tensor(0.8)
        s_present = s
        s_old = torch.zeros(s_present.shape).to(device)

        for i in range(num_itr):
            temp = torch.mul((inv_omega - 1), D) + torch.mul((inv_omega), U)
            s = torch.matmul(s, torch.matmul(invM_sor, temp)) + torch.matmul(yMF, invM_sor)

            s_new = omegaa * (gamma * (s - s_present) + (s_present - s_old)) + s_old
            s_old = s
            s_present = s_new

            traj.append(s_new)

        return s_new, traj


sor_cheby_model = SOR_CHEBY(itr).to(device)

## naive AOR with a constant step size 
class AOR(nn.Module):
    """Class implementing the Accelerated Over-Relaxation (AOR) algorithm for solving a linear system.

    Args:
        num_itr (int): The number of AOR iterations to perform.

    Attributes:
        num_itr (int): The number of AOR iterations to perform.

    Methods:
        forward(num_itr, bs, y): Performs the AOR iterations and returns the final solution.

    """

    def __init__(self, num_itr):
        """Initialize the AOR solver.

        Args:
            num_itr (int): The number of AOR iterations to perform.

        """
        super(AOR, self).__init__()
     
    def forward(self, num_itr, bs, y):
        """Perform the AOR iterations and return the final solution.

        Args:
            num_itr (int): The number of iterations to perform.
            bs (int): The batch size.
            y (torch.Tensor): The input tensor of shape (bs, n).

        Returns:
            torch.Tensor: The final solution tensor of shape (bs, n).
            list: List containing the trajectory of solutions throughout the iterations.

        """
        traj = []
       

        r = torch.tensor(0.2)  # Acceleration parameter
        omega = torch.tensor(0.3)  # Over-relaxation parameter
        inv_omega = torch.div(1, omega)
        M = (D - torch.mul(r, L))
        invM_aor = torch.linalg.inv(M)
        N = (torch.mul((1 - omega), D) + torch.mul((omega - r), L) + torch.mul(omega, U))

        s = torch.zeros(bs, n).to(device)
        traj.append(s)

        yMF = torch.matmul(y, H.T)  # Assuming H is defined
        s = torch.matmul(yMF, Dinv)  # Generate batch initial solution vector

        for i in range(num_itr):
            s = torch.matmul(s, torch.matmul(invM_aor, N)) + torch.mul(omega, torch.matmul(yMF, invM_aor))
            traj.append(s)

        return s, traj


aor_model = AOR(itr).to(device)

## naive AOR_CHEBY with a constant step size 
class AOR_CHEBY(nn.Module):
    """Class implementing the Accelerated Over-Relaxation (AOR) with Chebyshev acceleration algorithm for solving a linear system.

    Args:
        num_itr (int): The number of AOR-Chebyshev iterations to perform.

    Attributes:
        num_itr (int): The number of AOR-Chebyshev iterations to perform.

    Methods:
        forward(num_itr, bs, y): Performs the AOR-Chebyshev iterations and returns the final solution.

    """

    def __init__(self, num_itr):
        """Initialize the AOR-Chebyshev solver.

        Args:
            num_itr (int): The number of AOR-Chebyshev iterations to perform.

        """
        super(AOR_CHEBY, self).__init__()
        

    def forward(self, num_itr, bs, y):
        """Perform the AOR-Chebyshev iterations and return the final solution.

        Args:
            num_itr (int): The number of iterations to perform.
            bs (int): The batch size.
            y (torch.Tensor): The input tensor of shape (bs, n).

        Returns:
            torch.Tensor: The final solution tensor of shape (bs, n).
            list: List containing the trajectory of solutions throughout the iterations.

        """
        traj = []
        

        r = torch.tensor(0.1)  # Acceleration parameter
        omega = torch.tensor(0.1)  # Over-relaxation parameter
        s = torch.zeros(bs, n).to(device)
        traj.append(s)

        yMF = torch.matmul(y, H.T)  # Assuming H is defined
        s = torch.matmul(yMF, Dinv)  # Generate batch initial solution vector

        Y0 = s
        X0 = s

        M = (D - torch.mul(r, L))
        invM = torch.linalg.inv(M)
        N = (torch.mul((1 - omega), D) + torch.mul((omega - r), L) + torch.mul(omega, U))
        temp = torch.matmul(invM, N)

        rho = torch.tensor(0.1)
        mu0 = torch.tensor(1)
        mu1 = rho
        xhat1 = torch.matmul(s, temp) + omega * torch.matmul(yMF, invM)
        Y1 = xhat1
        Y = Y1

        for i in range(num_itr):
            f = 2 / (rho * mu1)
            j = 1 / mu0
            c = f - j
            mu = 1 / c
            a = (2 * mu) / (rho * mu1)
            Y = torch.matmul((Y1 * a), torch.matmul(invM, N)) - (((mu / mu0)) * Y0) + (a * torch.matmul(yMF, invM))
            Y0 = Y1
            Y1 = Y
            mu0 = mu1
            mu1 = mu

            traj.append(Y)

        return Y, traj


aor_cheby_model = AOR_CHEBY(itr).to(device)

solution = torch.normal(0.0*torch.ones(bs,n),1.0).to(device).detach()
y = solution@H.detach()

itr_list = []
## naive GS 
norm_list_GS = [] # Initialize the iteration list
for i in range(total_itr+1):
    s_hat, _ = gs_model(i, bs,y)
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    norm_list_GS.append(err)
    itr_list.append(i)
    ## naive RI 
norm_list_RI = [] # Initialize the iteration list
for i in range(total_itr+1):
    s_hat, _ = ri_model(i, bs,y)
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    norm_list_RI.append(err)
    #itr_list.append(i)
norm_list_Jacobi = [] # Initialize the iteration list
for i in range(total_itr+1):
    s_hat, _ = Jacobi_model(i, bs,y)
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    norm_list_Jacobi.append(err)
    #itr_list.append(i)

## naive SOR 
norm_list_SOR = [] # Initialize the iteration list
for i in range(total_itr+1):
    s_hat, _ = sor_model(i, bs,y)
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    norm_list_SOR.append(err)

## naive SOR_CHEBY 
norm_list_SOR_CHEBY = [] # Initialize the iteration list
for i in range(total_itr+1):
    s_hat, _ = sor_cheby_model(i, bs,y)
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    norm_list_SOR_CHEBY.append(err)


## naive AOR 
norm_list_AOR = [] # Initialize the iteration list
for i in range(total_itr+1):
    s_hat, _ = aor_model(i, bs,y)
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    norm_list_AOR.append(err)
norm_list_AOR_CHEBY = [] # Initialize the iteration list
for i in range(total_itr+1):
    s_hat, _ = aor_cheby_model(i, bs,y)
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    norm_list_AOR_CHEBY.append(err)