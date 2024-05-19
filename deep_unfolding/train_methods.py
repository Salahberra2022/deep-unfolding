from utils import device, decompose_matrix
import torch
import torch.nn as nn

def train_model(model, optimizer, loss_func, total_itr, solution, num_batch) : 
    loss_gen=[]
    for gen in range(total_itr):
        """
        Training process of SORNet.

        Args:
            gen (int): Generation number.

        """
        for i in range(num_batch):
            optimizer.zero_grad()
            x_hat, _ = model(gen + 1)
            loss = loss_func(x_hat, solution)
            loss.backward()
            optimizer.step()
            
            if i % 200 == 0:
                print("generation:", gen + 1, " batch:", i, "\t MSE loss:", loss.item())
        loss_gen.append(loss.item())
    return model, loss_gen

def evaluate_model(model, solution, n, bs, total_itr, device=device) : 
    norm_list = []
    for i in range(total_itr + 1):
        """
        Calculate the mean squared error (MSE) between solution and s_hat using SORNet.

        Args:
            i (int): Current iteration.

        """
        s_hat, _ = model(i)
        err = (torch.norm(solution.to(device) - s_hat.to(device)) ** 2).item() / (n * bs)
        norm_list.append(err)
    return norm_list

class SORNet(nn.Module):
    """Deep unfolded SOR with a constant step size."""

    def __init__(self, init_val_SORNet, A, H, bs, y, device=device):
        """
        Initialize the SORNet model.

        Args:
            num_itr (int): Number of iterations.
            init_val_SORNet (float): Initial value for inv_omega.
            D (torch.Tensor): Diagonal matrix D.
            L (torch.Tensor): Lower triangular matrix L.
            U (torch.Tensor): Upper triangular matrix U.
            H (torch.Tensor): Matrix H.
            bs (int): Batch Size
            y (toch.Tensor): Solution
            device (str): Device to run the model on ('cpu' or 'cuda').

        """
        super(SORNet, self).__init__()
        self.device = device
        self.inv_omega = nn.Parameter(torch.tensor(init_val_SORNet, device=device))
        
        A, D, L, U, _, _ = decompose_matrix(A)
        
        self.A = A.to(device)
        self.D = D.to(device)
        self.L = L.to(device)
        self.U = U.to(device)
        self.H = H.to(device)
        self.Dinv = torch.linalg.inv(D).to(device)
        self.bs = bs
        self.y = y.to(device)

    def forward(self, num_itr):
        """
        Perform forward pass of the SORNet model.

        Args:
            bs (int): Batch size.
            y (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
            list: List of intermediate results.

        """
        traj = []

        invM = torch.linalg.inv(self.inv_omega * self.D + self.L)
        s = torch.zeros(self.bs, self.H.size(1), device=self.device)
        traj.append(s)
        yMF = torch.matmul(self.y, self.H.T)
        s = torch.matmul(yMF, self.Dinv)

        for _ in range(num_itr):
            temp = torch.matmul(s, (self.inv_omega - 1) * self.D - self.U) + yMF
            s = torch.matmul(temp, invM)
            traj.append(s)

        return s, traj

# ========================================================================================================

class SOR_CHEBY_Net(nn.Module):
    """Deep unfolded SOR with Chebyshev acceleration."""

    def __init__(self, num_itr, init_val_SOR_CHEBY_Net_omega, init_val_SOR_CHEBY_Net_gamma, init_val_SOR_CHEBY_Net_alpha, A, H, bs, y, device=device):
        """
        Initialize the SOR_CHEBY_Net model.

        Args:
            num_itr (int): Number of iterations.
            init_val_SOR_CHEBY_Net_omega (float): Initial value for omega.
            init_val_SOR_CHEBY_Net_gamma (float): Initial value for gamma.
            init_val_SOR_CHEBY_Net_alpha (float): Initial value for inv_omega.
            D (torch.Tensor): Diagonal matrix D.
            L (torch.Tensor): Lower triangular matrix L.
            U (torch.Tensor): Upper triangular matrix U.
            H (torch.Tensor): Matrix H.
            bs (int): Batch Size
            y (torch.Tensor): Solution of the linear equation
            device (str): Device to run the model on ('cpu' or 'cuda').

        """
        super(SOR_CHEBY_Net, self).__init__()
        self.device = device
        self.gamma = nn.Parameter(init_val_SOR_CHEBY_Net_gamma * torch.ones(num_itr, device=device))
        self.omega = nn.Parameter(init_val_SOR_CHEBY_Net_omega * torch.ones(num_itr, device=device))
        self.inv_omega = nn.Parameter(torch.tensor(init_val_SOR_CHEBY_Net_alpha, device=device))
        
        A, D, L, U, _, _ = decompose_matrix(A)
        self.A = A
        self.D = D.to(device)
        self.L = L.to(device)
        self.U = U.to(device)
        self.H = H.to(device)
        self.Dinv = torch.linalg.inv(D).to(device)
        self.bs = bs
        self.y = y.to(device)

    def forward(self, num_itr):
        """
        Perform forward pass of the SOR_CHEBY_Net model.

        Args:
            bs (int): Batch size.
            y (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
            list: List of intermediate results.

        """
        traj = []

        invM = torch.linalg.inv(self.inv_omega * self.D + self.L)
        s = torch.zeros(self.bs, self.H.size(1), device=self.device)
        s_new = torch.zeros(self.bs, self.H.size(1), device=self.device)
        traj.append(s)
        yMF = torch.matmul(self.y, self.H.T)
        s = torch.matmul(yMF, self.Dinv)
        s_present = s
        s_old = torch.zeros_like(s_present)

        for i in range(num_itr):
            temp = torch.matmul(s, (self.inv_omega - 1) * self.D - self.U) + yMF
            s = torch.matmul(temp, invM)

            s_new = self.omega[i] * (self.gamma[i] * (s - s_present) + (s_present - s_old)) + s_old
            s_old = s
            s_present = s_new
            traj.append(s_new)

        return s_new, traj

# =====================================================================================

class AORNet(nn.Module):
    """Deep unfolded AOR with a constant step size."""

    def __init__(self, init_val_AORNet_r, init_val_AORNet_omega, A, H, bs, y, device=device):
        """
        Initialize the AORNet model.

        Args:
            init_val_AORNet_r (float): Initial value for r.
            init_val_AORNet_omega (float): Initial value for omega.
            D (torch.Tensor): Diagonal matrix D.
            L (torch.Tensor): Lower triangular matrix L.
            U (torch.Tensor): Upper triangular matrix U.
            H (torch.Tensor): Matrix H.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        super(AORNet, self).__init__()
        self.device = device
        self.r = nn.Parameter(torch.tensor(init_val_AORNet_r, device=device))
        self.omega = nn.Parameter(torch.tensor(init_val_AORNet_omega, device=device))
        
        A, D, L, U, _, _ = decompose_matrix(A)
        self.A = A.to(device)
        self.D = D.to(device)
        self.L = L.to(device)
        self.U = U.to(device)
        self.H = H.to(device)
        self.Dinv = torch.linalg.inv(D).to(device)
        self.bs = bs
        self.y = y

    def forward(self, num_itr):
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

        invM = torch.linalg.inv(self.L - self.r * self.D)
        N = (1 - self.omega) * self.D + (self.omega - self.r) * self.L + self.omega * self.U
        s = torch.zeros(self.bs, self.H.size(1), device=self.device)
        traj.append(s)
        yMF = torch.matmul(self.y, self.H.T)
        s = torch.matmul(yMF, self.Dinv)

        for _ in range(num_itr):
            s = torch.matmul(s, torch.matmul(invM, N)) + torch.matmul(yMF, invM)
            traj.append(s)

        return s, traj

# =====================================================================================

class RINet(nn.Module):
    """Deep unfolded Richardson iteration."""

    def __init__(self, init_val_RINet, A, H, bs, y, device=device):
        """
        Initialize the RINet model.

        Args:
            num_itr (int): Number of iterations.

        """
        super(RINet, self).__init__()
        self.inv_omega = nn.Parameter(torch.tensor(init_val_RINet, device=device))
        
        A, D, L, U, _, _ = decompose_matrix(A)
        self.A = A.to(device)
        self.D = D.to(device)
        self.L = L.to(device)
        self.U = U.to(device)
        self.H = H.to(device)
        self.Dinv = torch.linalg.inv(D).to(device)
        self.bs = bs
        self.y = y

    def forward(self, num_itr):
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

        s = torch.zeros(self.bs, self.A.shape[0]).to(device)
        traj.append(s)
        yMF = torch.matmul(self.y, self.H.T)
        s = torch.matmul(yMF, self.Dinv)

        for _ in range(num_itr):
            s = s + torch.mul(self.inv_omega[0], (yMF - torch.matmul(s, self.A)))
            traj.append(s)

        return s, traj