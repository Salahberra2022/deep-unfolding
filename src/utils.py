# Script to englobe all constants used in the different scripts
import numpy as np
import math
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU, if not CPU
print(f"Code run on : {device}")

def generate_A_H_sol(n:int=300,m:int=600, seed:int=12, bs:int=10, device:torch.device=device) : 
    """Generate A and H

    Args:
        n (int, optional): Number of rows. Defaults to 300.
        m (int, optional): Number of columns. Defaults to 600.
        seed (int, optional): Seed for random. Defaults to 12.

    Returns:
        A, H, W: Matrix A (square matrix), matrix H (random matrix), Matrix with diagonal : eigen values of A
    """
    np.random.seed(seed=seed)
    H = np.random.normal(0, 1.0 / math.sqrt(n), (n, m))
    A = np.dot(H, H.T)
    eig = np.linalg.eig(A)[0] # Eigenvalues
    
    W = torch.Tensor(np.diag(eig)).to(device)  # Define the appropriate 'device'
    H = torch.from_numpy(H).float().to(device)  # Define the appropriate 'device'
    
    print("Condition number, min. and max. eigenvalues of A:")
    print(np.max(eig) / np.min(eig), np.max(eig), np.min(eig))
    
    solution = torch.normal(torch.zeros(bs,n),1.0).to(device).detach()
    y = solution@H.detach()
    
    return A,H, W, solution, y

def decompose_matrix(A:np.array) : 
    # Decmposed matrix calculations
    D = np.diag(np.diag(A)) # diagonal matrix
    L = np.tril(A, -1) #  lower triangular matrix
    U = np.triu(A, 1) # upper triangular matrix
    Dinv = np.linalg.inv(D) # inversion diagonal matrix
    invM = np.linalg.inv(D + L) #  inversion matrix M
    # Convert to Torch tensors and move to device
    A = torch.Tensor(A).to(device) 
    D = torch.Tensor(D).to(device) 
    L = torch.Tensor(L).to(device) 
    U = torch.Tensor(U).to(device) 
    Dinv = torch.Tensor(Dinv).to(device) 
    Minv = torch.Tensor(invM).to(device) 
    
    return A, D, L, U, Dinv, Minv