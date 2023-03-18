import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
#from mse import mse
#from fnc import itr_list, norm_list_GS, norm_list_RI, norm_list_Jacobi, norm_list_SOR, norm_list_AOR, norm_list_AOR_CHEBY, norm_list_SOR_CHEBY
#from meth import model_SorNet, model_Sor_Cheby_Net
#from conIte import gs_model, ri_model, Jacobi_model, sor_model, sor_cheby_model, aor_model, aor_cheby_model
device = torch.device('cpu') # 'cpu' or 'cuda'
## model parameters 
itr = 25 # iteration steps $T$
total_itr = itr # max. iterations
n = 300 
m = 600 
##



## parameters for evauation of generalization error 
total_itr=25 # total number of iterations (multiple number of "itr")
bs = 10 # number of samples 
##
# generate A and H
seed_ = 12
np.random.seed(seed=seed_)
H = np.random.normal(0,1.0/math.sqrt(n),(n,m)) 
A = np.dot(H,H.T)
eig = np.linalg.eig(A)
eig = eig[0] # eigenvalues

W = torch.Tensor(np.diag(eig)).to(device)
H = torch.from_numpy(H).float().to(device)


D = np.diag(np.diag(A))
U = np.triu(A, 1)
L = np.tril(A, -1)
Dinv = np.linalg.inv(D)
invM = np.linalg.inv(D +L)  
A = torch.Tensor(A).to(device)
D = torch.Tensor(D).to(device)
U = torch.Tensor(U).to(device)
L = torch.Tensor(L).to(device)
Dinv = torch.Tensor(Dinv).to(device)
invM = torch.Tensor(invM).to(device)

print("condition number, min. and max. eigenvalues of A")
print(np.max(eig)/np.min(eig),np.max(eig),np.min(eig))


    ## naive GS with a constant step size 
class GS(nn.Module):
    def __init__(self, num_itr):
        super(GS, self).__init__()
    def forward(self, num_itr, bs,y):
        traj = []
        s  = torch.zeros(bs,n).to(device)
        traj.append(s)                  
        yMF = y@H.T
        s = torch.matmul( yMF, Dinv)                                           # Generate batch initial solution vevtor

        for i in range(num_itr):
            temp = -torch.matmul(s,U)+ yMF
            s = torch.matmul(temp, invM)
            traj.append(s) 
        return s, traj

gs_model = GS(itr).to(device)
## naive RI with a constant step size 
class RI(nn.Module):
    def __init__(self, num_itr):
        super(RI, self).__init__()
    def forward(self, num_itr, bs,y):
        traj = []
        omega = torch.tensor(0.25)
        s  = torch.zeros(bs,n).to(device)
        traj.append(s)                  
        yMF = y@H.T
        s = torch.matmul( yMF, Dinv)                                           # Generate batch initial solution vevtor

        for i in range(num_itr):
            #temp = -torch.matmul(s,U)+ yMF
            s = s + torch.mul(omega,(yMF-torch.matmul(s,A)))
            traj.append(s) 
        return s, traj

ri_model = RI(itr).to(device)
## naive Jacobi with a constant step size 
class Jacobi(nn.Module):
    def __init__(self, num_itr):
        super(Jacobi, self).__init__()
    def forward(self, num_itr, bs,y):
        traj = []
        omega = torch.tensor(0.2);
        s  = torch.zeros(bs,n).to(device)
        traj.append(s)                  
        yMF = y@H.T
        s = torch.matmul( yMF, Dinv)                                           # Generate batch initial solution vevtor

        for i in range(num_itr):
            temp = torch.matmul(Dinv,(D-A))
            s = torch.matmul(s,temp) + torch.matmul(yMF,Dinv)
            traj.append(s) 
        return s, traj

Jacobi_model = Jacobi(itr).to(device)
## naive SOR with a constant step size 
class SOR(nn.Module):
    def __init__(self, num_itr):
        super(SOR, self).__init__()
    def forward(self, num_itr, bs,y):
        traj = []
        omega = torch.tensor(1.2);     inv_omega = torch.div(1,omega);
        invM_sor = torch.linalg.inv(D-torch.mul(inv_omega, L))  
        s  = torch.zeros(bs,n).to(device)
        traj.append(s)                  
        yMF = y@H.T
        s = torch.matmul( yMF, Dinv)                                           # Generate batch initial solution vevtor

        for i in range(num_itr):
            temp = torch.mul((inv_omega-1),D)+ torch.mul((inv_omega),U)
            s = torch.matmul(s, torch.matmul(invM_sor,temp))+torch.matmul(yMF, invM_sor)
            traj.append(s) 
        return s, traj

sor_model = SOR(itr).to(device)

class SOR_CHEBY(nn.Module):
    def __init__(self, num_itr):
        super(SOR_CHEBY, self).__init__()
    def forward(self, num_itr, bs,y):
        traj = []
        omega = torch.tensor(1.8);     inv_omega = torch.div(1,omega);
        invM_sor = torch.linalg.inv(D-torch.mul(inv_omega, L))  
        s  = torch.zeros(bs,n).to(device)
        s_new  = torch.zeros(bs,n).to(device)
        traj.append(s)                  
        yMF = y@H.T
        s = torch.matmul( yMF, Dinv)                                           # Generate batch initial solution vevtor
        omegaa = torch.tensor(0.8);     gamma = torch.tensor(0.8);
        s_present = s
        s_old = torch.zeros(s_present.shape).to(device)
        for i in range(num_itr):
            temp = torch.mul((inv_omega-1),D)+ torch.mul((inv_omega),U)
            s = torch.matmul(s, torch.matmul(invM_sor,temp))+torch.matmul(yMF,invM_sor)

            s_new = omegaa * (gamma * (s - s_present) + (s_present -s_old) ) + s_old;
            s_old = s ; s_present = s_new;

            traj.append(s_new) 
        return s_new, traj

sor_cheby_model = SOR_CHEBY(itr).to(device)

## naive AOR with a constant step size 
class AOR(nn.Module):
    def __init__(self, num_itr):
        super(AOR, self).__init__()
    def forward(self, num_itr, bs,y):
        traj = []
        r=torch.tensor(0.2)        # accleration parameter
        omega = torch.tensor(0.3)  #overelexation parameter  
        inv_omega = torch.div(1,omega)
        M = (D-torch.mul(r,L)) ;    invM_aor = torch.linalg.inv(M)
        N = (torch.mul((1-omega),D) + torch.mul((omega-r),L) +torch.mul(omega,U))    
        
        s  = torch.zeros(bs,n).to(device)
        traj.append(s)                  
        yMF = y@H.T
        s = torch.matmul( yMF, Dinv)                                           # Generate batch initial solution vevtor

        for i in range(num_itr):
            s = torch.matmul(s,torch.matmul(invM_aor,N)) +  torch.mul(omega, torch.matmul(yMF,invM_aor))
            traj.append(s) 
        return s, traj

aor_model = AOR(itr).to(device)

## naive AOR_CHEBY with a constant step size 
class AOR_CHEBY(nn.Module):
    def __init__(self, num_itr):
        super(AOR_CHEBY, self).__init__()
    def forward(self, num_itr, bs,y):
        traj = []
        r=torch.tensor(0.1);              omega = torch.tensor(0.1)    #overelexation parameter  
        s  = torch.zeros(bs,n).to(device)
        traj.append(s)                  
        yMF = y@H.T
        s = torch.matmul( yMF, Dinv)                                           # Generate batch initial solution vevtor

        Y0 = s ;    X0= s                                          
        M = (D-torch.mul(r,L)) ;    invM = torch.linalg.inv(M)                                         # Generate batch initial solution vevtor
        N = (torch.mul((1-omega),D) + torch.mul((omega-r),L) +torch.mul(omega,U)) 
        temp=torch.matmul(invM,N)

        rho =torch.tensor(0.1) ;    
        mu0 = torch.tensor(1) ;    mu1 =rho
        xhat1 = torch.matmul(s, temp) + omega*torch.matmul(yMF, invM)                                                                                                                                                               
        Y1=xhat1
        Y = Y1

        for i in range(num_itr):

            f= 2/(rho*mu1) ;     j=1 /mu0 ;    c=f-j ;  mu=1/c ;     a=(2*mu)/(rho*mu1)
            Y = torch.matmul((Y1*a), torch.matmul(invM,N) ) - (((mu/mu0))*Y0)  + (a * torch.matmul(yMF, invM) )
            Y0 = Y1 ;        Y1 = Y ;        mu0 =mu1 ;       mu1 = mu

            traj.append(Y) 
        return Y, traj

aor_cheby_model = AOR_CHEBY(itr).to(device)

solution = torch.normal(0.0*torch.ones(bs,n),1.0).to(device).detach()
y = solution@H.detach()

itr_list = []
## naive GS 
norm_list_GS = []
for i in range(total_itr+1):
    s_hat, _ = gs_model(i, bs,y)
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    norm_list_GS.append(err)
    itr_list.append(i)
    ## naive RI 
norm_list_RI = []
for i in range(total_itr+1):
    s_hat, _ = ri_model(i, bs,y)
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    norm_list_RI.append(err)
    #itr_list.append(i)
norm_list_Jacobi = []
for i in range(total_itr+1):
    s_hat, _ = Jacobi_model(i, bs,y)
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    norm_list_Jacobi.append(err)
    #itr_list.append(i)

## naive SOR 
norm_list_SOR = []
for i in range(total_itr+1):
    s_hat, _ = sor_model(i, bs,y)
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    norm_list_SOR.append(err)

## naive SOR_CHEBY 
norm_list_SOR_CHEBY = []
for i in range(total_itr+1):
    s_hat, _ = sor_cheby_model(i, bs,y)
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    norm_list_SOR_CHEBY.append(err)


## naive AOR 
norm_list_AOR = []
for i in range(total_itr+1):
    s_hat, _ = aor_model(i, bs,y)
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    norm_list_AOR.append(err)
norm_list_AOR_CHEBY = []
for i in range(total_itr+1):
    s_hat, _ = aor_cheby_model(i, bs,y)
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    norm_list_AOR_CHEBY.append(err)