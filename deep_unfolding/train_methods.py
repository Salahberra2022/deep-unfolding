import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

device = torch.device('cpu') # 'cpu' or 'cuda'
## model parameters 
itr = 25 # iteration steps $T$
total_itr = itr # max. iterations
n = 300 
m = 600 
##

## training parameters
bs = 200 # mini batch size
num_batch = 500 # number of mini batches
lr_adam = 0.002 # learning rate of optimizer
init_val_SORNet = 1.1 # initial values of $\omega$
init_val_SOR_CHEBY_Net_omega = 0.6 # initial values of $\omega$
init_val_SOR_CHEBY_Net_gamma = 0.8 # initial values of $\gamma$
init_val_SOR_CHEBY_Net_alpha = 0.9 # initial values of $\alpha$
init_val_AORNet_r = 0.9 # initial values of $\r$
init_val_AORNet_omega = 1.5 # initial values of $\omega$
init_val_RINet = 0.1 # initial values of $\omega$
##

## parameters for evauation of generalization error 
total_itr=25 # total number of iterations (multiple number of "itr")
bs = 10000 # number of samples 
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



## Deep unfolded SOR with a constant step size 
class SORNet(nn.Module):
    def __init__(self, num_itr):
        super(SORNet, self).__init__()
        self.inv_omega = nn.Parameter(init_val_SORNet*torch.ones(1))
    def forward(self, num_itr, bs,y):
        traj = []
        
        invM = torch.linalg.inv(torch.mul(self.inv_omega[0], D) +L) 
        #invM_sor = torch.linalg.inv(D-torch.mul(self.inv_omega[0], L)) 
        s  = torch.zeros(bs,n).to(device)
        traj.append(s)                  
        yMF = y@H.T
        s = torch.matmul( yMF, Dinv)                                           # Generate batch initial solution vevtor

        for i in range(num_itr):
            #temp = torch.mul((self.inv_omega[0]-1),D)+ torch.mul((self.inv_omega[0]),U)
            
            #s = torch.matmul(s, torch.matmul(invM_sor,temp))+torch.matmul(yMF, invM_sor)
            temp = torch.matmul(s, torch.mul((self.inv_omega[0]-1),D)-U)+ yMF
            s = torch.matmul(temp,invM) 
            traj.append(s) 
        return s, traj

model_SorNet = SORNet(itr).to(device)
loss_func = nn.MSELoss()
opt1   = optim.Adam(model_SorNet.parameters(), lr=lr_adam)


class SOR_CHEBY_Net(nn.Module):
    def __init__(self, num_itr):
        super(SOR_CHEBY_Net, self).__init__()
        self.gamma = nn.Parameter(init_val_SOR_CHEBY_Net_omega*torch.ones(num_itr))
        self.omega = nn.Parameter(init_val_SOR_CHEBY_Net_gamma*torch.ones(num_itr))
        self.inv_omega = nn.Parameter(init_val_SOR_CHEBY_Net_alpha*torch.ones(1))
        
    def forward(self, num_itr, bs,y):
        traj = []
        
        invM = torch.linalg.inv(torch.mul(self.inv_omega[0], D) +L) 
        s  = torch.zeros(bs,n).to(device)
        s_new  = torch.zeros(bs,n).to(device)
        traj.append(s)                  
        yMF = y@H.T
        s = torch.matmul( yMF, Dinv)  
        s_present = s
        s_old = torch.zeros(s_present.shape).to(device)
        for i in range(num_itr):
            temp = torch.matmul(s, torch.mul((self.inv_omega[0]-1),D)-U)+ yMF
            s = torch.matmul(temp,invM)

            s_new = self.omega[i] * (self.gamma[i] * (s - s_present) + (s_present -s_old) ) + s_old;
            s_old = s ; s_present = s_new

            traj.append(s_new) 
        return s_new, traj

model_Sor_Cheby_Net = SOR_CHEBY_Net(itr).to(device)
loss_func = nn.MSELoss()
opt2   = optim.Adam(model_Sor_Cheby_Net.parameters(), lr=lr_adam)

## naive AORNet with a constant step size 
class AORNet(nn.Module):
    def __init__(self, num_itr):
        super(AORNet, self).__init__()
        self.r = nn.Parameter(init_val_AORNet_r*torch.ones(1))
        self.omega = nn.Parameter(init_val_AORNet_omega*torch.ones(1))
    def forward(self, num_itr, bs,y):
        traj = []
        
        #M = (D-torch.mul(self.r[0],L)) ;    invM = torch.linalg.inv(M)
        #M = (D-torch.mul(self.r[0],L)) ;    #invM = torch.linalg.inv(M)
        invM = torch.linalg.inv(L-torch.mul(self.r[0], D))
        N = (torch.mul((1-self.omega[0]),D) + torch.mul((self.omega[0]-self.r[0]),L) +torch.mul(self.omega[0],U))
        #N = torch.mul((1-self.omega[0]),D) + torch.mul((self.omega[0]-self.r[0] ),L) + torch.mul(self.omega[0],U)
        #invM = torch.linalg.inv(torch.mul(self.r[0], D) -L) 
            
        
        s  = torch.zeros(bs,n).to(device)
        traj.append(s)                  
        yMF = y@H.T
        s = torch.matmul( yMF, Dinv)                                           # Generate batch initial solution vevtor

        for i in range(num_itr):
            #temp = torch.mul((self.omega[0]-1),D)+U+torch.mul((self.omega[0]-self.r[0]),L)
            #s = torch.matmul(s,torch.matmul(temp,invM))+torch.matmul(yMF,invM)
            s = torch.matmul(s,torch.matmul(invM,N)) + torch.matmul(yMF,invM)
            #s = torch.matmul(s,torch.matmul(invM_aor,N)) +  torch.mul(self.omega[0], torch.matmul(yMF,invM_aor))
            #traj.append(s) 
        return s, traj

model_AorNet = AORNet(itr).to(device)
loss_func = nn.MSELoss()
opt3   = optim.Adam(model_AorNet.parameters(), lr=lr_adam)

## naive RI with a constant step size 
class RINet(nn.Module):
    def __init__(self, num_itr):
        super(RINet, self).__init__()
        self.inv_omega = nn.Parameter(init_val_RINet*torch.ones(1))
    def forward(self, num_itr, bs,y):
        traj = []
        #omega = torch.tensor(0.25);
        s  = torch.zeros(bs,n).to(device)
        traj.append(s)                  
        yMF = y@H.T
        s = torch.matmul( yMF, Dinv)                                           # Generate batch initial solution vevtor

        for i in range(num_itr):
            #temp = -torch.matmul(s,U)+ yMF
            s = s + torch.mul(self.inv_omega[0],(yMF-torch.matmul(s,A)))
            traj.append(s) 
        return s, traj

rinet_model = RINet(itr).to(device)
loss_func = nn.MSELoss()
opt4   = optim.Adam(rinet_model.parameters(), lr=lr_adam)

## training process of SORNet
# it takes about several minutes on Google Colaboratory

loss_gen=[]
for gen in (range(itr)): # incremental training
    for i in range(num_batch):
        opt1.zero_grad()
        solution = torch.normal(0.0*torch.ones(bs,n),1.0).to(device)
        y = solution @ H
        x_hat,_ = model_SorNet(gen + 1, bs,y)
        loss  = loss_func(x_hat, solution)
        loss.backward()
        opt1.step()
        if i % 200 == 0:
            print("generation:",gen+1, " batch:",i, "\t MSE loss:",loss.item() )
    loss_gen.append(loss.item())
## training process of SOR_CHEBY_Net
# it takes about several minutes on Google Colaboratory

loss_gen=[]
for gen in (range(itr)): # incremental training
    for i in range(num_batch):
        opt2.zero_grad()
        solution = torch.normal(0.0*torch.ones(bs,n),1.0).to(device)
        y = solution @ H
        x_hat,_ = model_Sor_Cheby_Net(gen + 1, bs,y)
        loss  = loss_func(x_hat, solution)
        loss.backward()
        opt2.step()
        if i % 200 == 0:
            print("generation:",gen+1, " batch:",i, "\t MSE loss:",loss.item() )
    loss_gen.append(loss.item())


loss_gen=[]
for gen in (range(itr)): # incremental training
    for i in range(num_batch):
        opt3.zero_grad()
        solution = torch.normal(0.0*torch.ones(bs,n),1.0).to(device)
        y = solution @ H
        x_hat,_ = model_AorNet(gen + 1, bs,y)
        loss  = loss_func(x_hat, solution)
        loss.backward()
        opt3.step()
        if i % 200 == 0:
            print("generation:",gen+1, " batch:",i, "\t MSE loss:",loss.item() )
    loss_gen.append(loss.item())


## training process of RINet
# it takes about several minutes on Google Colaboratory

loss_gen=[]
for gen in (range(itr)): # incremental training
    for i in range(num_batch):
        opt4.zero_grad()
        solution = torch.normal(0.0*torch.ones(bs,n),1.0).to(device)
        y = solution @ H
        x_hat,_ = rinet_model(gen + 1, bs,y)
        loss  = loss_func(x_hat, solution)
        loss.backward()
        opt4.step()
        if i % 200 == 0:
            print("generation:",gen+1, " batch:",i, "\t MSE loss:",loss.item() )
    loss_gen.append(loss.item())

itr_list = []
itr_list = []
#-----------------
## naive_SORNet
norm_list_SorNet= []
for i in range(total_itr+1):
    s_hat, _ = model_SorNet(i, bs,y)
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    norm_list_SorNet.append(err)
    itr_list.append(i)

## naive_Sor_Cheby_Ne
norm_list_Sor_Cheby_Net = []
for i in range(total_itr+1):
    s_hat, _ = model_Sor_Cheby_Net(i, bs,y)
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    norm_list_Sor_Cheby_Net.append(err)
    ## naive_AORNet
norm_list_AorNet = []
for i in range(total_itr+1):
    s_hat, _ = model_AorNet(i, bs,y)
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    norm_list_AorNet.append(err)
## naive_RINet
norm_list_rinet = []
for i in range(total_itr+1):
    s_hat, _ = rinet_model(i, bs,y)
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    norm_list_rinet.append(err)