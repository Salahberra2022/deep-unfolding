# Description the code 

## Initialization value
```python
## model parameters 
itr = 25 
'''
    iteration steps $T$
'''
total_itr = itr 
'''
    max. iterations
    '''
n = 300 
m = 600 
'''
    The size of matrix generation
    '''
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
'''
    the random number to gneration the whole data
    '''
H = np.random.normal(0,1.0/math.sqrt(n),(n,m)) 
A = np.dot(H,H.T)
eig = np.linalg.eig(A)
eig = eig[0] # eigenvalues

W = torch.Tensor(np.diag(eig)).to(device)
H = torch.from_numpy(H).float().to(device)


D = np.diag(np.diag(A))
'''
    the digonal matrix
    '''
U = np.triu(A, 1)
'''
    the model of SOR method
    '''
L = np.tril(A, -1)
'''
    the model of SOR method
    '''
Dinv = np.linalg.inv(D)
'''
    the model of SOR method
    '''
invM = np.linalg.inv(D +L)  
'''
    the model of SOR method
    '''
```
## The conventional SOR with optimum in value
```py
   class SOR(nn.Module):
    '''
    the model of SOR method
    '''
    def __init__(self, num_itr):
        super(SOR, self).__init__()
    def forward(self, num_itr, bs,y):
        traj = []
        omega = torch.tensor(1.2);     inv_omega = torch.div(1,omega)
        '''
        the paramter of SOR $\omega=1.2$
        '''
        invM_sor = torch.linalg.inv(D-torch.mul(inv_omega, L))  
        s  = torch.zeros(bs,n).to(device)
        traj.append(s)                  
        yMF = y@H.T
        s = torch.matmul( yMF, Dinv)   
        '''
        Generate solution vevtor
        '''                                      
        for i in range(num_itr):
            temp = torch.mul((inv_omega-1),D)+ torch.mul((inv_omega),U)
            s = torch.matmul(s, torch.matmul(invM_sor,temp))+torch.matmul(yMF, invM_sor)
            '''
            The s final solotion
             ''' 
            traj.append(s) 
        return s, traj

sor_model = SOR(itr).to(device)
```
## The deep unfolding of  SOR , as SORNet

```python
## Deep unfolded SOR with a constant step size 
class SORNet(nn.Module):
    def __init__(self, num_itr):
        super(SORNet, self).__init__()
        self.inv_omega = nn.Parameter(init_val_SORNet*torch.ones(1))
        '''
        the parameter of SORNet $\omega_t$ input to the Deep unfolded to the optimal value
        '''
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
```

## The training process of SORNet
```python
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
```
## The Loss function 

```python
generation: 1  batch: 0 	 MSE loss: 0.21193785965442657
generation: 1  batch: 200 	 MSE loss: 0.16124731302261353
generation: 1  batch: 400 	 MSE loss: 0.15413512289524078
generation: 2  batch: 0 	 MSE loss: 0.10527851432561874
generation: 2  batch: 200 	 MSE loss: 0.09313789755105972
generation: 2  batch: 400 	 MSE loss: 0.07886634021997452
generation: 3  batch: 0 	 MSE loss: 0.043511394411325455
generation: 3  batch: 200 	 MSE loss: 0.03456632420420647
generation: 3  batch: 400 	 MSE loss: 0.032005008310079575
generation: 4  batch: 0 	 MSE loss: 0.012704577296972275
generation: 4  batch: 200 	 MSE loss: 0.01162696722894907
generation: 4  batch: 400 	 MSE loss: 0.013365212827920914
generation: 5  batch: 0 	 MSE loss: 0.0043519423343241215
generation: 5  batch: 200 	 MSE loss: 0.004235091619193554
generation: 5  batch: 400 	 MSE loss: 0.004391568712890148
generation: 6  batch: 0 	 MSE loss: 0.0018134030979126692
generation: 6  batch: 200 	 MSE loss: 0.0015237266197800636
generation: 6  batch: 400 	 MSE loss: 0.001996097154915333
generation: 7  batch: 0 	 MSE loss: 0.0007896940805949271
generation: 7  batch: 200 	 MSE loss: 0.0007846651133149862
generation: 7  batch: 400 	 MSE loss: 0.000536845822352916
generation: 8  batch: 0 	 MSE loss: 0.00029308913508430123
generation: 8  batch: 200 	 MSE loss: 0.00028003635816276073
generation: 8  batch: 400 	 MSE loss: 0.0002928390458691865
generation: 9  batch: 0 	 MSE loss: 0.00010200320684816688
generation: 9  batch: 200 	 MSE loss: 0.00010443529754411429
generation: 9  batch: 400 	 MSE loss: 9.555269207339734e-05
generation: 10  batch: 0 	 MSE loss: 4.350042945588939e-05
generation: 10  batch: 200 	 MSE loss: 4.744587931782007e-05
generation: 10  batch: 400 	 MSE loss: 3.300118987681344e-05
generation: 11  batch: 0 	 MSE loss: 1.364766103506554e-05
generation: 11  batch: 200 	 MSE loss: 1.927423545566853e-05
generation: 11  batch: 400 	 MSE loss: 1.7909926100401208e-05
generation: 12  batch: 0 	 MSE loss: 5.9777075875899754e-06
generation: 12  batch: 200 	 MSE loss: 6.701727215840947e-06
generation: 12  batch: 400 	 MSE loss: 7.5510765782382805e-06
generation: 13  batch: 0 	 MSE loss: 2.803974439302692e-06
generation: 13  batch: 200 	 MSE loss: 2.259379471070133e-06
generation: 13  batch: 400 	 MSE loss: 2.2352169253281318e-06
generation: 14  batch: 0 	 MSE loss: 1.5446163388332934e-06
generation: 14  batch: 200 	 MSE loss: 1.128724989030161e-06
generation: 14  batch: 400 	 MSE loss: 8.900381089915754e-07
generation: 15  batch: 0 	 MSE loss: 4.1349861135131505e-07
generation: 15  batch: 200 	 MSE loss: 3.26549326246095e-07
generation: 15  batch: 400 	 MSE loss: 3.319844381621806e-07
generation: 16  batch: 0 	 MSE loss: 1.6063326313542348e-07
generation: 16  batch: 200 	 MSE loss: 2.1569556452050165e-07
generation: 16  batch: 400 	 MSE loss: 2.5400643721695815e-07
generation: 17  batch: 0 	 MSE loss: 7.991993555833687e-08
generation: 17  batch: 200 	 MSE loss: 8.242315630013763e-08
generation: 17  batch: 400 	 MSE loss: 9.084590857355579e-08
generation: 18  batch: 0 	 MSE loss: 4.878432946497924e-08
generation: 18  batch: 200 	 MSE loss: 2.3802479987011793e-08
generation: 18  batch: 400 	 MSE loss: 4.135335984756239e-08
generation: 19  batch: 0 	 MSE loss: 8.965354680867677e-09
generation: 19  batch: 200 	 MSE loss: 1.9243438842408978e-08
generation: 19  batch: 400 	 MSE loss: 1.1302278224434303e-08
generation: 20  batch: 0 	 MSE loss: 5.0089292713551e-09
generation: 20  batch: 200 	 MSE loss: 3.921176272569937e-09
generation: 20  batch: 400 	 MSE loss: 4.728981650714559e-09
generation: 21  batch: 0 	 MSE loss: 3.098753031949286e-09
generation: 21  batch: 200 	 MSE loss: 2.4030586409651278e-09
generation: 21  batch: 400 	 MSE loss: 2.713620883554313e-09
generation: 22  batch: 0 	 MSE loss: 9.762120001255425e-10
generation: 22  batch: 200 	 MSE loss: 9.139658474488499e-10
generation: 22  batch: 400 	 MSE loss: 6.508619754264089e-10
generation: 23  batch: 0 	 MSE loss: 2.9331595485793116e-10
generation: 23  batch: 200 	 MSE loss: 5.554421922404629e-10
generation: 23  batch: 400 	 MSE loss: 2.654713393557273e-10
generation: 24  batch: 0 	 MSE loss: 1.9251777949591542e-10
generation: 24  batch: 200 	 MSE loss: 1.535244015249404e-10
generation: 24  batch: 400 	 MSE loss: 1.4029258310621628e-10
generation: 25  batch: 0 	 MSE loss: 5.961817844957196e-11
generation: 25  batch: 200 	 MSE loss: 7.959573278260024e-11
generation: 25  batch: 400 	 MSE loss: 7.604925023052544e-11
```

## The calculation error 
```python
norm_list_SOR = []
for i in range(total_itr+1):
    s_hat, _ = sor_model(i, bs,y)
    '''
     s_hat the output value from the model of SOR method
    '''
    err = (torch.norm(solution.to(device) - s_hat.to(device))**2).item()/(n*bs)
    '''
    err the norm error from the exact one and the output of SOR method
    '''
    norm_list_SOR.append(err)
```
## The result  
![1 (1)](https://github.com/Salahberra2022/deep_unfolding/assets/119638218/99363b50-9853-49eb-8fc8-6025c7ac4e89)


 
![Iterative2](https://user-images.githubusercontent.com/119638218/ebdbfe18-eb64-46ee-9eeb-67664832f3a5)


 
 

