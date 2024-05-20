## Deep unfolding of iterative method
[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://pypi.org/project/unfolding-linear/) [![PyPI](https://badge.fury.io/py/unfolding-linear.svg)](https://pypi.org/project/unfolding-linear/)
The package includes iterative methods for solving linear equations. However, due to the various parameters and performance of the iterative approach, it is necessary to optimize these parameters to improve the convergence rate. Such a proposed tool called **unfolding_linear**, which takes an iterative algorithm with a fixed number of iterations T, unravels its structure and adds trainable parameters. These parameters are then trained using deep learning techniques such as loss functions, stochastic gradient descent, and back-propagation.
The package contains two different Iterative methods. The first package is called **methods**, which contains the conventional iterative method. The other package is called **train_methods**, which contains the deep unfolding of the iterative method.



### Installation 
```python
pip install --upgrade pip
pip install unfolding-linear
```
### Quick start
```python
from unfolding_linear.train_methods import SORNet 
from unfolding_linear.utils import device, generate_A_H_sol

total_itr = 25  # Total number of iterations (multiple of "itr")
n = 300  # Number of rows # ? suppose to be a variable ?
m = 600  # Number of columns # ? suppose to be a variable ?
bs = 10000  # Mini-batch size (samples)
num_batch = 500  # Number of mini-batches
lr_adam = 0.002  # Learning rate of optimizer
init_val_SORNet = 1.1  # Initial value of omega for SORNet

seed = 12

A, H, W, solution, y = generate_A_H_sol(n=n, m=m, seed=seed, bs=bs)
loss_func = nn.MSELoss()

# Model
model_SorNet = SORNet(init_val_SORNet, A, H, bs, y, device=device)
# Optimizer
opt_SORNet = optim.Adam(model_SorNet.parameters(), lr=lr_adam)

trained_model_SorNet, loss_gen_SORNet = train_model(model_SorNet, opt_SORNet, loss_func, total_itr, solution, num_batch)

norm_list_SORNet = evaluate_model(trained_model_SorNet, solution, n, bs, total_itr, device=device)
```
### Package contents
This package implements various iterative techniques for approximating the solutions of linear problems of the type $Ax = b$. The conventional methods implemented in the ``unfolding_linear.methods`` sub-module are : 
- Gauss-Seidel (GS) algorithm
- Richardson iteration algorithm
- Jacobi iteration (RI) algorithm
- Successive Over-Relaxation (SOR) algorithm
- Successive Over-Relaxation (SOR) with Chebyshev acceleration algorithm
- Accelerated Over-Relaxation (AOR) algorithm
- Accelerated Over-Relaxation (AOR) with Chebyshev acceleration algorithm

This package also implements several models based on **Deep Unfolding Learning**, enabling optimization of the parameters of some of the preceding algorithms to obtain an optimal approximation. The models implemented in the sub-module ``unfolding_linear.train_methods`` are : 
- **SORNet**: Optimization via Deep Unfolding Learning of the Successive Over-Relaxation (SOR) algorithm
- **SOR_CHEBY_Net**: Optimization via Deep Unfolding Learning of the Successive Over-Relaxation (SOR) with Chebyshev acceleration algorithm
- AORNet**: Optimization via Deep Unfolding Learning of the Accelerated Over-Relaxation (AOR) algorithm
- RINet**: Optimization via Deep Unfolding Learning of the Richardson iteration (RI) algorithm

### Reference
If you use this software, please cite the following reference:

### License

[GPLv3 License](LICENSE)