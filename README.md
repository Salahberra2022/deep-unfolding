[![Tests](https://github.com/Salahberra2022/deep_unfolding/actions/workflows/tests.yml/badge.svg)](https://github.com/Salahberra2022/deep_unfolding/actions/workflows/tests.yml)
[![docs](https://img.shields.io/badge/docs-click_here-blue.svg)](https://Salahberra2022.github.io/deep_unfolding/)
[![PyPI](https://img.shields.io/pypi/v/unfolding-linear)](https://pypi.org/project/unfolding-linear/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/unfolding-linear?color=blueviolet)
[![GPLv3](https://img.shields.io/badge/license-GPLv3-yellowgreen.svg)](https://www.tldrlegal.com/license/gnu-general-public-license-v3-gpl-3)

# unfolding-linear: Deep unfolding of iterative methods

The **unfolding-linear** package includes iterative methods for solving linear equations. However, due to the various parameters and performance characteristics of the iterative approach, it is necessary to optimize these parameters to improve the convergence rate. **unfolding-linear** takes an iterative algorithm with a fixed number of iterations $T$, unravels its structure, and adds trainable parameters. These parameters are then trained using deep learning techniques such as loss functions, stochastic gradient descent, and backpropagation.

The package contains two different modules containing iterative methods. The first, `methods`, includes conventional iterative methods. The second, `train_methods`, includes deep unfolding versions of the conventional methods.

## Installation

```bash
pip install --upgrade pip
pip install unfolding-linear
```

## Quick start

```python
from unfolding_linear import device, evaluate_model, generate_A_H_sol, SORNet, train_model
from torch import nn, optim

total_itr = 25  # Total number of iterations
n = 300  # Number of rows
m = 600  # Number of columns
bs = 10000  # Mini-batch size (samples)
num_batch = 500  # Number of mini-batches
lr_adam = 0.002  # Learning rate of optimizer
init_val_SORNet = 1.1  # Initial value of omega for SORNet

seed = 12

A, H, W, solution, y = generate_A_H_sol(n=n, m=m, seed=seed, bs=bs)
loss_func = nn.MSELoss()

# Model
model_SorNet = SORNet(A, H, bs, y, init_val_SORNet, device=device)

# Optimizer
opt_SORNet = optim.Adam(model_SorNet.parameters(), lr=lr_adam)

trained_model_SorNet, loss_gen_SORNet = train_model(model_SorNet, opt_SORNet, loss_func, solution, total_itr, num_batch)

norm_list_SORNet = evaluate_model(trained_model_SorNet, solution, n, bs, total_itr, device=device)
```

## Package contents

This package implements various iterative techniques for approximating the solutions of linear problems of the type $Ax = b$. The conventional methods implemented in the `methods` module are:

- **GS**: Gauss-Seidel (GS) algorithm
- **RI**: Richardson iteration algorithm
- **Jacobi**: Jacobi iteration (RI) algorithm
- **SOR**: Successive Over-Relaxation (SOR) algorithm
- **SOR_CHEBY**: Successive Over-Relaxation (SOR) with Chebyshev acceleration algorithm
- **AOR**: Accelerated Over-Relaxation (AOR) algorithm
- **AOR_CHEBY**: Accelerated Over-Relaxation (AOR) with Chebyshev acceleration algorithm

This package also implements several models based on **Deep Unfolding Learning**, enabling optimization of the parameters of some of the preceding algorithms to obtain an optimal approximation. The models implemented in the module `train_methods` are:

- **SORNet**: Optimization via Deep Unfolding Learning of the Successive Over-Relaxation (SOR) algorithm
- **SOR_CHEBY_Net**: Optimization via Deep Unfolding Learning of the Successive Over-Relaxation (SOR) with Chebyshev acceleration algorithm
- **AORNet**: Optimization via Deep Unfolding Learning of the Accelerated Over-Relaxation (AOR) algorithm
- **RINet**: Optimization via Deep Unfolding Learning of the Richardson iteration (RI) algorithm

## Reference

If you use this software, please cite the following reference: _available soon_

## License

[GPLv3 License](LICENSE)