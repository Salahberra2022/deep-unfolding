## Deep unfolding of iterative method

The package includes iterative methods for solving linear equations. However, due to the various parameters and performance of the iterative approach, it is necessary to optimize these parameters to improve the convergence rate. Such a proposed tool called **deep_unfolding**, which takes an iterative algorithm with a fixed number of iterations T, unravels its structure and adds trainable parameters. These parameters are then trained using deep learning techniques such as loss functions, stochastic gradient descent, and back-propagation.
The package contains two different Iterative methods. The first package is called **methods**, which contains the conventional iterative method. The other package is called **train_methods**, which contains the deep unfolding of the iterative method.

### Installation 
```python
pip install --upgrade pip
pip install deep-unfolding
```
### Quick start
```python
from deep_unfolding.train_methods import SORNet 

model = SORNet()
```
### The diagram of the Deep unfolded network (DUN)


  
### The Rest of package

### Reference
If you use this software, please cite the following reference:



### License

[GPL License](LICENSE)





