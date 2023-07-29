### deep unfolding 

The package includes iterative methods for solving linear equations. However, due to the various parameters and performance of the iterative approach, it is necessary to optimize these parameters to improve the convergence rate. Such proposed tool called **deep_unfolding**, which takes an iterative algorithm with a fixed number of iterations T, unravels its structure, and adds trainable parameters. These parameters are then trained using deep learning techniques such as loss functions, stochastic gradient descent, and back-propagation.

### Installation 
```python
pip install --upgrade pip
pip install deep_unfolding
```
### Quick start

```python
from deep_unfolding import Iterative
from iterative import result
list_iterative=['RI', 'SOR', 'GS']
result(list_iterative)
```
![Iterative2 (1)](https://github.com/Salahberra2022/deep_unfolding_2023/assets/119638218/184e9342-669c-42e1-9ad5-5d29c6c1227a)


```python
from deep_unfolding import IterativeNet
from IterativeNet import main
list_iterative=['RINet','RI', 'SORNet','SOR', 'GS']
main(list_iterative)
```
![Iterative4](https://github.com/Salahberra2022/deep_unfolding_2023/assets/119638218/8b2efa0d-b8b3-4f5b-a0a4-96181b91932e)


# The training process 

![SORNet](https://user-images.githubusercontent.com/119638218/226128700-f03ae894-a69b-48b1-a4bf-a0a3d2820d8e.png)
