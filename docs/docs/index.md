### deep unfolding 

<<<<<<< HEAD
The package includes iterative methods for solving linear equations. However, due to the various parameters and performance of the iterative approach, it is necessary to optimize these parameters to improve the convergence rate. Such proposed tool called **deep_unfolding**, which takes an iterative algorithm with a fixed number of iterations T, unravels its structure, and adds trainable parameters. These parameters are then trained using deep learning techniques such as loss functions, stochastic gradient descent, and back-propagation.
=======
The package includes iterative methods for solving linear equations. However, due to the various parameters and performance of the iterative approach, it is necessary to optimize these parameters to improve the convergence rate. Such proposed tool called $deep_unfolding$, which takes an iterative algorithm with a fixed number of iterations T, unravels its structure, and adds trainable parameters. These parameters are then trained using deep learning techniques such as loss functions, stochastic gradient descent, and back-propagation.
>>>>>>> 49129962c65b00396d7c679ff04f60b715fb28ac

### Installation 
```python
pip install --upgrade pip
pip install deep_unfolding
```
### Quick start

```python
from deep_unfolding import Iterative
from iterative import result
<<<<<<< HEAD
list_iterative=['RI', 'SOR', 'GS']
result(list_iterative)
```
![Iterative2 (1)](https://github.com/Salahberra2022/deep_unfolding_2023/assets/119638218/184e9342-669c-42e1-9ad5-5d29c6c1227a)

=======
list_iterative=['RI', 'SOR', 'AOR']
result(list_iterative)
```
![Iterative](https://user-images.githubusercontent.com/119638218/226128243-a2709a81-ede9-44d7-97d6-9e2081c8b10b.png)
>>>>>>> 49129962c65b00396d7c679ff04f60b715fb28ac

```python
from deep_unfolding import IterativeNet
from IterativeNet import main
<<<<<<< HEAD
list_iterative=['RINet','RI', 'SORNet','SOR', 'GS']
main(list_iterative)
```
![Iterative4](https://github.com/Salahberra2022/deep_unfolding_2023/assets/119638218/8b2efa0d-b8b3-4f5b-a0a4-96181b91932e)

=======
list_iterative=['RINet', 'SORNet', 'AORNet']
main(list_iterative)
```
![IterativeNet](https://user-images.githubusercontent.com/119638218/226128512-9b8187bb-2433-40b3-bf71-510461ab62d5.png)
>>>>>>> 49129962c65b00396d7c679ff04f60b715fb28ac

# The training process 

![SORNet](https://user-images.githubusercontent.com/119638218/226128700-f03ae894-a69b-48b1-a4bf-a0a3d2820d8e.png)
