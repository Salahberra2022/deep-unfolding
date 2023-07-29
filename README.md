## Deep unfolding of iterative method

<<<<<<< HEAD
The package includes iterative methods for solving linear equations. However, due to the various parameters and performance of the iterative approach, it is necessary to optimize these parameters to improve the convergence rate. Such proposed tool called **deep_unfolding**, which takes an iterative algorithm with a fixed number of iterations T, unravels its structure, and adds trainable parameters. These parameters are then trained using deep learning techniques such as loss functions, stochastic gradient descent, and back-propagation.
The package contain two different Iterative methods. First package called **Iterative**, which contain the conventional iterative method. The other package called **IterativeNet**, which contain the deep unfolding of iterative method.
=======
The package includes iterative methods for solving linear equations. However, due to the various parameters and performance of the iterative approach, it is necessary to optimize these parameters to improve the convergence rate. Such proposed tool called $deep_unfolding$, which takes an iterative algorithm with a fixed number of iterations T, unravels its structure, and adds trainable parameters. These parameters are then trained using deep learning techniques such as loss functions, stochastic gradient descent, and back-propagation.
The package contain two different Iterative methods. First package called $Iterative$, which contain the conventional iterative method. The other package called $IterativeNet$, which contain the deep unfolding of iterative method.
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
list_iterative=['RI', 'SOR', 'AOR']
result(list_iterative)
```

  ![Iterative](https://user-images.githubusercontent.com/119638218/226128243-a2709a81-ede9-44d7-97d6-9e2081c8b10b.png)
  

```python
from deep_unfolding import IterativeNet
from IterativeNet import main
list_iterative=['RINet', 'SORNet', 'AORNet']
main(list_iterative)
```

 ![IterativeNet](https://user-images.githubusercontent.com/119638218/226128512-9b8187bb-2433-40b3-bf71-510461ab62d5.png)



  ![SORNet](https://user-images.githubusercontent.com/119638218/226128700-f03ae894-a69b-48b1-a4bf-a0a3d2820d8e.png){
  

# The Rest of package

The package includes several conventional iterative methods for solving the linear equation (**Ax=b**), such as 
<h4> The iterative methods</h4>
<ul>
  <li>Accelerated Over-relaxation (AOR)</li>
  <li>Successive Over-relaxation (SOR)</li>
  <li>Jacobi (JA)</li>
  <li>gauss seidel (GS)</li>
  <li>Richardson iteration (RI)</li>
</ul>


The package includes the following deep unfolded iterative methods:
<h4> deep unfolded iterative methods </h4>
<ul>
  <li>AORNet</li>
  <li>SORNet</li>
  <li>ChebySORNet</li>
  <li>ChebyAORNet</li>
  <li>RINet</li>
</ul>

<<<<<<< HEAD
# Reference
If you use this software, please cite the following reference:



# License

![GPL  License](https://github.com/Salahberra2022/deep_unfolding/blob/main/LICENSE)

=======
>>>>>>> 49129962c65b00396d7c679ff04f60b715fb28ac

