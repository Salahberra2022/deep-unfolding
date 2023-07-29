## Deep unfolding of iterative method

The package includes iterative methods for solving linear equations. However, due to the various parameters and performance of the iterative approach, it is necessary to optimize these parameters to improve the convergence rate. Such proposed tool called **deep_unfolding**, which takes an iterative algorithm with a fixed number of iterations T, unravels its structure, and adds trainable parameters. These parameters are then trained using deep learning techniques such as loss functions, stochastic gradient descent, and back-propagation.
The package contain two different Iterative methods. First package called **Iterative**, which contain the conventional iterative method. The other package called **IterativeNet**, which contain the deep unfolding of iterative method.

### Installation 
```python
pip install --upgrade pip
pip install deep_unfolding
```
### Quick start

```python
from deep_unfolding import Iterative
from Iterative import main
list_iterative=['RI', 'SOR', 'GS']
main(list_iterative)
```

  ![Iterative2 (1)](https://github.com/Salahberra2022/deep_unfolding/assets/119638218/c5e53af3-445a-4607-8cec-b9ba33400f26)

  

```python
from deep_unfolding import IterativeNet
from IterativeNet import main
list_iterative=['RINet', 'RI', 'SORNet', 'SOR', 'GS']
main(list_iterative)
```

 ![Iterative4](https://github.com/Salahberra2022/deep_unfolding/assets/119638218/c53ceec4-458f-44e8-b6cb-72e559b69ffc)



  ![SORNet](https://user-images.githubusercontent.com/119638218/226128700-f03ae894-a69b-48b1-a4bf-a0a3d2820d8e.png)
  

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

# Reference
If you use this software, please cite the following reference:



# License

[GPL License](LICENSE)





