### deep unfolding 

The package includes iterative methods for solving linear equations. However, due to the various parameters and performance of the iterative approach, it is necessary to optimize these parameters to improve the convergence rate. Such proposed tool called $deep_unfolding$, which takes an iterative algorithm with a fixed number of iterations T, unravels its structure, and adds trainable parameters. These parameters are then trained using deep learning techniques such as loss functions, stochastic gradient descent, and back-propagation.

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
<figure markdown>
  ![Iterative](https://user-images.githubusercontent.com/119638218/226128243-a2709a81-ede9-44d7-97d6-9e2081c8b10b.png){ width="500" }
  <figcaption>The Iterative method </figcaption>
</figure>

```python
from deep_unfolding import IterativeNet
from IterativeNet import main
list_iterative=['RINet', 'SORNet', 'AORNet']
main(list_iterative)
```
<figure markdown>
 ![IterativeNet](https://user-images.githubusercontent.com/119638218/226128512-9b8187bb-2433-40b3-bf71-510461ab62d5.png){ width="500" }
  <figcaption>The Iterative method based on the deep unfolding network </figcaption>
</figure>

# The training process 

<figure markdown>
  ![The Iterative method SOR and SORNet](https://user-images.githubusercontent.com/119638218/226128700-f03ae894-a69b-48b1-a4bf-a0a3d2820d8e.png){ width="500" }
  <figcaption>The implementation of training process</figcaption>
</figure>