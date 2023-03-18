### Theory of the package

The package includes iterative methods for solving linear equations. However, due to the various parameters and performance of the iterative approach, it is necessary to optimize these parameters to improve the convergence rate. Here are two iterative methods, starting with the linear equation:

$\mathbf{A}\mathbf{x}=\mathbf{b}$
where decmpose for $\mathbf{A}$

 $\mathbf{A}=\mathbf{D}-\mathbf{U}-\mathbf{L}.$

 where diagonal matrix $\mathbf{D}$ , strictly upper triangular matrix $\mathbf{U}$, and  strictly lower triangular matrix $\mathbf{L}$, respectively.

to solve $\mathbf{x}$ from the equation above, it can solve in an iteratively way as accelerated over relaxation AOR as 

 $\mathbf{x}^{(n+1)}=(\mathbf{D}-\beta\mathbf{L})^{-1}((1-\alpha)\mathbf{D}+(\alpha-\beta)\mathbf{L}+\alpha \mathbf{U})  \mathbf{x}^{(n)} \\ +(\mathbf{D}-\beta\mathbf{L})^{-1}\alpha\mathbf{b}.$

for successive over-relaxation SOR as 

$\mathbf{x}^{(n+1)}=(\mathbf{D}-\omega\mathbf{L})^{-1}((1-\omega)\mathbf{D}+\omega \mathbf{U})\mathbf{x}^{(n)}+(\mathbf{D}-\omega\mathbf{L})^{-1}\mathbf{b}.$ 

 To further accelerate these methods, we propose another technique called Chebyshev acceleration method to enhance the convergence rate. The Chebyshev acceleration method can be expressed as:

 $\mathbf{y}^{(n+1)} = p_{n+1}\Big(\gamma (\mathbf{x}^{(n+1)} - \mathbf{y}^{(n)}) + (\mathbf{y}^{(n)}-\mathbf{y}^{(n-1)}) \Big)  + \mathbf{y}^{(n-1)}.$

To optimize the parameters $\alpha, \omega, \beta, p_{n+1}, \gamma$, deep learning can assist in accelerating the convergence rate of the iterative method as the spectral radius is difficult to calculate in various applications of linear equations.

One such application is "deep unfolding", which takes an iterative algorithm with a fixed number of iterations T, unravels its structure, and adds trainable parameters. These parameters are then trained using deep learning techniques such as loss functions, stochastic gradient descent, and back propagation.

The deep unfolding of the iterative algorithm proposed in this package is as follows:
<ul>
  <li>SORNet</li>
  <li>AORNet</li>
  <li>RINet</li>
  <li>ChebySORNet</li>
  <li>ChebyAORNet</li>
</ul>
For example SORNet as 

$\mathbf{x}^{(t+1)}=(\mathbf{D}-\omega_{t}\mathbf{L})^{-1}((1-\omega_{t})\mathbf{D}+\omega_{t} \mathbf{U})\mathbf{x}^{(t)}+(\mathbf{D}-\omega_{t}\mathbf{L})^{-1}\mathbf{b}.$ 

here $t$ is the number of training. The trainable internal parameters, such as $\omega_{t}$ parameter, can be optimized with standard deep learning techniques, i.e., the back propagation and stochastic gradient descent algorithms.


