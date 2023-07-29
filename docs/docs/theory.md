# Theory

In this section, we will describe the mathematical theory behind iterative methods and their parameters for solving linear systems of equations.

# Overview

Iterative methods have been proposed for solving linear equations. However, due to the numerous parameters and varying performance of these methods, it is necessary to optimize them to improve their convergence rate. Taking inspiration from deep learning tools, the deep unfolding technique has been used to jointly optimize the iterative method's parameters, resulting in accelerated convergence rates.

The linear equations can expressed as 

$\mathbf{A}\mathbf{x}=\mathbf{b}$ 

where decmpose for $\mathbf{A}$

$\mathbf{A}=\mathbf{D}-\mathbf{U}-\mathbf{L}.$

| matrix      | Description                          |
| ----------- | ------------------------------------ |
| $\mathbf{D}$| diagonal matrix  |
| $\mathbf{U}$| upper triangular matrix |
| $\mathbf{L}$| lower triangular matrix |

to solve $\mathbf{x}$ from the equation above, it can solve in an iteratively way.
The accelerated over-relaxation AOR: 

 $\mathbf{x}^{(n+1)}=(\mathbf{D}-\beta\mathbf{L})^{-1}((1-\alpha)\mathbf{D}+(\alpha-\beta)\mathbf{L}+\alpha \mathbf{U})  \mathbf{x}^{(n)} \\ +(\mathbf{D}-\beta\mathbf{L})^{-1}\alpha\mathbf{b}.$

The successive over-relaxation SOR: 

$\mathbf{x}^{(n+1)}=(\mathbf{D}-\omega\mathbf{L})^{-1}((1-\omega)\mathbf{D}+\omega \mathbf{U})\mathbf{x}^{(n)}+(\mathbf{D}-\omega\mathbf{L})^{-1}\mathbf{b}.$ 

The Gauss–Seidal (GS): 

$\mathbf{x}^{(n+1)}=(\mathbf{D}-\mathbf{L})^{-1}\mathbf{U}\mathbf{x}^{(n)}+\mathbf{U}\mathbf{b}.$

The Jacobi (JA): 

$\mathbf{x}^{(n+1)}=\mathbf{D}^{-1}(\mathbf{D}-\mathbf{A})\mathbf{x}^{n}+\mathbf{D}^{-1}\mathbf{b}.$

The Richardson iteration (RI):

$\mathbf{x}^{(n+1)}=(\mathbf{I}-\omega \mathbf{A})\mathbf{x}^{(n)}+\omega\mathbf{b}.$

 To further accelerate these methods, we propose another technique called Chebyshev acceleration method to enhance the convergence rate. The Chebyshev acceleration method can be expressed as:

 $\mathbf{y}^{(n+1)} = p_{n+1}\Big(\gamma (\mathbf{x}^{(n+1)} - \mathbf{y}^{(n)}) + (\mathbf{y}^{(n)}-\mathbf{y}^{(n-1)}) \Big)  + \mathbf{y}^{(n-1)}.$

| Parameter      | Description                          |
| ----------- | ------------------------------------ |
| $\alpha$| the relaxation parameter of AOR  |
| $\beta$| acceleration parameter |
| $\omega$| the relaxation parameter of SOR |
| $p_{n+1}$| under-relaxation |
| $\gamma$| under-extrapolation |

# Mathematical optimization of parameters

The iterative method can be rewrite as different form as 

$\mathbf{x}^{(n+1)}=\mathbf{x}^{(n)}\mathbf{G}+\mathbf{c}.$

| math vale      | Description                          |
| ----------- | ------------------------------------ |
| $\rho$| the spectral radius |
| $\overline{\mu}$| The maximum of spectral radius |
| $\underline{\mu}$| The minimum of spectral radius |
| $\mathbf{G}$| The iteration matrix |

The paramters of iterative emthods has been optmized.
The SOR parameter $\omega^{opt}$:

$\omega^{opt}=\frac{2}{1+\sqrt{1-\overline{\mu}^{2}}}$

The AOR parameters $\alpha^{opt}, \beta^{opt}$: 

$\alpha^{opt}=\frac{2}{1+\sqrt{1-\overline{\mu}^{2}}}$

$\beta^{opt}=\frac{\overline{\mu}^{2}-\underline{\mu}^{2}}{\overline{\mu}^{2}(1-\underline{\mu}^{2})}.$

The RI parameter $\omega^{opt} $: 

$\omega^{opt}=\frac{2}{\underline{\mu}+\overline{\mu}}$
The Chebyshev acceleration method parameters $p_{n+1}, \gamma $: 

$p_{n+1}^{opt}=\frac{4}{4-\rho p_{n}}$

$\gamma^{opt}=\frac{2}{2-\underline{\mu}+\overline{\mu}}$

All of $\underline{\mu}, \overline{\mu}$ can calculated by the iteration matrix of Jacobi as 

<<<<<<< HEAD
$\mu =\rho(\mathbf{G}_{JA})=\rho(\mathbf{D}^{-1}(\mathbf{D}-\mathbf{A})).$
=======
$ \mu =\rho(\mathbf{G}_{JA})=\rho(\mathbf{D}^{-1}(\mathbf{D}-\mathbf{A})).$
>>>>>>> 49129962c65b00396d7c679ff04f60b715fb28ac

finding the spectral radius and eigenvectors can be a challenging task despite optimization efforts because these are intrinsic properties of a matrix and are not directly influenced by its parameters. This requires specialized techniques for their computation, and a high spectral radius can still negatively impact the convergence rate of iterative methods

# Deep unfolding network

A deep learning techniques such as the deep unfolding network can be used to address the challenge of finding optimization parameters without the need for difficult calculations of spectral radius and eigenvectors. This approach has shown promising results and can accelerate the iterative process used to solve linear systems, leading to faster convergence rates.

deep unfolding , which takes an iterative algorithm with a fixed number of iterations T, unravels its structure, and adds trainable parameters. These parameters are then trained using deep learning techniques such as loss functions, stochastic gradient descent, and back propagation.

The deep unfolding of the iterative algorithm  as follows:
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
The implement training can be express as 
![SORNet](https://user-images.githubusercontent.com/119638218/226128700-f03ae894-a69b-48b1-a4bf-a0a3d2820d8e.png)

| math vale      | Description                          |
| ----------- | ------------------------------------ |
| number of mini batches| 500 |
| learning rate of optimizer| 0.002 |
| initial values of $\omega$| 1.1|

