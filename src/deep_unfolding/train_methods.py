import torch
import torch.nn as nn

from .utils import decompose_matrix, device


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_func: torch.nn.Module,
    solution: torch.Tensor,
    total_itr: int = 25,
    num_batch: int = 10000,
) -> tuple[torch.nn.Module, list[float]]:
    """Train the given model using the specified optimizer and loss function.

    Args:
      model: The model to be trained.
      optimizer: The optimizer to use for training.
      loss_func: The loss function to use for training.
      total_itr: The total number of iterations (generations) for training.
      solution: The target solution tensor.
      num_batch: The number of batches per iteration.

    Returns:
      The trained model and the list of loss values per iteration.
    """
    loss_gen = []
    for gen in range(total_itr):
        for i in range(num_batch):
            optimizer.zero_grad()
            x_hat, _ = model(gen + 1)
            loss = loss_func(x_hat, solution)
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                print("generation:", gen + 1, " batch:", i, "\t MSE loss:", loss.item())
        loss_gen.append(loss.item())
    return model, loss_gen


def evaluate_model(
    model: torch.nn.Module,
    solution: torch.Tensor,
    n: int,
    bs: int = 10000,
    total_itr: int = 25,
    device: torch.device = device,
) -> list[float]:
    """Evaluate the model by calculating the mean squared error (MSE) between
      the solution and the model's predictions.

    Args:
      model: The model to be evaluated.
      solution: The target solution tensor.
      n: The dimension of the solution.
      bs: The batch size.
      total_itr: The total number of iterations for evaluation.
      device: The device to run the evaluation on.

    Returns:
      A list of MSE values for each iteration.
    """
    norm_list = []
    for i in range(total_itr + 1):
        s_hat, _ = model(i)
        err = (torch.norm(solution.to(device) - s_hat.to(device)) ** 2).item() / (
            n * bs
        )
        norm_list.append(err)
    return norm_list


class SORNet(nn.Module):
    """Deep unfolded SOR with a constant step size."""

    device: torch.device
    """Device to run the model on."""

    inv_omega: nn.Parameter
    """Inverse of the relaxation parameter omega."""

    A: torch.Tensor
    """Matrix $A$ of the linear system."""

    D: torch.Tensor
    """Diagonal matrix $D$."""

    L: torch.Tensor
    """Lower triangular matrix $L$."""

    U: torch.Tensor
    """Upper triangular matrix $U$."""

    H: torch.Tensor
    """Matrix $H$."""

    Dinv: torch.Tensor
    """Inverse of the diagonal matrix $D$."""

    bs: int
    """Batch size."""

    y: torch.Tensor
    """Solution of the linear equation."""

    def __init__(
        self,
        A: torch.Tensor,
        H: torch.Tensor,
        bs: int,
        y: torch.Tensor,
        init_val_SORNet: float = 1.1,
        device: torch.device = device,
    ):
        """Initialize the SORNet model.

        Args:
          A: Matrix $A$ of the linear system.
          H: Matrix $H$.
          bs: Batch size.
          y: Solution of the linear equation.
          init_val_SORNet: Initial value for `inv_omega`.
          device: Device to run the model on ('cpu' or 'cuda').
        """
        super(SORNet, self).__init__()
        self.device = device
        self.inv_omega = nn.Parameter(torch.tensor(init_val_SORNet, device=device))

        A, D, L, U, _, _ = decompose_matrix(A)

        self.A = A.to(device)
        self.D = D.to(device)
        self.L = L.to(device)
        self.U = U.to(device)
        self.H = H.to(device)
        self.Dinv = torch.linalg.inv(D).to(device)
        self.bs = bs
        self.y = y.to(device)

    def forward(self, num_itr: int = 25) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Perform forward pass of the SORNet model.

        Args:
          num_itr: Number of iterations.

        Returns:
          A tuple with the following contents:
            - Output tensor.
            - List of intermediate results.
        """
        traj = []

        invM = torch.linalg.inv(self.inv_omega * self.D + self.L)
        s = torch.zeros(self.bs, self.H.size(0), device=self.device)
        traj.append(s)
        yMF = torch.matmul(self.y, self.H.T)
        s = torch.matmul(yMF, self.Dinv)

        for _ in range(num_itr):
            temp = torch.matmul(s, (self.inv_omega - 1) * self.D - self.U) + yMF
            s = torch.matmul(temp, invM)
            traj.append(s)

        return s, traj


class SOR_CHEBY_Net(nn.Module):
    """Deep unfolded SOR with Chebyshev acceleration."""

    device: torch.device
    """Device to run the model on."""

    gamma: nn.Parameter
    """Gamma parameter for each iteration."""

    omega: nn.Parameter
    """Omega parameter for each iteration."""

    inv_omega: nn.Parameter
    """Inverse of the relaxation parameter omega."""

    A: torch.Tensor
    """Matrix $A$ of the linear system."""

    D: torch.Tensor
    """Diagonal matrix $D$."""

    L: torch.Tensor
    """Lower triangular matrix $L$."""

    U: torch.Tensor
    """Upper triangular matrix $U$."""

    H: torch.Tensor
    """Matrix $H$."""

    Dinv: torch.Tensor
    """Inverse of the diagonal matrix $D$."""

    bs: int
    """Batch size."""

    y: torch.Tensor
    """Solution of the linear equation."""

    def __init__(
        self,
        num_itr: int,
        A: torch.Tensor,
        H: torch.Tensor,
        bs: int,
        y: torch.Tensor,
        init_val_SOR_CHEBY_Net_omega: float = 0.6,
        init_val_SOR_CHEBY_Net_gamma: float = 0.8,
        init_val_SOR_CHEBY_Net_alpha: float = 0.9,
        device: torch.device = device,
    ):
        """Initialize the SOR_CHEBY_Net model.

        Args:
          num_itr: Number of iterations.
          A: Matrix $A$ of the linear system.
          H: Matrix $H$.
          bs: Batch size.
          y: Solution of the linear equation.
          init_val_SOR_CHEBY_Net_omega: Initial value for `omega`.
          init_val_SOR_CHEBY_Net_gamma: Initial value for `gamma`.
          init_val_SOR_CHEBY_Net_alpha: Initial value for `inv_omega`.
          device: Device to run the model on ('cpu' or 'cuda').
        """
        super(SOR_CHEBY_Net, self).__init__()
        self.device = device
        self.gamma = nn.Parameter(
            init_val_SOR_CHEBY_Net_gamma * torch.ones(num_itr, device=device)
        )
        self.omega = nn.Parameter(
            init_val_SOR_CHEBY_Net_omega * torch.ones(num_itr, device=device)
        )
        self.inv_omega = nn.Parameter(
            torch.tensor(init_val_SOR_CHEBY_Net_alpha, device=device)
        )

        A, D, L, U, _, _ = decompose_matrix(A)
        self.A = A
        self.D = D.to(device)
        self.L = L.to(device)
        self.U = U.to(device)
        self.H = H.to(device)
        self.Dinv = torch.linalg.inv(D).to(device)
        self.bs = bs
        self.y = y.to(device)

    def forward(self, num_itr: int = 25) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Perform forward pass of the SOR_CHEBY_Net model.

        Args:
          num_itr: Number of iterations.

        Returns:
          A tuple with the following contents:
            - Output tensor.
            - List of intermediate results.
        """
        traj = []

        invM = torch.linalg.inv(self.inv_omega * self.D + self.L)
        s = torch.zeros(self.bs, self.H.size(0), device=self.device)  # modif to size(0)
        s_new = torch.zeros(
            self.bs, self.H.size(0), device=self.device
        )  # modif to size(0)
        traj.append(s)
        yMF = torch.matmul(self.y, self.H.T)
        s = torch.matmul(yMF, self.Dinv)
        s_present = s
        s_old = torch.zeros_like(s_present)

        for i in range(num_itr):
            temp = torch.matmul(s, (self.inv_omega - 1) * self.D - self.U) + yMF
            s = torch.matmul(temp, invM)

            s_new = (
                self.omega[i] * (self.gamma[i] * (s - s_present) + (s_present - s_old))
                + s_old
            )
            s_old = s
            s_present = s_new
            traj.append(s_new)

        return s_new, traj


# =====================================================================================


class AORNet(nn.Module):
    """Deep unfolded AOR with a constant step size."""

    device: torch.device
    """Device to run the model on."""

    r: nn.Parameter
    """Parameter `r` for AOR."""

    omega: nn.Parameter
    """Relaxation parameter omega."""

    A: torch.Tensor
    """Matrix $A$ of the linear system."""

    D: torch.Tensor
    """Diagonal matrix $D$."""

    L: torch.Tensor
    """Lower triangular matrix $L$."""

    U: torch.Tensor
    """Upper triangular matrix $U$."""

    H: torch.Tensor
    """Matrix $H$."""

    Dinv: torch.Tensor
    """Inverse of the diagonal matrix $D$."""

    bs: int
    """Batch size."""

    y: torch.Tensor
    """Solution of the linear equation."""

    def __init__(
        self,
        A: torch.Tensor,
        H: torch.Tensor,
        bs: int,
        y: torch.Tensor,
        init_val_AORNet_r: float = 0.9,
        init_val_AORNet_omega: float = 1.5,
        device: torch.device = device,
    ):
        """
        Initialize the AORNet model.

        Args:
          A: Matrix $A$ of the linear system.
          H: Matrix $H$.
          bs: Batch size.
          y: Solution of the linear equation.
          init_val_AORNet_r: Initial value for `r`.
          init_val_AORNet_omega: Initial value for `omega`.
          device: Device to run the model on ('cpu' or 'cuda').
        """
        super(AORNet, self).__init__()
        self.device = device
        self.r = nn.Parameter(torch.tensor(init_val_AORNet_r, device=device))
        self.omega = nn.Parameter(torch.tensor(init_val_AORNet_omega, device=device))

        A, D, L, U, _, _ = decompose_matrix(A)
        self.A = A.to(device)
        self.D = D.to(device)
        self.L = L.to(device)
        self.U = U.to(device)
        self.H = H.to(device)
        self.Dinv = torch.linalg.inv(D).to(device)
        self.bs = bs
        self.y = y.to(device)

    def forward(self, num_itr: int = 25) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Perform forward pass of the AORNet model.

        Args:
          num_itr: Number of iterations.

        Returns:
          A tuple with the following contents:
          - Output tensor.
          - List of intermediate results.
        """
        traj = []

        invM = torch.linalg.inv(self.L - self.r * self.D)
        N = (
            (1 - self.omega) * self.D
            + (self.omega - self.r) * self.L
            + self.omega * self.U
        )
        s = torch.zeros(
            self.bs, self.H.size(0), device=self.device
        )  # change to size(0)
        traj.append(s)
        yMF = torch.matmul(self.y, self.H.T)
        s = torch.matmul(yMF, self.Dinv)

        for _ in range(num_itr):
            s = torch.matmul(s, torch.matmul(invM, N)) + torch.matmul(yMF, invM)
            traj.append(s)

        return s, traj


class RINet(nn.Module):
    """Deep unfolded Richardson iteration."""

    inv_omega: nn.Parameter
    """Inverse of the relaxation parameter omega."""

    A: torch.Tensor
    """Matrix $A$ of the linear system."""

    D: torch.Tensor
    """Diagonal matrix $D$."""

    L: torch.Tensor
    """Lower triangular matrix $L$."""

    U: torch.Tensor
    """Upper triangular matrix $U$."""

    H: torch.Tensor
    """Matrix $H$."""

    Dinv: torch.Tensor
    """Inverse of the diagonal matrix $D$."""

    bs: int
    """Batch size."""

    y: torch.Tensor
    """Solution of the linear equation."""

    def __init__(
        self,
        A: torch.Tensor,
        H: torch.Tensor,
        bs: int,
        y: torch.Tensor,
        init_val_RINet: float = 0.1,
        device: torch.device = device,
    ):
        """Initialize the RINet model.

        Args:
          A: Matrix $A$ of the linear system.
          H: Matrix $H$.
          bs: Batch size.
          y: Solution of the linear equation.
          init_val_RINet: Initial value for `inv_omega`.
          device: Device to run the model on ('cpu' or 'cuda').
        """
        super(RINet, self).__init__()
        self.device = device
        self.inv_omega = nn.Parameter(torch.tensor(init_val_RINet, device=device))

        A, D, L, U, _, _ = decompose_matrix(A)
        self.A = A.to(device)
        self.D = D.to(device)
        self.L = L.to(device)
        self.U = U.to(device)
        self.H = H.to(device)
        self.Dinv = torch.linalg.inv(D).to(device)
        self.bs = bs
        self.y = y.to(device)

    def forward(self, num_itr: int = 25) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Perform forward pass of the RINet model.

        Args:
          num_itr: Number of iterations.

        Returns:
          A tuple with the following contents:
            - Output tensor.
            - List of intermediate results.
        """
        traj = []

        s = torch.zeros(self.bs, self.A.shape[0], device=self.device)
        traj.append(s)
        yMF = torch.matmul(self.y, self.H.T)
        s = torch.matmul(yMF, self.Dinv)

        for _ in range(num_itr):
            s = s + self.inv_omega * (yMF - torch.matmul(s, self.A))
            traj.append(s)

        return s, traj
