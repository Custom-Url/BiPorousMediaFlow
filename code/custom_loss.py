import torch

from utils.helpers import calc_pressure_grad_2d, compute_laplacian_fd_2d
from neuralop.losses import LpLoss, H1Loss
from neuralop.losses.data_losses import central_diff_2d


torch.manual_seed(0)


class MacroPermeabilityLoss(object):
    """
    Computes the macro-permeability loss based on the difference between
    predicted and true pressure gradients.
    """

    def __init__(self):
        super(MacroPermeabilityLoss, self).__init__()
        self.l1 = LpLoss(d=2, p=1, reduce_dims=[0, 1])

    def __call__(self, u_pred, x, y):
        # Implements || log(G(u)) - log(G(u_pred)) ||^2_2
        loss = 0.0
        data = {"x": x, "y": y}
        for xi, yi, u_predi in zip(data["x"], data["y"], u_pred):
            datai = {"x": xi, "y": yi}
            pgrad_true, pgrad_pred = calc_pressure_grad_2d(
                datai,
                u_predi.unsqueeze(0),
                mu=1,
                mue=1,
                L=[1, 1],
                device=u_predi.device,
                inputEncoded=False,
                pressureProvided=False,
            )

            loss += self.l1(pgrad_pred, pgrad_true)

        return loss / u_pred.shape[0]


class BetaULoss(object):
    """
    Computes the loss for the product of beta and velocity fields.
    """

    def __init__(self):
        super(BetaULoss, self).__init__()
        self.beta_loss = LpLoss(d=2, p=1, reduce_dims=[0, 1])

    def __call__(self, u_pred, x, y):
        beta = x[:, 0].unsqueeze(1)
        return self.beta_loss(beta * u_pred, beta * y)


class LaplacianLoss(object):
    """
    Computes the loss based on the Laplacian of the predicted and ground truth vector fields.
    """

    def __init__(self):
        super(LaplacianLoss, self).__init__()
        self.L2 = LpLoss(d=2, p=2, reduce_dims=[0, 1])

    def __call__(self, u_pred, x, y):
        _, _, dx, dy = (
            1 / u_pred.shape[0],
            1 / u_pred.shape[1],
            1 / u_pred.shape[2],
            1 / u_pred.shape[3],
        )
        laplacian_pred = compute_laplacian_fd_2d(
            u_pred, dx=dx, dy=dy, fix_x_bnd=True, fix_y_bnd=True
        )
        laplacian_gt = compute_laplacian_fd_2d(
            y, dx=dx, dy=dy, fix_x_bnd=True, fix_y_bnd=True
        )
        return self.L2(laplacian_pred, laplacian_gt)


class H1BetaULoss(object):
    """
    Combines H1 loss with beta-u regularisation term.

    Args:
        lam: Weighting factor for the beta loss.
    """

    def __init__(self, lam=0.5):
        super(H1BetaULoss, self).__init__()
        self.beta_loss = LpLoss(d=2, p=1, reduce_dims=[0, 1])
        self.h1_loss = H1Loss(d=2, reduce_dims=[0, 1], fix_x_bnd=True, fix_y_bnd=True)
        self.lam = lam

    def __call__(self, u_pred, x, y):
        beta = x[:, 0].unsqueeze(1)
        return self.h1_loss(u_pred, y) + self.lam * self.beta_loss(
            beta * u_pred, beta * y
        )


class H1BetaULaplacianLoss(object):
    """
    Combines H1 loss with beta-u and Laplacian of u regularisation term.

    Args:
        lam1: Weighting factor for the beta regularisation term.
        lam2: Weighting factor for the Laplacian regularisation term.
    """

    def __init__(self, lam1=0.1, lam2=0.1):
        super(H1BetaULaplacianLoss, self).__init__()
        self.beta_loss = LpLoss(d=2, p=1, reduce_dims=[0, 1])
        self.laplacian_loss = LaplacianLoss()
        self.h1_loss = H1Loss(d=2, reduce_dims=[0, 1], fix_x_bnd=True, fix_y_bnd=True)
        self.lam1 = lam1
        self.lam2 = lam2

    def __call__(self, u_pred, x, y):
        beta = x[:, 0].unsqueeze(1)
        return self.h1_loss(u_pred, y) + self.lam1 * self.beta_loss(
            beta * u_pred, beta * y + self.lam2 * self.laplacian_loss(u_pred, x, y)
        )


class H1LaplacianLoss(object):
    """
    Combines H1 loss with Laplacian of u regularisation term.

    Args:
        lam: Weighting factor for the Laplacian regularisation term.
    """

    def __init__(self, lam):
        super(H1LaplacianLoss, self).__init__()
        self.lam = lam
        self.h1_loss = H1Loss(d=2, reduce_dims=[0, 1], fix_x_bnd=True, fix_y_bnd=True)
        self.laplacian_loss = LaplacianLoss()

    def __call__(self, u_pred, x, y):
        return self.h1_loss(u_pred, y) + self.lam * self.laplacian_loss(u_pred, x, y)


class H2Loss(object):
    """
    Computes the H2 loss, which includes second-order derivatives of the predicted field.

    Args:
        reduce_dims: Dimensions to reduce during loss computation.
        reductions: Reduction method ('sum' or 'mean').
    """

    def __init__(self, reduce_dims=None, reductions="sum"):
        super(H2Loss, self).__init__()

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == "sum" or reductions == "mean"
                self.reductions = [reductions] * len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == "sum" or reductions[j] == "mean"
                self.reductions = reductions

    def compute_int_hessian_2d(self, u, dx, dy):
        """
        Computes the Hessian matrix (second derivatives) of a 2D field.

        Args:
            u: Input field.
            dx: Grid spacing in the x-direction.
            dy: Grid spacing in the y-direction.

        Returns:
            Tuple of first and second derivatives.
        """
        # Compute first derivatives
        u_x, u_y = central_diff_2d(u, [dx, dy], fix_x_bnd=True, fix_y_bnd=True)

        # Compute second derivatives
        u_xx, _ = central_diff_2d(u_x, [dx, dy], fix_x_bnd=True, fix_y_bnd=True)
        _, u_yy = central_diff_2d(u_y, [dx, dy], fix_x_bnd=True, fix_y_bnd=True)

        u_xy, _ = central_diff_2d(u_x, [dx, dy], fix_x_bnd=True, fix_y_bnd=True)

        u_x = torch.flatten(u_x, start_dim=-2)
        u_y = torch.flatten(u_y, start_dim=-2)
        u_xx = torch.flatten(u_xx, start_dim=-2)
        u_yy = torch.flatten(u_yy, start_dim=-2)
        u_xy = torch.flatten(u_xy, start_dim=-2)

        return u_x, u_y, u_xx, u_yy, u_xy

    # part copied from neuralop package
    def reduce_all(self, x):
        """
        Reduces the dimensions of a tensor based on the specified reduction method.

        Args:
            x: Input tensor.

        Returns:
            Reduced tensor.
        """
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == "sum":
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)

        return x

    def __call__(self, u_pred, x, y):
        _, _, dx, dy = (
            1 / u_pred.shape[0],
            1 / u_pred.shape[1],
            1 / u_pred.shape[2],
            1 / u_pred.shape[3],
        )
        x_pred, y_pred, xx_pred, yy_pred, xy_pred = self.compute_int_hessian_2d(
            u=u_pred, dx=dx, dy=dy
        )
        x_gt, y_gt, xx_gt, yy_gt, xy_gt = self.compute_int_hessian_2d(u=y, dx=dx, dy=dy)

        u_pred_flat = torch.flatten(u_pred, start_dim=-2)
        y_flat = torch.flatten(y, start_dim=-2)

        diff = torch.norm(u_pred_flat - y_flat, p=2, dim=-1, keepdim=False) ** 2
        ynorm = torch.norm(y_flat, p=2, dim=-1, keepdim=False) ** 2

        diff_1 = torch.norm(x_pred - x_gt, p=2, dim=-1, keepdim=False) ** 2
        ynorm_1 = torch.norm(x_gt, p=2, dim=-1, keepdim=False) ** 2
        diff_1 += torch.norm(y_pred - y_gt, p=2, dim=-1, keepdim=False) ** 2
        ynorm_1 += torch.norm(y_gt, p=2, dim=-1, keepdim=False) ** 2
        diff_2 = torch.norm(xx_pred - xx_gt, p=2, dim=-1, keepdim=False) ** 2
        ynorm_2 = torch.norm(xx_gt, p=2, dim=-1, keepdim=False) ** 2
        diff_2 += torch.norm(yy_pred - yy_gt, p=2, dim=-1, keepdim=False) ** 2
        ynorm_2 += torch.norm(yy_gt, p=2, dim=-1, keepdim=False) ** 2
        diff_2 += 2 * torch.norm(xy_pred - xy_gt, p=2, dim=-1, keepdim=False) ** 2
        ynorm_2 += 2 * torch.norm(xy_gt, p=2, dim=-1, keepdim=False) ** 2

        diff = ((diff + diff_1 + diff_2) ** 0.5) / ((ynorm + ynorm_1 + ynorm_2) ** 0.5)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff


class H2BetaLoss(object):
    """
    Combines H2 loss with beta-u regularisation term.

    Args:
        lam: Weighting factor for the beta regularisation term.
    """

    def __init__(self, lam=0.5):
        super(H2BetaLoss, self).__init__()
        self.beta_loss = LpLoss(d=2, p=1, reduce_dims=[0, 1])
        self.h2_loss = H2Loss(reduce_dims=[0, 1])
        self.lam = lam

    def __call__(self, u_pred, x, y):
        beta = x[:, 0].unsqueeze(1)
        return self.h2_loss(u_pred, x, y) + self.lam * self.beta_loss(
            beta * u_pred, beta * y
        )
