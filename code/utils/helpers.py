import numpy as np
from utils.classes import microstructure, load_fluid_condition, param_algo, grid
from neuralop.losses.differentiation import central_diff_2d

import torch

torch.manual_seed(0)
np.random.seed(0)


def recon_pressure_gradient(beta, vfield, L=[1, 1, 1]):
    """
    Reconstruct the pressure gradient using the given beta and velocity field.

    Args:
        beta: Tensor representing the beta field.
        vfield: Tensor representing the velocity field.
        L: List of domain dimensions.

    Returns:
        Tensor representing the reconstructed pressure gradient.
    """
    # beta - shape (BxCxWxHxD)
    # vfield - shape (BxCxWxHxD)

    Lx, Ly, Lz = L
    nx, ny, nz = beta.shape  # [2:]
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    # laplacian of velocity
    ii = torch.pi * torch.fft.fftfreq(nx, 1.0 / nx) / nx
    jj = torch.pi * torch.fft.fftfreq(ny, 1.0 / ny) / ny
    kk = torch.pi * torch.fft.fftfreq(nz, 1.0 / nz) / nz
    jj, ii, kk = torch.meshgrid(jj, ii, kk, indexing="ij")
    freq = torch.zeros((nx, ny, nz, 3), device=vfield.device)
    freq[..., 0] = 2.0 / dx * torch.sin(ii) * torch.cos(jj) * torch.cos(kk)
    freq[..., 1] = 2.0 / dy * torch.cos(ii) * torch.sin(jj) * torch.cos(kk)
    freq[..., 2] = 2.0 / dz * torch.cos(ii) * torch.cos(jj) * torch.sin(kk)
    freqSquare = freq[..., 0] ** 2 + freq[..., 1] ** 2 + freq[..., 2] ** 2
    freqSquare = freqSquare[None, None, ...]

    # term1: phi*laplacian(vfield)
    term1 = torch.zeros_like(vfield)
    for j0 in range(3):  # Assuming vfield has at least 2 channels
        term1[:, j0 : j0 + 1] = torch.fft.ifftn(
            -freqSquare * torch.fft.fftn(vfield[:, j0 : j0 + 1])
        ).real  # *phi which is 1 everywhere

    # term2: beta*vfield
    term2 = beta * vfield

    # grad pressure: term1 - term2, phi*laplacian(vfield) - beta*vfield
    return term1 - term2


def compute_laplacian_fd_2d(vfield, dx=1.0, dy=1.0, fix_x_bnd=False, fix_y_bnd=False):
    """
    Compute the Laplacian of a 2D velocity field using finite differences.

    Args:
        vfield: Tensor representing the velocity field.
        dx: Grid spacing in the x-direction.
        dy: Grid spacing in the y-direction.
        fix_x_bnd: Whether to fix boundary conditions in the x-direction.
        fix_y_bnd: Whether to fix boundary conditions in the y-direction.

    Returns:
        Tensor representing the Laplacian of the velocity field.
    """
    # Compute first derivatives using central_diff_2d
    first_dx, first_dy = central_diff_2d(vfield, [dx, dy], fix_x_bnd, fix_y_bnd)

    # Compute second derivatives using central_diff_2d on the first derivatives
    second_dxx, _ = central_diff_2d(first_dx, [dx, dy], fix_x_bnd, fix_y_bnd)
    _, second_dyy = central_diff_2d(first_dy, [dx, dy], fix_x_bnd, fix_y_bnd)

    # Sum of second derivatives is the Laplacian
    laplacian = second_dxx + second_dyy

    return laplacian


def recon_pressure_grad_2d_fd(beta, vfield, L=[1, 1], fix_x_bnd=False, fix_y_bnd=False):
    """
    Reconstruct the 2D pressure gradient using finite differences.

    Args:
        beta: Tensor representing the beta field.
        vfield: Tensor representing the velocity field.
        L: List of domain dimensions.
        fix_x_bnd: Whether to fix boundary conditions in the x-direction.
        fix_y_bnd: Whether to fix boundary conditions in the y-direction.

    Returns:
        Tensor representing the reconstructed pressure gradient.
    """
    vfield = vfield.unsqueeze(0)
    Lx, Ly = L
    _, _, nx, ny = vfield.shape
    dx, dy = Lx / nx, Ly / ny
    # compute Laplacian
    term1 = compute_laplacian_fd_2d(
        vfield=vfield, dx=dx, dy=dy, fix_x_bnd=fix_x_bnd, fix_y_bnd=fix_y_bnd
    )
    # compute beta*vfield
    term2 = beta * vfield
    return term1 - term2


def recon_pressure_grad_2d_fd_separate(
    beta, vfield, L=[1, 1], fix_x_bnd=False, fix_y_bnd=False
):
    """
    Reconstruct the 2D pressure gradient and return separate terms.

    Args:
        beta: Tensor representing the beta field.
        vfield: Tensor representing the velocity field.
        L: List of domain dimensions.
        fix_x_bnd: Whether to fix boundary conditions in the x-direction.
        fix_y_bnd: Whether to fix boundary conditions in the y-direction.

    Returns:
        Tuple of tensors representing separate terms of the pressure gradient.
    """
    vfield = vfield.unsqueeze(0)
    Lx, Ly = L
    _, _, nx, ny = vfield.shape
    dx, dy = Lx / nx, Ly / ny
    # compute Laplacian
    term1 = compute_laplacian_fd_2d(
        vfield=vfield, dx=dx, dy=dy, fix_x_bnd=fix_x_bnd, fix_y_bnd=fix_y_bnd
    )
    # compute beta*vfield
    term2 = beta * vfield
    return term1, term2


def recon_pressure_gradient_2d(beta, vfield, L=[1, 1]):
    """
    Reconstruct the 2D pressure gradient using the given beta and velocity field.

    Args:
        beta: Tensor representing the beta field.
        vfield: Tensor representing the velocity field.
        L: List of domain dimensions.

    Returns:
        Tensor representing the reconstructed pressure gradient.
    """
    # beta - shape (BxCxWxH)
    # vfield - shape (CxWxH)
    vfield = vfield.unsqueeze(0)
    Lx, Ly = L
    _, _, nx, ny = vfield.shape
    dx, dy = Lx / nx, Ly / ny

    # laplacian of velocity v1
    ii = torch.pi * torch.fft.fftfreq(nx, 1.0 / nx) / nx
    jj = torch.pi * torch.fft.fftfreq(ny, 1.0 / ny) / ny
    jj, ii = torch.meshgrid(jj, ii, indexing="ij")
    freq = torch.zeros((nx, ny, 2), device=vfield.device)
    freq[..., 0] = 2.0 / dx * torch.sin(ii) * torch.cos(jj)
    freq[..., 1] = 2.0 / dy * torch.cos(ii) * torch.sin(jj)
    freqSquare = freq[..., 0] ** 2 + freq[..., 1] ** 2
    freqSquare = freqSquare[None, None, ...]

    # term1: phi*laplacian(vfield)
    term1 = torch.zeros_like(vfield)
    for j0 in range(vfield.shape[1]):
        term1[:, j0] = torch.fft.ifftn(
            -freqSquare * torch.fft.fftn(vfield[:, j0])
        ).real  # *phi which is 1 everywhere

    # term2: beta*vfield
    term2 = beta * vfield

    # grad pressure: term1 - term2, phi*laplacian(vfield) - beta*vfield
    return term1 - term2


def calc_pressure_grad(
    data,
    out=None,
    mu=1,
    mue=1,
    L=[1, 1, 1],
    device="cpu",
    inputEncoded=True,
    pressureProvided=False,
):
    """
    Calculate the pressure gradient for 3D data.

    Args:
        data: Dictionary containing input data.
        out: Optional tensor for predicted output.
        mu: Fluid viscosity.
        mue: Solid region viscosity.
        L: List of domain dimensions.
        device: Device to perform computations on.
        inputEncoded: Whether the input is encoded.
        pressureProvided: Whether the pressure is provided in the data.

    Returns:
        Tensor representing the calculated pressure gradient.
    """
    if inputEncoded == False:  # out: BxCxWxH
        beta = data["x"][0]  # data['x']: CxWxH, data['y']: CxWxH
    elif inputEncoded == True:
        beta = 10 ** data["x"][0]
    phi = beta * 0
    phi[data["x"][0] == 0] = mu
    phi[data["x"][0] != 0] = mue

    phi = phi[..., None].detach().to(device)
    beta = beta[..., None].detach().to(device)

    if pressureProvided == True:
        return torch.moveaxis(data["y"][2:], 0, -1)[:, :, None, :].detach().to(
            device
        ), torch.moveaxis(out[0, 2:], 0, -1)[:, :, None, :].detach().to(device)
    else:
        if out is None:  # use the ground truth
            vfield = (
                torch.moveaxis(data["y"][:2], 0, -1)[:, :, None, :].detach().to(device)
            )
            return recon_pressure_grad(phi, beta, vfield, L=L, device=device)
        else:
            vfield_true = (
                torch.moveaxis(data["y"][:2], 0, -1)[:, :, None, :].detach().to(device)
            )
            vfield_pred = (
                torch.moveaxis(out[0, :2], 0, -1)[:, :, None, :].detach().to(device)
            )

            return recon_pressure_gradient(
                beta, vfield_true, L=L
            ), recon_pressure_gradient(beta, vfield_pred, L=L)


def calc_pressure_grad_2d(
    data,
    out=None,
    mu=1,
    mue=1,
    L=[1, 1],
    device="cpu",
    inputEncoded=True,
    pressureProvided=False,
):
    """
    Calculate the pressure gradient for 2D data.

    Args:
        data: Dictionary containing input data.
        out: Optional tensor for predicted output.
        mu: Fluid viscosity.
        mue: Solid region viscosity.
        L: List of domain dimensions.
        device: Device to perform computations on.
        inputEncoded: Whether the input is encoded.
        pressureProvided: Whether the pressure is provided in the data.

    Returns:
        Tensor representing the calculated pressure gradient.
    """
    if not inputEncoded:  # out: BxCxWxH
        beta = data["x"][0]  # data['x']: CxW, data['y']: CxW
    else:
        beta = torch.pow(data["x"][0], 10)
    phi = torch.zeros_like(beta)
    phi[data["x"][0] == 0] = mu
    phi[data["x"][0] != 0] = mue

    phi = phi.detach().to(device)
    beta = beta.detach().to(device)

    if pressureProvided:
        return data["y"].detach().to(device), out[0, 1:].detach().to(device)
    else:
        if out is None:  # use the ground truth
            vfield = data["y"].detach().to(device)
            return recon_pressure_grad_2d_fd(
                phi, beta, vfield, L=L, device=device, fix_x_bnd=True, fix_y_bnd=True
            )
        else:
            vfield_true = data["y"].to(device)
            vfield_pred = out[0].to(device)
            x_component = vfield_pred[..., 0]
            y_component = vfield_pred[..., 1]
            return recon_pressure_grad_2d_fd(
                beta, vfield_true, L=L, fix_x_bnd=True, fix_y_bnd=True
            ), recon_pressure_grad_2d_fd(
                beta, vfield_pred, L=L, fix_x_bnd=True, fix_y_bnd=True
            )


def preparation(data, out, J=[1, 0, 0], inputEncoded=True, mu=1):
    """
    Prepare the input data and initial conditions for the simulation.

    Args:
        data: Dictionary containing input data.
        out: Tensor for predicted output.
        J: Gradient of pressure.
        inputEncoded: Whether the input is encoded.
        mu: Fluid viscosity.

    Returns:
        Tuple containing microstructure, fluid condition, algorithm parameters, and initial velocity field.
    """
    # input
    x = torch.movedim(data["x"], 0, -1).unsqueeze(2)[..., 0].detach()
    if inputEncoded:
        x = torch.where(x > 0, 10**x, x)

    # rescale beta with new mu
    if mu != 1:
        x = x * mu

    # output
    out = torch.movedim(out[0], 0, -1).unsqueeze(2).detach()

    Ifn = (x > 0).to(torch.uint8)
    beta = torch.stack([x, x, x, x * 0, x * 0, x * 0], dim=-1)

    beta0 = beta.max() / 2
    phi0 = (mu + mue) / 2  # mue needs to be defined or passed as an argument

    # initial velocity field
    vfield0 = torch.stack((out[..., 0], out[..., 1], out[..., 1] * 0.0), dim=-1)

    # microstructure
    m0 = microstructure(
        Ifn=Ifn,  # labeled image
        L=[1, 1, 1 / beta.shape[0]],  # physical dimension of RVE
        label_fluid=0,  # label for fluid region
        label_solid=1,  # label for solid(porous) region
        micro_beta=beta,  # local fluctuation field beta
    )

    # algorithm parameters
    p0 = param_algo(
        cv_criterion=1e-6,  # convergence criterion
        reference_phi0=phi0,
        reference_beta0=beta0,
        itMax=10000,  # max number of iterations
        cv_acc=True,
        AA_depth=4,
        AA_increment=2,
    )

    # -load & fluid condition
    l0 = load_fluid_condition(
        macro_load=J,  # gradient of pressure
        viscosity=mu,  # fluid viscosity
        viscosity_solid=mue,  # fluid viscosity in solid region
    )

    return m0, l0, p0, vfield0
