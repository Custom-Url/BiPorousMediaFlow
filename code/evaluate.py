# %%
import os
import torch
import argparse
import datetime
import wandb
import copy
import numpy as np
from neuralop.models.fno import TFNO, FNO
from matplotlib.ticker import LogLocator, ScalarFormatter


from neuralop.training.trainer import Trainer
from neuralop.utils import count_model_params
from neuralop.losses.data_losses import LpLoss, H1Loss
from utils.brinkman_amitex import load_stokesbrinkman
from utils.helpers import (
    calc_pressure_grad_2d,
    recon_pressure_grad_2d_fd,
)

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

device = "cuda"
torch.manual_seed(0)
np.random.seed(0)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return "True"
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return "False"
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# %%
def unpack_data(model, data, data_processor, idir, encode_output=None):
    """
    Unpack data for model prediction and postprocess the output.

    Args:
        model: The model to use for prediction.
        data: Input data dictionary containing 'x' and 'y'.
        data_processor: Processor for data preprocessing and postprocessing.
        idir: Direction indicator (0 or 1).
        encode_output: Optional encoding for output data.

    Returns:
        Tuple of input, ground truth, and model output tensors.
    """
    # Model prediction
    if idir == 0:
        out = model(data["x"].unsqueeze(0))
        out, data = data_processor.postprocess(out, data)
        x = data["x"].cpu()
        y = data["y"].cpu()
    elif idir == 1:
        out = model(torch.transpose(data["x"].unsqueeze(0), -2, -1))
        x = torch.transpose(data["x"], -2, -1).cpu()
        y = torch.transpose(data["y"], -2, -1).cpu()

    out = out.detach().cpu()

    # decoding
    if encode_output is not None:
        out[0] = torch.as_tensor(encode_output.decode(out.numpy()[0]))
        y = torch.as_tensor(encode_output.decode(y))

    return x, y, out


def compare_vfield(model, data, data_processor, index, encode_output=None, fig=None):
    """
    Compare vector fields between ground truth and model predictions.

    Args:
        model: The model to use for prediction.
        data: Input data dictionary containing 'x' and 'y'.
        data_processor: Processor for data preprocessing and postprocessing.
        index: Index for subplot arrangement.
        encode_output: Optional encoding for output data.
        fig: Matplotlib figure for plotting.

    Returns:
        Updated figure with subplots showing comparisons.
    """
    # unpack
    x, y, out = unpack_data(model, data, index, data_processor, encode_output)

    #
    if encode_output is not None and encode_output.options["pvout"] == True:
        ncols = 9
    else:
        ncols = 5
    # input - log beta
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    ax = fig.add_subplot(3, ncols, index * ncols + 1)
    im = ax.imshow(x[0], cmap="gray")
    if index == 0:
        ax.set_title(r"$\log_{10} \beta$")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax, orientation="horizontal")
    # output - ux
    vmin, vmax = y[0].min(), y[0].max()
    ax = fig.add_subplot(3, ncols, index * ncols + 2)
    im = ax.imshow(y[0].squeeze(), vmin=vmin, vmax=vmax)
    if index == 0:
        ax.set_title(r"$u_1$ (GT)")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax, orientation="horizontal")
    #
    ax = fig.add_subplot(3, ncols, index * ncols + 3)
    im = ax.imshow(out[0, 0].squeeze().detach().numpy(), vmin=vmin, vmax=vmax)
    if index == 0:
        ax.set_title(r"$u_1$ (TFNO)")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax, orientation="horizontal")
    # output - uy
    vmin, vmax = y[1].min(), y[1].max()
    ax = fig.add_subplot(3, ncols, index * ncols + 4)
    im = ax.imshow(y[1].squeeze(), vmin=vmin, vmax=vmax)
    if index == 0:
        ax.set_title(r"$u_2$ (GT)")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax, orientation="horizontal")
    #
    ax = fig.add_subplot(3, ncols, index * ncols + 5)
    im = ax.imshow(out[0, 1].squeeze().detach().numpy(), vmin=vmin, vmax=vmax)
    if index == 0:
        ax.set_title(r"$u_2$ (TFNO)")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar(im, ax=ax, orientation="horizontal")

    if encode_output is not None and encode_output.options["pvout"] == True:
        # output - px
        vmin, vmax = y[2].min(), y[2].max()
        ax = fig.add_subplot(3, ncols, index * ncols + 6)
        im = ax.imshow(y[2].squeeze(), vmin=vmin, vmax=vmax)
        if index == 0:
            ax.set_title("px (FFT)")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.colorbar(im, ax=ax, orientation="horizontal")
        #
        ax = fig.add_subplot(3, ncols, index * ncols + 7)
        im = ax.imshow(out[0, 2].squeeze().detach().numpy(), vmin=vmin, vmax=vmax)
        if index == 0:
            ax.set_title("px (FNO)")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.colorbar(im, ax=ax, orientation="horizontal")
        # output - py
        vmin, vmax = y[3].min(), y[2].max()
        ax = fig.add_subplot(3, ncols, index * ncols + 8)
        im = ax.imshow(y[3].squeeze(), vmin=vmin, vmax=vmax)
        if index == 0:
            ax.set_title("py (FFT)")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.colorbar(im, ax=ax, orientation="horizontal")
        #
        ax = fig.add_subplot(3, ncols, index * ncols + 9)
        im = ax.imshow(out[0, 3].squeeze().detach().numpy(), vmin=vmin, vmax=vmax)
        if index == 0:
            ax.set_title("py (FNO)")
        plt.xticks([], [])
        plt.yticks([], [])
        plt.colorbar(im, ax=ax, orientation="horizontal")
    plt.tight_layout()

    return fig


def load_outputs_loss_functions(
    model, FOLDER_DIR, model_checkpoint_arr, test_loader, data_processor
):
    """
    Load model outputs and compute loss functions for multiple checkpoints.

    Args:
        model: The model to evaluate.
        FOLDER_DIR: Directory containing model checkpoints.
        model_checkpoint_arr: List of checkpoint filenames.
        test_loader: DataLoader for test data.
        data_processor: Processor for data preprocessing and postprocessing.

    Returns:
        Tensor array of model outputs for each checkpoint.
    """
    out_arr = torch.zeros(
        (len(model_checkpoint_arr) + 2, 100, 2, 64, 64), device=torch.device("cuda")
    )

    for i in range(len(model_checkpoint_arr) + 2):
        if i == 0:
            for j, test_data in enumerate(test_loader):
                out_arr[i, j] = test_data["x"].to(device)
        elif i == 1:
            for j, test_data in enumerate(test_loader):
                out_arr[i, j] = test_data["y"].to(device)
        else:
            checkpoint = torch.load(
                FOLDER_DIR + f"/<checkpoints_dir>/{model_checkpoint_arr[i-2]}_last.pt"
            )
            model.load_state_dict(checkpoint["state_dict"])
            with torch.no_grad():
                model.eval()
                for j, data in enumerate(test_loader):
                    data = data_processor.preprocess(data, batched=False)
                    x = data["x"].unsqueeze(0).to(device)
                    out = model(x)
                    out, data = data_processor.postprocess(out, data)
                    out_arr[i, j] = out

    return out_arr


def plot_images_grid(out_arr, idx=0):
    """
    Plot a grid of images for model outputs and ground truth.

    Args:
        out_arr: Array of outputs to plot.
        idx: Index of the sample to visualize.
    """
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    loss_arr_1 = [
        r"$\log_{10} \beta$",
        "GT",
        r"$L^2$",
        r"$H^1$",
        r"$H^1 \beta u$",
        r"$H^1 \beta u \Delta u$",
    ]
    loss_arr_2 = [
        r"$\log_{10} \beta$",
        "GT",
        r"$H^2$",
        r"$H^2 \beta u$",
        r"G",
    ]

    for k in range(2):
        if k == 0:
            end = 6
            loss_arr = loss_arr_1
            fig, axes = plt.subplots(2, end, figsize=(15, 10))
        else:
            end = 5
            loss_arr = loss_arr_2
            fig, axes = plt.subplots(2, end, figsize=(15, 10))
        for i in range(2):
            for j in range(end):
                ax = axes[i, j]
                if j == 0:
                    im = ax.imshow(
                        out_arr[j, idx, i].detach().cpu().numpy(), cmap="gray"
                    )
                elif j == 1:
                    im = ax.imshow(out_arr[j, idx, i].detach().cpu().numpy())
                else:
                    if k == 0:
                        im = ax.imshow(out_arr[j, idx, i].detach().cpu().numpy())
                    else:
                        im = ax.imshow(out_arr[j + 4, idx, i].detach().cpu().numpy())

                cbar = fig.colorbar(
                    im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04
                )
                cbar.formatter = ScalarFormatter(useMathText=True)
                cbar.formatter.set_powerlimits((-3, 3))
                cbar.update_ticks()
                cbar.ax.tick_params(labelsize=8)
                ax.set_title(f"{loss_arr[j]}")
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 1:
                    if i % 2 == 0:
                        ax.set_ylabel(r"$u_{1}$")
                    else:
                        ax.set_ylabel(r"$u_{2}$")
                fig.add_subplot(ax)

        plt.tight_layout()
        plt.show()


def compute_K(beta1, beta2, u1, u2, device, L=[1, 1]):
    """
    Compute the macro-permeability tensor K using Darcy's law.

    Args:
        beta1: Logarithm of beta for loading condition 1.
        beta2: Logarithm of beta for loading condition 2.
        u1: Velocity field for loading condition 1.
        u2: Velocity field for loading condition 2.
        device: Device to perform computations on.
        L: Domain dimensions.

    Returns:
        Macro-permeability tensor K.
    """
    beta1 = 10**beta1
    beta2 = 10**beta2
    G1 = recon_pressure_grad_2d_fd(
        beta1.to(device), u1.to(device), L=L, fix_x_bnd=True, fix_y_bnd=True
    )
    G2 = recon_pressure_grad_2d_fd(
        beta2.to(device), u2.to(device), L=L, fix_x_bnd=True, fix_y_bnd=True
    )
    H = torch.zeros((2, 2), device=device)
    H[:, 0] = -torch.as_tensor(
        [torch.mean(G1[:, i, :, :].flatten()).item() for i in range(2)]
    )
    H[:, 1] = -torch.as_tensor(
        [torch.mean(G2[:, i, :, :].flatten()).item() for i in range(2)]
    )
    K = torch.linalg.solve(H, torch.eye(2, device=device))
    return K


def plot_Kcompar(
    K11_true,
    K11_pred,
    K12_true,
    K12_pred,
    K21_true,
    K21_pred,
    K22_true,
    K22_pred,
    indices=None,
    save_path=None,
):
    """
    Plot comparisons of true and predicted macro permeability.

    Args:
        K11_true, K12_true, K21_true, K22_true: True macro permeability.
        K11_pred, K12_pred, K21_pred, K22_pred: Predicted macro permeability.
        indices: Optional indices to highlight specific points.
        save_path: Optional path to save the plot.
    """
    plt.rcParams["font.size"] = "10"
    fig, ax = plt.subplots(2, 2)
    mksiz = 2

    ax[0, 0].loglog(K11_true, K11_pred, ".", markersize=mksiz)
    lims = [min(K11_true + K11_pred).cpu().numpy(), max(K11_true + K11_pred).numpy()]
    ax[0, 0].set_xlim(lims)
    ax[0, 0].set_ylim(lims)
    ax[0, 0].loglog(lims, lims, "-k", linewidth=1)
    xticks = np.array(ax[0, 0].get_xticks())
    xticks = xticks[(xticks >= lims[0]) & (xticks <= lims[1])]
    ax[0, 0].set_xticks(xticks)
    ax[0, 0].set_yticks(xticks)
    ax[0, 0].set_aspect("equal", "box")
    ax[0, 0].set_xlabel(r"GT $K_{11}$")
    ax[0, 0].set_ylabel(r"Predicted $K_{11}$")

    ax[0, 1].loglog(K12_true, K12_pred, ".", markersize=mksiz)
    lims = [min(K12_true + K12_pred).cpu().numpy(), max(K12_true + K12_pred).numpy()]
    ax[0, 1].set_xlim(lims)
    ax[0, 1].set_ylim(lims)
    ax[0, 1].loglog(lims, lims, "-k", linewidth=1)
    xticks = np.array(ax[0, 1].get_xticks())
    xticks = xticks[(xticks >= lims[0]) & (xticks <= lims[1])]
    ax[0, 1].set_xticks(xticks)
    ax[0, 1].set_yticks(xticks)
    ax[0, 1].set_aspect("equal", "box")
    ax[0, 1].set_xlabel(r"GT $K_{12}$")
    ax[0, 1].set_ylabel(r"Predicted $K_{12}$")

    ax[1, 0].loglog(K21_true, K21_pred, ".", markersize=mksiz)
    lims = [min(K21_true + K21_pred).cpu().numpy(), max(K21_true + K21_pred).numpy()]
    ax[1, 0].set_xlim(lims)
    ax[1, 0].set_ylim(lims)
    ax[1, 0].loglog(lims, lims, "-k", linewidth=1)
    xticks = np.array(ax[1, 0].get_xticks())
    xticks = xticks[(xticks >= lims[0]) & (xticks <= lims[1])]
    ax[1, 0].set_xticks(xticks)
    ax[1, 0].set_yticks(xticks)
    ax[1, 0].set_aspect("equal", "box")
    ax[1, 0].set_xlabel(r"GT $K_{21}$")
    ax[1, 0].set_ylabel(r"Predicted $K_{21}$")

    ax[1, 1].loglog(K22_true, K22_pred, ".", markersize=mksiz)
    lims = [min(K22_true + K22_pred).cpu().numpy(), max(K22_true + K22_pred).numpy()]
    ax[1, 1].set_xlim(lims)
    ax[1, 1].set_ylim(lims)
    ax[1, 1].loglog(lims, lims, "-k", linewidth=1)
    xticks = np.array(ax[1, 1].get_xticks())
    xticks = xticks[(xticks >= lims[0]) & (xticks <= lims[1])]
    ax[1, 1].set_xticks(xticks)
    ax[1, 1].set_yticks(xticks)
    ax[1, 1].set_aspect("equal", "box")
    ax[1, 1].set_xlabel(r"GT $K_{22}$")
    ax[1, 1].set_ylabel(r"Predicted $K_{22}$")

    if indices is not None:
        indices = np.array(indices)
        ax[0, 0].loglog(K11_true[indices], K11_pred[indices], "d")
        ax[0, 1].loglog(K12_true[indices], K12_pred[indices], "d")
        ax[1, 0].loglog(K21_true[indices], K21_pred[indices], "d")
        ax[1, 1].loglog(K22_true[indices], K22_pred[indices], "d")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)


def calc_macro_permeability(model0, model1, res, save_path=None):
    """
    Calculate macro-permeability.

    Args:
        model0: Model for loading condition 0.
        model1: Model for loading condition 1.
        res: Resolution of the data.
        save_path: Optional path to save the results.

    Returns:
        True and predicted permeability tensor components.
    """
    # load the data (two simulations per sample)
    device = torch.device("cuda")
    RESs = [res]
    test_loaders = ({}, {})
    for velo_dir in range(2):
        if velo_dir == 0:
            for RES in RESs:
                root_dir = f"<data_root_dir>/R{RES}/"
                _, test_loaders[velo_dir][RES], data_processor0 = load_stokesbrinkman(
                    root_dir=root_dir,
                    n_train=700,
                    n_tests=[100],
                    batch_size=1,
                    test_batch_sizes=[1],
                    test_resolutions=[RES],
                    train_resolution=RES,
                    grid_boundaries=[[0, 1], [0, 1]],
                    positional_encoding=True,
                    encode_input=False,
                    encode_output=False,
                    encoding="channel-wise",
                    loading_cond="[0,1]",
                )
            data_processor0 = data_processor0.to(device)
        else:
            for RES in RESs:
                _, test_loaders[velo_dir][RES], data_processor1 = load_stokesbrinkman(
                    root_dir=root_dir,
                    n_train=700,
                    n_tests=[100],
                    batch_size=1,
                    test_batch_sizes=[1],
                    test_resolutions=[RES],
                    train_resolution=RES,
                    grid_boundaries=[[0, 1], [0, 1]],
                    positional_encoding=True,
                    encode_input=False,
                    encode_output=False,
                    encoding="channel-wise",
                    loading_cond="[1,0]",
                )
                data_processor1 = data_processor1.to(device)

    # Macro-permeability
    pressureProvided = False

    l2 = {}
    mue = 1
    mu = 1

    l2 = ({}, {})
    K11_true = []
    K22_true = []
    K12_true = []
    K21_true = []

    K11_pred = []
    K22_pred = []
    K12_pred = []
    K21_pred = []

    idx_RES = []
    idx_SAM = []

    for RES in RESs:
        l2[0][RES] = []
        l2[1][RES] = []
        i = 0
        for data1, data2 in zip(
            test_loaders[0][RES][RES].dataset, test_loaders[1][RES][RES].dataset
        ):
            idx_RES.append(RES)
            idx_SAM.append(i)
            i = i + 1
            # LOAD 1 and 2
            data1 = data_processor0.preprocess(data1, batched=False)
            data2 = data_processor1.preprocess(data2, batched=False)

            # unpack data
            x1, y1, out1 = unpack_data(
                model=model0,
                data=data1,
                data_processor=data_processor0,
                idir=0,
                encode_output=None,
            )
            x2, y2, out2 = unpack_data(
                model=model1,
                data=data2,
                data_processor=data_processor1,
                idir=0,
                encode_output=None,
            )
            data1["y"], data2["y"] = y1, y2
            data1["x"], data2["x"] = x1, x2

            # L2 difference norm
            l2_0 = torch.sqrt(torch.sum((y1 - out1[0]) ** 2))
            l2_1 = torch.sqrt(torch.sum((y2 - out2[0]) ** 2))
            l2_y1 = torch.sqrt(torch.sum((y1) ** 2))
            l2_y2 = torch.sqrt(torch.sum((y2) ** 2))
            l2[0][RES].append((l2_0 / l2_y1).item())
            l2[1][RES].append((l2_1 / l2_y2).item())
            print(
                f"RES {RES}, isample {i}, L2 norm: {l2[0][RES][-1]}, {l2[1][RES][-1]}"
            )

            # local pressure gradient field
            pgrad1_true, pgrad1_pred = calc_pressure_grad_2d(
                data1,
                out1,
                mu=1,
                mue=1,
                L=[0, 1],
                device=device,
                inputEncoded=True,
                pressureProvided=pressureProvided,
            )
            pgrad2_true, pgrad2_pred = calc_pressure_grad_2d(
                data2,
                out2,
                mu=1,
                mue=1,
                L=[1, 0],
                device=device,
                inputEncoded=True,
                pressureProvided=pressureProvided,
            )
            # darcys law
            H_true = torch.zeros((2, 2))
            H_true[:, 0] = (
                -torch.as_tensor(
                    [
                        torch.mean(pgrad1_true[:, i, :, :].flatten()).item()
                        for i in range(2)
                    ]
                )
                / mu
            )
            H_true[:, 1] = (
                -torch.as_tensor(
                    [
                        torch.mean(pgrad2_true[:, i, :, :].flatten()).item()
                        for i in range(2)
                    ]
                )
                / mu
            )

            H_pred = torch.zeros((2, 2))
            H_pred[:, 0] = (
                -torch.as_tensor(
                    [
                        torch.mean(pgrad1_pred[:, i, :, :].flatten()).item()
                        for i in range(2)
                    ]
                )
                / mu
            )
            H_pred[:, 1] = (
                -torch.as_tensor(
                    [
                        torch.mean(pgrad2_pred[:, i, :, :].flatten()).item()
                        for i in range(2)
                    ]
                )
                / mu
            )

            # macro permeability
            try:
                K_true = torch.linalg.solve(H_true, torch.eye(2))
            except RuntimeError as e:
                print("Skipping due to singular matrix", str(e))
            K_pred = torch.linalg.solve(H_pred, torch.eye(2))

            # register
            K11_true.append(abs(K_true[0, 0].cpu()))
            K22_true.append(abs(K_true[1, 1].cpu()))
            K12_true.append(abs(K_true[0, 1].cpu()))
            K21_true.append(abs(K_true[1, 0].cpu()))
            K11_pred.append(abs(K_pred[0, 0].cpu()))
            K22_pred.append(abs(K_pred[1, 1].cpu()))
            K12_pred.append(abs(K_pred[0, 1].cpu()))
            K21_pred.append(abs(K_pred[1, 0].cpu()))

    print(
        "Avg L2 norm: ",
        torch.mean(torch.tensor(l2[0][res])),
        torch.mean(torch.tensor(l2[1][res])),
    )

    ## compare K
    plot_Kcompar(
        K11_true,
        K11_pred,
        K12_true,
        K12_pred,
        K21_true,
        K21_pred,
        K22_true,
        K22_pred,
        save_path=save_path,
    )
    plt.show()
    return (
        K11_true,
        K11_pred,
        K12_true,
        K12_pred,
        K21_true,
        K21_pred,
        K22_true,
        K22_pred,
    )


def calc_macro_permeability_train(save_path=None):
    """
    Calculate macro-permeability for training data.

    Args:
        save_path: Optional path to save the results.
    """
    # load the data (two simulations per sample)
    RES = 64
    train_loaders = ({}, {})
    for velo_dir in range(2):
        root_dir = f"<data_root_dir>/R{RES}/"
        train_loaders[velo_dir][RES], _, data_processor = load_stokesbrinkman(
            root_dir=root_dir,
            n_train=700,
            n_tests=[100],
            batch_size=1,
            test_batch_sizes=[1],
            test_resolutions=[RES],
            train_resolution=RES,
            grid_boundaries=[[0, 1], [0, 1]],
            positional_encoding=True,
            encode_input=False,
            encode_output=False,
            encoding="channel-wise",
        )
        data_processor = data_processor.to(device)

    # Macro-permeability
    pressureProvided = False

    l2 = {}
    mue = 1
    mu = 1

    l2 = ({}, {})
    K11_true = []
    K22_true = []
    K12_true = []
    K21_true = []

    K11_pred = []
    K22_pred = []
    K12_pred = []
    K21_pred = []

    errPI1 = []
    errPI2 = []

    idx_RES = []
    idx_SAM = []

    l2[0][RES] = []
    l2[1][RES] = []
    i = 0
    for data1, data2 in zip(
        train_loaders[0][RES].dataset, train_loaders[1][RES].dataset
    ):
        idx_RES.append(RES)
        idx_SAM.append(i)
        i = i + 1
        # LOAD 1 and 2
        data1 = data_processor.preprocess(data1, batched=False)
        data2 = data_processor.preprocess(data2, batched=False)

        # unpack data
        x1, y1, out1 = unpack_data(data1, 0, encode_output=None)
        x2, y2, out2 = unpack_data(data2, 1, encode_output=None)
        data1["y"], data2["y"] = y1, y2

        # L2 difference norm
        l2[0][RES].append((torch.sqrt(torch.sum((y1 - out1[0]) ** 2)) / RES).item())
        l2[1][RES].append((torch.sqrt(torch.sum((y2 - out2[0]) ** 2)) / RES).item())

        # local pressure gradient field
        pgrad1_true, pgrad1_pred = calc_pressure_grad_2d(
            data1,
            out1,
            mu=1,
            mue=1,
            L=[1, 1],
            device=device,
            inputEncoded=True,
            pressureProvided=pressureProvided,
        )
        pgrad2_true, pgrad2_pred = calc_pressure_grad_2d(
            data2,
            out2,
            mu=1,
            mue=1,
            L=[1, 1],
            device=device,
            inputEncoded=True,
            pressureProvided=pressureProvided,
        )
        # darcys law
        H_true = torch.zeros((2, 2))
        H_true[:, 0] = (
            -torch.as_tensor([pgrad1_true[..., i].mean().item() for i in range(2)]) / mu
        )
        H_true[:, 1] = (
            -torch.as_tensor([pgrad2_true[..., i].mean().item() for i in range(2)]) / mu
        )

        H_pred = torch.zeros((2, 2))
        H_pred[:, 0] = (
            -torch.as_tensor([pgrad1_pred[..., i].mean().item() for i in range(2)]) / mu
        )
        H_pred[:, 1] = (
            -torch.as_tensor([pgrad2_pred[..., i].mean().item() for i in range(2)]) / mu
        )

        # macro permeability
        try:
            K_true = torch.linalg.solve(H_true, torch.eye(2, device=device))
        except RuntimeError as e:
            print("Skipping due to singular matrix", str(e))
        K_pred = torch.linalg.solve(H_pred, torch.eye(2, device=device))

        # register
        K11_true.append(abs(K_true[0, 0]))
        K22_true.append(abs(K_true[1, 1]))
        K12_true.append(abs(K_true[0, 1]))
        K21_true.append(abs(K_true[1, 0]))
        K11_pred.append(abs(K_pred[0, 0]))
        K22_pred.append(abs(K_pred[1, 1]))
        K12_pred.append(abs(K_pred[0, 1]))
        K21_pred.append(abs(K_pred[1, 0]))

    ## compare K
    plot_Kcompar(
        K11_true,
        K11_pred,
        K12_true,
        K12_pred,
        K21_true,
        K21_pred,
        K22_true,
        K22_pred,
        save_path=save_path,
    )
    plt.show()


def test_on_unseen_data(loading_cond):
    """
    Test the model on unseen data and visualize predictions.

    Args:
        loading_cond: Loading condition for the test data.
    """
    root_dir = "<data_root_dir>/R128"
    train_loader, test_loaders, data_processor = load_stokesbrinkman(
        root_dir=root_dir,
        n_train=1,
        n_tests=[50],
        batch_size=1,
        test_batch_sizes=[1],
        test_resolutions=[128],
        train_resolution=128,
        grid_boundaries=[[0, 1], [0, 1]],
        positional_encoding=True,
        encode_input=False,
        encode_output=False,
        encoding="channel-wise",
        loading_cond=loading_cond,
    )
    data_processor = data_processor.to(device)

    test_samples = test_loaders[128].dataset
    model.to(device)

    cm_colo = cm.ScalarMappable(norm=None, cmap="viridis")

    fig = plt.figure(figsize=(7, 7))
    for index in range(3):
        data = test_samples[index + 0]
        data = data_processor.preprocess(data, batched=False)

        # Model prediction
        out = model(data["x"].unsqueeze(0))
        out, data = data_processor.postprocess(out, data)
        out = out.cpu()

        # Input
        x = data["x"].cpu()
        # Ground truth
        y = data["y"].cpu()

        #
        ax = fig.add_subplot(3, 5, index * 5 + 1)
        im = ax.imshow(x[0], cmap="gray")
        if index == 0:
            ax.set_title("Input x")
        plt.xticks([], [])
        plt.yticks([], [])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        ax = fig.add_subplot(3, 5, index * 5 + 2)
        vmin = y[0].min()
        vmax = y[0].max()
        im = ax.imshow(y[0].squeeze(), cmap=cm_colo.get_cmap(), vmin=vmin, vmax=vmax)
        if index == 0:
            ax.set_title("Ground-truth y1")
        plt.xticks([], [])
        plt.yticks([], [])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        #
        ax = fig.add_subplot(3, 5, index * 5 + 3)
        im = ax.imshow(
            out[0, 0].squeeze().detach().numpy(),
            cmap=cm_colo.get_cmap(),
            vmin=vmin,
            vmax=vmax,
        )
        if index == 0:
            ax.set_title("Model prediction")
        plt.xticks([], [])
        plt.yticks([], [])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        #
        ax = fig.add_subplot(3, 5, index * 5 + 4)
        vmin = y[1].min()
        vmax = y[1].max()
        im = ax.imshow(y[1].squeeze(), cmap=cm_colo.get_cmap(), vmin=vmin, vmax=vmax)
        if index == 0:
            ax.set_title("Ground-truth y2")
        plt.xticks([], [])
        plt.yticks([], [])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        #
        ax = fig.add_subplot(3, 5, index * 5 + 5)
        im = ax.imshow(
            out[0, 1].squeeze().detach().numpy(),
            cmap=cm_colo.get_cmap(),
            vmin=vmin,
            vmax=vmax,
        )
        if index == 0:
            ax.set_title("Model prediction")
        plt.xticks([], [])
        plt.yticks([], [])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    fig.suptitle("Inputs, ground-truth output and prediction.", y=0.98)
    plt.tight_layout()
    plt.show()


# load the trained model
# Function to extract datetime from the filename
def extract_datetime(filename):
    datetime_str = filename.split("_")[0]  # Get the datetime part
    return datetime.datetime.strptime(datetime_str, "%Y-%m-%dT%H%M%SZ")


def plot_images(test_samples, data_processor, model):
    """
    Plot input, ground truth, and model predictions for test samples.

    Args:
        test_samples: Dataset of test samples.
        data_processor: Processor for data preprocessing and postprocessing.
        model: Trained model for predictions.
    """
    fig = plt.figure(figsize=(7, 7))
    for index in range(3):
        data = test_samples[index + 0]
        data = data_processor.preprocess(data, batched=False)

        # Model prediction
        out = model(data["x"].unsqueeze(0))
        out, data = data_processor.postprocess(out, data)
        out = out.cpu()

        # Input
        x = data["x"].cpu()
        # Ground truth
        y = data["y"].cpu()

        #
        ax = fig.add_subplot(3, 5, index * 5 + 1)
        ax.imshow(x[0], cmap="gray")
        if index == 0:
            ax.set_title("Input x")
        plt.xticks([], [])
        plt.yticks([], [])
        #
        ax = fig.add_subplot(3, 5, index * 5 + 2)
        ax.imshow(y[0].squeeze())
        if index == 0:
            ax.set_title("Ground-truth y1")
        plt.xticks([], [])
        plt.yticks([], [])
        #
        ax = fig.add_subplot(3, 5, index * 5 + 3)
        ax.imshow(out[0, 0].squeeze().detach().numpy())
        if index == 0:
            ax.set_title("Model prediction")
        plt.xticks([], [])
        plt.yticks([], [])
        #
        ax = fig.add_subplot(3, 5, index * 5 + 4)
        ax.imshow(y[1].squeeze())
        if index == 0:
            ax.set_title("Ground-truth y2")
        plt.xticks([], [])
        plt.yticks([], [])
        #
        ax = fig.add_subplot(3, 5, index * 5 + 5)
        ax.imshow(out[0, 1].squeeze().detach().numpy())
        if index == 0:
            ax.set_title("Model prediction")
        plt.xticks([], [])
        plt.yticks([], [])

    fig.suptitle("Inputs, ground-truth output and prediction.", y=0.98)
    plt.tight_layout()
    plt.savefig(f"<figures_dir>/{CHECKPOINT}_last.png")
    fig.show()


def compute_loss_per_sample(test_samples, loss, model, data_processor, device):
    """
    Compute the loss for each sample and find the one with the lowest loss.

    Args:
        test_samples: Dataset of test samples.
        loss: Loss function to compute the loss.
        model: Trained model for predictions.
        data_processor: Processor for data preprocessing and postprocessing.
        device: Device to perform computations on.

    Returns:
        Index of the sample with the lowest loss and its output.
    """
    model.eval()  # Set the model to evaluation mode
    model.to(device)

    lowest_loss = float("inf")
    lowest_loss_index = -1
    ll_out = torch.zeros((1, 2, 64, 64))

    for idx, sample in enumerate(test_samples):
        # Preprocess the sample
        data = data_processor.preprocess(sample, batched=False)
        input_tensor = data["x"].unsqueeze(0).to(device)
        target_tensor = data["y"].unsqueeze(0).to(device)

        # Get the model's prediction
        with torch.no_grad():
            output = model(input_tensor)

        # Compute the loss (Mean Squared Error)
        output, data = data_processor.postprocess(output, data)
        loss_value = loss(output, input_tensor, target_tensor)

        # Update the lowest loss and index
        if loss_value.item() < lowest_loss:
            lowest_loss = loss_value.item()
            lowest_loss_index = idx
            ll_out = output
    print(f"Lowest loss: {lowest_loss}, Index: {lowest_loss_index}")

    return lowest_loss_index, ll_out


# %%
def compute_l2_loss_funcs(
    model, FOLDER_DIR, model_checkpoint_arr, test_loader, data_processor
):
    """
    Compute L2 loss for model predictions across multiple checkpoints.

    Args:
        model: The model to evaluate.
        FOLDER_DIR: Directory containing model checkpoints.
        model_checkpoint_arr: List of checkpoint filenames.
        test_loader: DataLoader for test data.
        data_processor: Processor for data preprocessing and postprocessing.

    Returns:
        Array of L2 loss values for each checkpoint.
    """
    l2_metric = LpLoss(d=2, p=2, reduce_dims=[0, 1], reductions="mean")
    out_arr = torch.zeros(
        (len(model_checkpoint_arr) + 1, 100, 2, 64, 64), device=torch.device("cuda")
    )

    for i in range(len(model_checkpoint_arr) + 1):
        checkpoint = torch.load(
            FOLDER_DIR + f"/<checkpoints_dir>/{model_checkpoint_arr[i-1]}_last.pt"
        )
        model.load_state_dict(checkpoint["state_dict"])
        with torch.no_grad():
            model.eval()
            for j, data in enumerate(test_loader):
                data = data_processor.preprocess(data, batched=False)
                x = data["x"].unsqueeze(0).to(device)
                out = model(x)
                out, test_data = data_processor.postprocess(out, data)
                if i == 0:
                    out_arr[i, j] = test_data["y"].to(device)
                else:
                    out_arr[i, j] = out
    l2_loss_arr = torch.ones((out_arr.shape[0] - 1))
    u_gt = out_arr[0]
    for i in range(out_arr.shape[0] - 1):
        u_pred = out_arr[i + 1]
        l2 = l2_metric(u_gt, u_pred)
        l2_loss_arr[i] = l2

    return l2_loss_arr


# %%
def compute_sup_norm(K_pred, K_gt):
    """
    Compute the supremum norm between predicted and ground truth permeability tensors.

    Args:
        K_pred: Predicted macro permeability.
        K_gt: Ground truth macro permeability.

    Returns:
        Supremum norm value.
    """
    sup_norm = 0.0
    for i in range(K_pred.shape[0]):
        log_K_diff = torch.abs(torch.log(K_pred[i]) - torch.log(K_gt[i]))
        sup_norm_K_gt = torch.max(torch.abs(torch.log(K_gt[i])))
        sup_norm += torch.max(log_K_diff) / sup_norm_K_gt

    sup_norm /= K_pred.shape[0]

    return sup_norm


def compute_frobenius_norm(K_pred, K_gt):
    """
    Compute the Frobenius norm between predicted and ground truth macro permeability.

    Args:
        K_pred: Predicted macro permeability.
        K_gt: Ground truth macro permeability.
    Returns:
        Frobenius norm value.
    """
    fro_norm = 0.0
    for i in range(K_pred.shape[0]):
        diff_K = torch.abs(K_pred[i] - K_gt[i])
        fro_norm += torch.norm(diff_K, p="fro") / torch.norm(K_gt[i], p="fro")

    fro_norm /= K_pred.shape[0]

    return fro_norm


# %%
if __name__ == "__main__":
    entity = "crunkel-research"
    project = "DualScaleFlow"

    model_type = "fno"
    # model_type = "tfno"

    loading_cond = "[1,0]"
    # loading_cond = "[0,1]"

    root_dir = r"<data_root_dir>/R64"
    train_loader, test_loaders, data_processor = load_stokesbrinkman(
        root_dir=root_dir,
        n_train=700,
        n_tests=[100],
        batch_size=1,
        test_batch_sizes=[1],
        test_resolutions=[64],
        train_resolution=64,
        grid_boundaries=[[0, 1], [0, 1]],
        positional_encoding=True,
        encode_input=False,
        encode_output=False,
        encoding="channel-wise",
        loading_cond=loading_cond,
    )
    data_processor = data_processor.to(device)
    # create a tensorised FNO model
    if model_type == "tfno":
        model = TFNO(
            n_modes=(64, 64),
            in_channels=3,
            out_channels=2,
            hidden_channels=32,
            projection_channels=64,
            factorization="tucker",
            rank=0.42,
        )
    # create an FNO model
    elif model_type == "fno":
        model = FNO(
            n_modes=(64, 64),
            in_channels=3,
            out_channels=2,
            hidden_channels=32,
            projection_channels=64,
        )
    else:
        raise ValueError(f"Model type {model_type} not recognised.")
    model = model.to(device)
    save_dir = "<checkpoints_dir>"

    trainer = Trainer(
        model=model,
        n_epochs=1,
        device=device,
        callbacks=[
            CheckpointCallback(
                save_dir=save_dir,
                save_best=False,
                save_interval=10,
                save_optimizer=True,
                save_scheduler=True,
            ),
            BasicLoggerCallback(),
        ],
        data_processor=data_processor,
        wandb_log=True,
        log_test_interval=3,
        use_distributed=False,
        verbose=True,
    )

    FOLDER_DIR = "<project_root_dir>"
    if loading_cond == "[1,0]":
        # TFNO checkpoints loading cond U=[1,0]
        if model_type == "tfno":
            checkpoint_arr = [
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
            ]
        elif model_type == "fno":
            # FNO checkpoints loading cond U=[1,0]
            checkpoint_arr = [
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
            ]
    elif loading_cond == "[0,1]":
        # TFNO checkpoints loading cond U=[0, 1]
        if model_type == "tfno":
            checkpoint_arr = [
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
            ]
        elif model_type == "fno":
            # FNO checkpoints loading cond U=[0, 1]
            checkpoint_arr = [
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
                "<timestamp_placeholder>",
            ]

    l2_loss_arr = compute_l2_loss_funcs(
        model, FOLDER_DIR, checkpoint_arr, test_loaders[64].dataset, data_processor
    )
    print(l2_loss_arr)


# %%
if __name__ == "__main__":
    # Replace WandB initialization with placeholders
    api = wandb.Api()
    run = wandb.init(entity="<wandb_entity>", project="<wandb_project>", id="<wandb_run_id>", resume="None")

    # Access the config of the run
    config = run.config

    loading_cond0 = "[0,1]"
    loading_cond1 = "[1,0]"
    RES = 64
    # Define placeholders for checkpoints
    CHECKPOINT0 = "<checkpoint_0_placeholder>"
    CHECKPOINT1 = "<checkpoint_1_placeholder>"

    # load the data
    root_dir = r"<data_root_dir>/R64"
    train_loader1, test_loaders1, data_processor1 = load_stokesbrinkman(
        root_dir=root_dir,
        n_train=config.n_train,
        n_tests=config.n_tests,
        batch_size=config.batch_size,
        test_batch_sizes=config.test_batch_sizes,
        test_resolutions=config.test_resolutions,
        train_resolution=config.train_resolution,
        grid_boundaries=[[0, 1], [0, 1]],
        positional_encoding=config.positional_encoding,
        encode_input=config.encode_input,
        encode_output=config.encode_output,
        encoding=config.encoding,
        loading_cond=loading_cond1,
    )
    data_processor1 = data_processor1.to(device)
    train_loader0, test_loaders0, data_processor0 = load_stokesbrinkman(
        root_dir=root_dir,
        n_train=config.n_train,
        n_tests=config.n_tests,
        batch_size=config.batch_size,
        test_batch_sizes=config.test_batch_sizes,
        test_resolutions=config.test_resolutions,
        train_resolution=config.train_resolution,
        grid_boundaries=[[0, 1], [0, 1]],
        positional_encoding=config.positional_encoding,
        encode_input=config.encode_input,
        encode_output=config.encode_output,
        encoding=config.encoding,
        loading_cond=loading_cond0,
    )
    data_processor0 = data_processor0.to(device)
    # create a tensorised FNO model
    if config.model_type == "tfno":
        model = TFNO(
            n_modes=(64, 64),
            in_channels=3,
            out_channels=2,
            hidden_channels=32,
            projection_channels=64,
            factorization="tucker",
            rank=0.42,
        )
    # create an FNO model
    elif config.model_type == "fno":
        model = FNO(
            n_modes=(64, 64),
            in_channels=3,
            out_channels=2,
            hidden_channels=32,
            projection_channels=64,
        )
    else:
        raise ValueError(f"Model type {config.model_type} not recognised.")
    model0 = model.to(device)

    model1 = copy.deepcopy(model0)

    FOLDER_DIR = "<project_root_dir>"

    SAVE_PATH = os.path.join(
        FOLDER_DIR, f"figures/K_{config.model_type}_{CHECKPOINT1}_{RES}n.png"
    )
    checkpoint1 = torch.load(FOLDER_DIR + f"/<checkpoints_dir>/{CHECKPOINT1}_last.pt")
    checkpoint0 = torch.load(FOLDER_DIR + f"/<checkpoints_dir>/{CHECKPOINT0}_last.pt")
    model1.load_state_dict(checkpoint1["state_dict"])
    model0.load_state_dict(checkpoint0["state_dict"])
    print(config.model_type)

    # Plot the prediction, and compare with the ground-truth

    model1.to(device)
    model0.to(device)

    (
        K11_true,
        K11_pred,
        K12_true,
        K12_pred,
        K21_true,
        K21_pred,
        K22_true,
        K22_pred,
    ) = calc_macro_permeability(
        model0=model0, model1=model1, res=RES, save_path=SAVE_PATH
    )
    K_pred = torch.tensor([[K11_pred, K12_pred], [K21_pred, K22_pred]]).permute(2, 0, 1)
    K_true = torch.tensor([[K11_true, K12_true], [K21_true, K22_true]]).permute(2, 0, 1)

    sup_norm = compute_sup_norm(K_pred, K_true)
    print(f"Sup norm: {sup_norm:.4f}")
    fro_norm = compute_frobenius_norm(K_pred, K_true)
    print(f"Fro norm: {fro_norm:.4f}")

    run.finish()


# %%
def plot_beta():
    root_dir = r"<data_root_dir>/R64"
    train_loader, test_loaders, data_processor = load_stokesbrinkman(
        root_dir=root_dir,
        n_train=700,
        n_tests=[100],
        batch_size=16,
        test_batch_sizes=[1],
        test_resolutions=[64],
        train_resolution=64,
        grid_boundaries=[[0, 1], [0, 1]],
        positional_encoding=True,
        encode_input=False,
        encode_output=False,
        encoding="channel-wise",
    )
    beta_test = torch.zeros((100, 4096))
    beta_train = torch.zeros((700, 4096))
    for i, data in enumerate(test_loaders[64]):
        # plot histogram of beta values
        beta_test[i] = data["x"][0].flatten()
    plt.rcParams["font.size"] = "12"
    plt.hist(beta_test.flatten(), bins=100, log=True, density=True)
    plt.xscale("log")
    plt.gca().xaxis.set_major_locator(LogLocator(base=10.0))
    plt.xlabel(r"Distribution of $\log_{10}(\beta)$ values in test dataset.")
    plt.savefig("<figures_dir>/beta_test.png", dpi=300)
    plt.show()
    print(beta_test.min(), beta_test.max())

    for j, data in enumerate(train_loader.dataset):
        # plot histogram of beta values
        beta_train[j] = data["x"][0].flatten()
    plt.hist(beta_train.flatten(), bins=100, log=True, density=True)
    plt.xscale("log")
    plt.gca().xaxis.set_major_locator(LogLocator(base=10.0))
    plt.xlabel(r"Distribution of $\log_{10}(\beta)$ values in training dataset.")
    plt.savefig("<figures_dir>/beta_train.png", dpi=300)
    plt.show()
    print(beta_train.min(), beta_train.max())


# %%
plot_beta()
# %%
