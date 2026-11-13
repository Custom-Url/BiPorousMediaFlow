import os

from utils.IOfcts import vtkFieldReader, vtiFieldReader, extract_mat

import torch

import numpy as np
import glob

from neuralop.data.datasets.tensor_dataset import TensorDataset

from neuralop.layers.embeddings import GridEmbedding2D
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.data.transforms.data_processors import DefaultDataProcessor

import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)


def load_betamap(dir0, idname):
    """
    Loads the heterogeneous coefficient beta.
    Args:
        dir0 (str): Root directory containing the data.
        idname (str): Identifier name for the sample.
    Returns:
        torch.Tensor: Betamap tensor for the sample.
    """
    # matID & zone ID
    matID = os.path.join(dir0, "mesh", "mID_" + idname + ".vtk")
    matID, orig, spac = vtkFieldReader(matID, "matID")

    # local field beta (mu/ks)
    matxml = os.path.join(dir0, "mate", "mate_" + idname + ".xml")
    beta = extract_mat(matxml, mID=1, idx=[1, 2, 3], relativePath=True)

    # map of local field beta
    betamap = torch.zeros(matID.shape, dtype=beta.dtype)
    betamap[matID == 1] = beta[:, 0]  # isotropic

    return betamap


def load_velomap(dir0, idname, prefix="vmacro", velo_dir=1):
    """
    Loads the velocity field.
    Args:
        dir0 (str): Root directory containing the data.
        idname (str): Identifier name for the sample.
        prefix (str): Prefix for the velocity file name.
        velo_dir (int): Direction of velocity to load.
    Returns:
        torch.Tensor: Velocity map tensor for the sample.
    """
    vti_name = (
        dir0
        + "/resu/resu_"
        + idname
        + "/"
        + prefix
        + "_dir"
        + str(velo_dir)
        + "_velocity.vti"
    )
    return torch.tensor(np.array(vtiFieldReader(vti_name, components=[0, 1])))


def load_stokesbrinkman(
    root_dir,
    n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    test_resolutions=[256],
    train_resolution=256,
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=False,
    encode_input=False,
    encode_output=False,
    encoding="channel-wise",
    loading_cond="[1,0]",
):
    """
    Loads the Stokes-Brinkman dataset, splits into train/test, applies encoding and returns data loaders and processor.
    Args:
        root_dir (str): Root directory containing the dataset.
        n_train (int): Number of training samples.
        n_tests (list): List of number of test samples per resolution.
        batch_size (int): Batch size for training.
        test_batch_sizes (list): List of batch sizes for testing.
        test_resolutions (list): List of test resolutions.
        train_resolution (int): Resolution for training.
        grid_boundaries (list): Boundaries for positional encoding.
        positional_encoding (bool): Whether to use positional encoding.
        encode_input (bool): Whether to encode input data.
        encode_output (bool): Whether to encode output data.
        encoding (str): Encoding type.
        loading_cond (str): Loading condition for data orientation.
    Returns:
        tuple: (train_loader, test_loaders, data_processor)
    """
    torch.manual_seed(0)
    # Load the whole dataset
    idnames = glob.glob(os.path.join(root_dir, "mesh", "*.vtk"))
    idnames = [os.path.basename(idname)[4:-4] for idname in idnames]

    x = torch.stack([load_betamap(root_dir, idname) for idname in idnames])
    x = x.unsqueeze(1)  # BxCxWxHxD
    x[x > 0] = torch.log10(x[x > 0])  # log scale
    y = torch.stack([load_velomap(root_dir, idname, velo_dir=1) for idname in idnames])

    indices = torch.randperm(len(x))
    idx_train = indices[:n_train]
    n_test = n_tests  # currently, only 1 resolution possible
    idx_test = indices[n_train : n_train + n_test]  # Randomly picking train and test

    ##########################################
    # Defining x and y for loading cond = 1,0
    ##########################################
    x_train10 = x[idx_train, :, :, :, 0].float().clone()
    y_train10 = y[idx_train, :, :, :, 0].float().clone()

    x_test10 = x[idx_test, :, :, :, 0].float().clone()
    y_test10 = y[idx_test, :, :, :, 0].float().clone()

    
    ##########################################
    # Defining x and y for loading cond = 0,1
    ##########################################
    x_train01 = x[idx_train, :, :, :, 0].float().clone()
    y_train01 = y[idx_train, :, :, :, 0].float().clone()

    x_test01 = x[idx_test, :, :, :, 0].float().clone()
    y_test01 = y[idx_test, :, :, :, 0].float().clone()

    x_train01 = torch.rot90(x_train01, k=1, dims=[-2, -1])  # 90° CCW
    x_test01  = torch.rot90(x_test01, k=1, dims=[-2, -1])

    y_train01 = torch.rot90(y_train01, k=1, dims=[-2, -1])
    y_test01 = torch.rot90(y_test01, k=1, dims=[-2, -1])

    # Step 2: Rotate vector components: (u_x, u_y) → (-u_y, u_x)
    u_x_train01, u_y_train01 = y_train01[:, 0], y_train01[:, 1]
    u_x_test01,  u_y_test01  = y_test01[:, 0], y_test01[:, 1]

    u_x_train_rot01 = -u_y_train01
    u_y_train_rot01 =  u_x_train01
    u_x_test_rot01  = -u_y_test01
    u_y_test_rot01  =  u_x_test01

    # Reassemble rotated vector field
    y_train01 = torch.stack([u_x_train_rot01, u_y_train_rot01], dim=1)
    y_test01  = torch.stack([u_x_test_rot01,  u_y_test_rot01], dim=1)

    test_batch_size = test_batch_sizes[0]
    test_resolution = test_resolutions[0]

    # Plotting x and y for both loading conditions
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(x_train10[0, 0].numpy(), cmap='viridis')
    axs[0, 0].set_title('x_train (loading_cond [1,0])')
    axs[0, 1].imshow(y_train10[0, 0].numpy(), cmap='viridis')
    axs[0, 1].set_title('y_train (loading_cond [1,0])')
    axs[1, 0].imshow(x_train01[0, 0].numpy(), cmap='viridis')
    axs[1, 0].set_title('x_train (loading_cond [0,1])')
    axs[1, 1].imshow(y_train01[0, 0].numpy(), cmap='viridis')
    axs[1, 1].set_title('y_train (loading_cond [0,1])')
    plt.tight_layout()
    plt.show()


    # Input encoding
    if encode_input:
        # Assuming encoding is channel-wise for simplicity
        input_encoder = UnitGaussianNormalizer(dim=list(range(x_train.ndim)))
        input_encoder.fit(x_train)
    else:
        input_encoder = None

    # Output encoding
    if encode_output:
        # Assuming encoding is channel-wise for simplicity
        output_encoder = UnitGaussianNormalizer(dim=list(range(y_train.ndim)))
        output_encoder.fit(y_train)
    else:
        output_encoder = None

    # Training dataset
    train_db = TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    # Test dataset
    test_db = TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_db,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    test_loaders = {train_resolution: test_loader}

    # Positional encoding
    if positional_encoding:
        pos_encoding = GridEmbedding2D(in_channels=1)
    else:
        pos_encoding = None
    data_processor = DefaultDataProcessor(
        in_normalizer=input_encoder,
        out_normalizer=output_encoder
    )

    return train_loader, test_loaders, data_processor
