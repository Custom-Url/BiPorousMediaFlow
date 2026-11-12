import os

from utils.IOfcts import vtkFieldReader, vtiFieldReader, extract_mat

import torch

import numpy as np
import glob

from neuralop.data.datasets.tensor_dataset import TensorDataset

from neuralop.layers.embeddings import GridEmbedding2D
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.data.transforms.data_processors import DefaultDataProcessor

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

    ############################################################################################# 
    print('y', y.shape)

    
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
            data1 = _preprocess_sample(data_processor0, data1)
            data2 = _preprocess_sample(data_processor1, data2)

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
            pgrad1_true, _ = calc_pressure_grad_2d(
                data1,
                _,
                mu=1,
                mue=1,
                L=[1, 1],
                device=device,
                inputEncoded=True,
                pressureProvided=pressureProvided,
            )
            pgrad2_true, _ = calc_pressure_grad_2d(
                data2,
                _,
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


            # macro permeability
            try:
                K_true = torch.linalg.solve(H_true, torch.eye(2))
            except RuntimeError as e:
                print("Skipping due to singular matrix", str(e))

            # register
            K11_true.append(abs(K_true[0, 0].cpu()))
            K22_true.append(abs(K_true[1, 1].cpu()))
            K12_true.append(abs(K_true[0, 1].cpu()))
            K21_true.append(abs(K_true[1, 0].cpu()))


    ############################################################################################# 


    indices = torch.randperm(len(x))
    idx_train = indices[:n_train]
    n_test = n_tests  # currently, only 1 resolution possible
    idx_test = indices[n_train : n_train + n_test]  # Randomly picking train and test

    if loading_cond == "[1,0]":
        x_train = x[idx_train, :, :, :, 0].float().clone()
        y_train = y[idx_train, :, :, :, 0].float().clone()

        x_test = x[idx_test, :, :, :, 0].float().clone()
        y_test = y[idx_test, :, :, :, 0].float().clone()

    elif loading_cond == "[0,1]":
        x_train = x[idx_train, :, :, :, 0].float().clone()
        y_train = y[idx_train, :, :, :, 0].float().clone()

        x_test = x[idx_test, :, :, :, 0].float().clone()
        y_test = y[idx_test, :, :, :, 0].float().clone()

        x_train = torch.rot90(x_train, k=1, dims=[-2, -1])  # 90° CCW
        x_test  = torch.rot90(x_test, k=1, dims=[-2, -1])

        y_train = torch.rot90(y_train, k=1, dims=[-2, -1])
        y_test = torch.rot90(y_test, k=1, dims=[-2, -1])

        # Step 2: Rotate vector components: (u_x, u_y) → (-u_y, u_x)
        u_x_train, u_y_train = y_train[:, 0], y_train[:, 1]
        u_x_test,  u_y_test  = y_test[:, 0], y_test[:, 1]

        u_x_train_rot = -u_y_train
        u_y_train_rot =  u_x_train
        u_x_test_rot  = -u_y_test
        u_y_test_rot  =  u_x_test

        # Reassemble rotated vector field
        y_train = torch.stack([u_x_train_rot, u_y_train_rot], dim=1)
        y_test  = torch.stack([u_x_test_rot,  u_y_test_rot], dim=1)


    elif loading_cond == "both":
        x_train = x[idx_train, :, :, :, 0].float().clone()
        y_train = y[idx_train, :, :, :, 0].float().clone()
        x_train_transposed = torch.transpose(x_train, -2, -1)
        x_train = torch.cat([x_train, x_train_transposed], dim=0)
        y_train_transposed = torch.transpose(y_train, -2, -1)
        y_train = torch.cat([y_train, y_train_transposed], dim=0)

        x_test = x[idx_test, :, :, :, 0].float().clone()
        y_test = y[idx_test, :, :, :, 0].float().clone()
        x_test_transposed = torch.transpose(x_test, -2, -1)
        x_test = torch.cat([x_test, x_test_transposed], dim=0)
        y_test_transposed = torch.transpose(y_test, -2, -1)
        y_test = torch.cat([y_test, y_test_transposed], dim=0)
    else:
        raise ValueError("Invalid loading condition")
    test_batch_size = test_batch_sizes[0]
    test_resolution = test_resolutions[0]


    print('x_train', x_train.shape)
    print('y_train', y_train.shape)
    print('x_test', x_test.shape)
    print('y_test', y_test.shape)

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
