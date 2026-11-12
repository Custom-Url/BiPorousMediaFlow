# Operator learning for bi-porous media flow problem with extreme permeability contrast

This is the official repository of the "Operator learning regularization for macroscopic permeability prediction in dual-scale flow problem" paper (https://arxiv.org/abs/2412.00579). The project includes the custom loss functions, training and evaluation scripts, and utilities for handling data and domain-specific operations.

## Features
- Implementation of custom loss functions
- Training and evaluation scripts for FNO and TFNO models
- Utilities for data I/O, domain-specific operations, and helper functions
- Configurable hyperparameter sweeps via YAML files

## Project Structure
```
LICENSE.txt
README.md
script_lr_FNO_custom_loss.sh
script_lr_FNO_standard_loss.sh
script_lr_TFNO_custom_loss.sh
script_lr_TFNO_standard_loss.sh
sweep_config_lambda.yaml
sweep_config.yaml
code/
    custom_loss.py
    evaluate.py
    train.py
    utils/
        brinkman_amitex.py
        classes.py
        custom_trainer.py
        helpers.py
        IOfcts.py
```

## Getting Started

### Prerequisites
- Conda (Anaconda or Miniconda)
- Python 3.8+ (managed by the environment)

### Installation
1. Clone the repository:
  ```bash
  git clone <repo-url>
  cd BiPorousMediaFlow
  ```
2. Create and activate the conda environment:
  ```bash
  conda env create -f environment.yaml
  conda activate bmf
  ```
  *(This will install all required dependencies as specified in environment.yaml)*

### Usage
- **Training:**
  ```bash
  python code/train.py --config sweep_config.yaml
  ```
- **Evaluation:**
  ```bash
  python code/evaluate.py --model <model_path>
  ```
- **Custom Loss:**
  Custom loss functions are defined in `code/custom_loss.py` and can be configured in the training script.

### Experimental permeability-oriented target

`experimental/train.py` uses `experimental/brinkman_amitex.py` to train on
two-channel, 2D fields `z = (beta * u - Laplacian(u)) / mu`, rather than
velocity. The loader applies this transformation after orienting each sample,
before output normalization, with `mu=1` and a unit domain. It follows
`code/evaluate.py`'s `compute_K` convention: beta is `10**x`, where `x` is
the loader's log-beta input, including `x=0`. This is not the same as the
`calc_pressure_grad_2d` helper's current `x**10` expression.

For two independent, unit-flux loading conditions on the **same material**,
aligned in the same coordinate system and ordered along the two coordinate
axes, the spatial means of their decoded latent fields are the columns of
`H`; macro permeability is `K = H^-1`. Use
`macro_permeability_from_latent(z1, z2)` with tensors shaped `(B, 2, H, W)`.
It raises an error if `H` is singular. One loading condition alone cannot
recover the full tensor. Rotating a sample as a training augmentation does
not by itself supply a second loading for the original material.

If `encode_output` is enabled, decode model outputs using the fitted
`data_processor` before calculating permeability. Experimental checkpoints
save the input and output normalizers' `mean`, `std`, and `eps` under
`normalizers`; recreate each enabled `UnitGaussianNormalizer` with those
values to preprocess inputs and decode outputs when reloading. Velocity-based
evaluation and checkpoints trained on velocity are not interchangeable with
these latent targets. The transformation is intended to emphasize the quantity
used in permeability calculation; improved learning accuracy is not guaranteed.

- **Shell Scripts:**
  Use the provided `.sh` scripts to run training with different settings:
  - `script_lr_FNO_custom_loss.sh`
  - `script_lr_FNO_standard_loss.sh`
  - `script_lr_TFNO_custom_loss.sh`
  - `script_lr_TFNO_standard_loss.sh`

### Configuration
- Hyperparameter sweeps and experiment settings are defined in `sweep_config.yaml` and `sweep_config_lambda.yaml`.

## License
See [LICENSE.txt](LICENSE.txt) for license information.

## Citation
If you use this code for your research, please cite the corresponding publication:
```
@article{runkel2024operator,
  title={Operator learning regularization for macroscopic permeability prediction in dual-scale flow problem},
  author={Runkel, Christina and Xiao, Sinan and Boull{\'e}, Nicolas and Chen, Yang},
  journal={arXiv preprint arXiv:2412.00579},
  year={2024}
}
```

## Contact
For questions or contributions, please contact the project maintainer.
