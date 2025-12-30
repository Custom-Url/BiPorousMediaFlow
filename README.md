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
