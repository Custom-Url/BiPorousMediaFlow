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

### Paired permeability sweep

`code/permeability_sweep.py` trains a pair of TFNOs in each W&B run: one for
`[0,1]` and one for `[1,0]`. It uses every sample in `R64` for training and
every sample in the separate `R64_unseen` directory for model selection.
Ensure these datasets are genuinely disjoint: the script cannot establish
whether files in different directories represent the same simulations.
Both directories must have the `mesh`, `mate`, and `resu` layout expected by
`load_stokesbrinkman`.

In the existing BPMF environment, authenticate with `wandb login` and run:

```bash
python code/permeability_sweep.py --data-root /path/to/data --entity YOUR_WANDB_ENTITY
```

The root must contain both `R64` and `R64_unseen`. `BPMF_DATA_ROOT` can be
used instead of `--data-root`; `--project` defaults to `BPMF`. The process
creates a Bayesian sweep, repeatedly runs one agent job at a time, and stops
starting jobs after 12 hours from process start. A job already in progress
finishes its current epoch, validates, saves and uploads its best checkpoint, so
the total time can exceed 12 hours. `--hours` changes the time budget and
`--sweep-id` resumes an existing sweep with the same project and entity.
This timer is local to the process; it is not a W&B sweep-wide limit across
multiple agents.

The base model follows the current TFNO training configuration (32 modes,
32 hidden channels, Tucker rank 0.42), with H1 training loss and the input and
output encoding flags from `config/BPMF_config.yaml`. The sweep varies learning
rate, weight decay and StepLR decay, with 100 epochs per run. Edit
`config/permeability_sweep.yaml` to change the search space. Each run validates
both models together every 10 epochs and at the final epoch; change the
interval with `--validation-interval`. Validation uses normalizers fitted on
the entire training dataset, not the validation dataset.

The sweep objective, `permeability/best_relative_frobenius`, is the lowest
validation mean of `||abs(K_pred)-abs(K_true)||_F / ||abs(K_true)||_F` seen
in a run. It follows the existing evaluator's use of entrywise magnitudes,
so it does **not** measure sign errors in K. K is reconstructed using both
directions and a unit-square domain. Non-finite or singular permeability
results fail the run rather than being silently omitted. Training still
optimizes H1 loss; permeability is used for model selection, not as a
training loss.

Each run saves its best *pair* of model weights and their fitted normalizers
to `save_dir/permeability_sweep/<run-id>/best_pair.pt` and uploads it as a
W&B model artifact with the `best` alias. Checkpoints contain the selected
epoch, score, directions, architecture and run configuration. To reproduce
the validation score, use the same dataset and environment:

```bash
python code/permeability_sweep.py --data-root /path/to/data --evaluate save_dir/permeability_sweep/RUN_ID/best_pair.pt
```

Load only trusted checkpoints: PyTorch checkpoint loading can execute code.
The paired checkpoint format is separate from the legacy `evaluate.py`
single-model format. Sample-ID hashes check dataset membership, not file
contents. The validation set has been used for model selection, so report
final performance on a further independent test set.

For ARIA, W&B receives the objective, per-direction relative L2 errors, training
H1 losses, learning rate, best epoch, dataset counts and ID hashes, encoding
settings, hyperparameters, and the paired checkpoint artifact. Open the sweep in
W&B and ask ARIA to compare runs by `permeability/best_relative_frobenius`,
including the validation L2 metrics and learning-rate/weight-decay settings.

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
