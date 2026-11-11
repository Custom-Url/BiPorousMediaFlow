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
32 hidden channels, Tucker rank 0.42), using the input and output encoding
flags from `config/BPMF_config.yaml`. The sweep varies learning rate, weight
decay, StepLR decay and separate pressure-gradient loss weights, with 100 epochs per run. Edit
`config/permeability_sweep.yaml` to change the search space. Each run validates
both models together every 10 epochs and at the final epoch; change the
interval with `--validation-interval`. Validation uses normalizers fitted on
the entire training dataset, not the validation dataset.

The sweep objective, `permeability/best_relative_log_supremum`, is the lowest
validation mean of the per-sample relative log-supremum error
`max(abs(log(abs(K_pred))-log(abs(K_true)))) / max(abs(log(abs(K_true))))`
seen in a run. This matches the target paper's definition using entrywise
permeability magnitudes and therefore does **not** measure sign errors in K.
Since `log(0)` is undefined, entries at or below `1e-12` times the largest
absolute entry of either tensor are floored to that relative threshold for
this metric only. Matching structural zeros produce zero error; zero/nonzero
mismatches are penalized. W&B logs how many entries hit the floor so ARIA can
identify when this convention materially affects a run.
The W&B history also includes median and 90th-percentile log-supremum errors,
the relative Frobenius error, and per-direction velocity relative L2 errors.
K is reconstructed using both directions and a unit-square domain.
Non-finite or singular permeability results fail the run rather than being
silently omitted.

Training minimizes `H1 + field_weight * field_error + mean_weight * mean_error`
independently for each direction. H1 uses normalized outputs as before. The
auxiliary losses use decoded physical velocities, raw log-beta inputs and
`g = laplacian(u) - beta*u`, with exactly the same unit-square spacing and
boundary flags as permeability evaluation. In neuraloperator 2.0 those flags
select periodic finite differences; confirm the installed version's behavior.
No matrix inverse is used in the loss.

The default `stable` normalization uses a training-set-only scale `s`: the
median over training samples of `||g_true||₂ / sqrt(H*W)`, separately per
direction. With `r = ||g_true||₂ / sqrt(H*W)`, `m = ||mean(g_true)||₂`, and
`f = max(1e-12, 0.01*s)`, the denominators are `max(r, f)` for the field RMS
error and `max(m, 0.1*r, f)` for the mean-vector error. These floors limit
amplification caused by tiny references or cancellation; the constants are
starting choices, not validated optimal values. Field norms combine both
components and grid points, while the mean norm combines both components.
Each term sums over samples in a batch. H1's default reduction in
neuraloperator 2.0 sums per-sample **and per-component** relative H1 errors.
Zero targets therefore have finite auxiliary loss, but do not imply that K is
well-defined. Non-finite gradients or losses fail explicitly.

The new sweep tries separate weights `0`, `0.001`, and `0.01`. Setting both to
zero bypasses the physics calculation completely and reproduces the original
H1-only path. `legacy` normalization retains the earlier unfloored relative
field and mean norms and rejects zero reference norms. Existing W&B sweeps
using `pressure_gradient_weight` automatically retain `legacy` behavior with
that weight applied to both terms; start a new sweep to use the new parameters.

W&B records unweighted and weighted terms, total loss, per-direction training
reference scales, denominator minimum/median/maximum and fraction of mean
denominators clamped. Every `--gradient-interval` epochs (default 10) and at
epoch 1, it records each enabled term's raw and weighted gradient norms and
pairwise cosine similarities over all trainable parameters for the first batch
of each direction. Cosines are omitted when a norm is zero, with a corresponding
`grad_cosine_defined_*` flag. Gradient diagnostics are not computed when both
weights are zero. Epoch losses are divided by the number of samples; diagnostic
gradient norms are for that one batch's summed losses. These costs can increase
training time and memory. The physics loss is a proxy, not a direct K loss;
whether it improves permeability must be measured.

For a controlled comparison with the original winner `z6a3nv6k`, use its
**trusted** `best_pair.pt` as the reference. The command checks the recorded
optimizer/architecture settings, seed, encoding flags and both sample-ID hashes,
then starts one new W&B run with the winner's exact recorded optimizer settings,
100 epochs, seed 0 and matching validation interval. It writes to a new run-ID
directory under `save_dir/permeability_comparison/` and refuses to overwrite an
existing `best_pair.pt`. Run the H1 baseline first:

```bash
env -u PYTHONPATH -u PYTHONHOME /mnt/data/dg765/conda_envs/bpmf/bin/python \
  code/permeability_sweep.py --data-root /mnt/data2/yc2634/ML/StokesBrinkman \
  --entity dylangray --compare-baseline save_dir/permeability_sweep/z6a3nv6k/best_pair.pt \
  --field-weight 0 --mean-weight 0
```

Once that run finishes, repeat the same command with `--field-weight 0.001
--mean-weight 0.001`, then with `--field-weight 0.01 --mean-weight 0.01`.
To reproduce the earlier pressure-loss formula instead, add
`--pressure-normalization legacy` and set equal weights. Use the same software,
hardware and dataset contents as the winner when possible: ID hashes alone do
not verify file contents, and matching seeds does not guarantee bitwise
reproducibility across environments. These are separate W&B runs, not a W&B
sweep. Validation metrics and best-checkpoint selection are unchanged.

The beta convention shared with evaluation represents both zero beta and
beta=1 as encoded zero, and does not correctly decode positive beta below 1.
Verify that material values avoid those ambiguous cases before interpreting
the physics loss or permeability metric; this change does not alter that
existing evaluation convention.

Each run saves its best *pair* of model weights and their fitted normalizers
to `save_dir/permeability_sweep/<run-id>/best_pair.pt` and uploads it as a
W&B model artifact with the `best` alias. Checkpoints contain the selected
epoch, score, directions, architecture and run configuration. To reproduce
the validation score, use the same dataset and environment:

```bash
python code/permeability_sweep.py --data-root /path/to/data --evaluate save_dir/permeability_sweep/RUN_ID/best_pair.pt
```

Add `--plots` to save a `K_comparison.png` scatter plot for all validation
samples and velocity-field comparisons for the first three samples in both
loading directions. For example:

```bash
python code/permeability_sweep.py --data-root /path/to/data \
  --evaluate save_dir/permeability_sweep/RUN_ID/best_pair.pt --plots
```

The figures are written to `plots/` beside the checkpoint. Use `--plot-dir`
to choose another output directory and `--plot-samples N` to change the number
of field samples shown. The plots and scores use the checkpoint's saved
normalizers and the same `R64_unseen` validation IDs used for selection.
K plots show magnitudes, not signs; exact-zero pairs cannot be shown on a log
axis and are counted in each panel. Those values still contribute to the
reported error metric using its documented floor. Field comparison colors use
the ground-truth range for both ground truth and prediction.

Load only trusted checkpoints: PyTorch checkpoint loading can execute code.
The paired checkpoint format is separate from the legacy `evaluate.py`
single-model format. Sample-ID hashes check dataset membership, not file
contents. The validation set has been used for model selection, so report
final performance on a further independent test set.

For ARIA, W&B receives the objective, per-direction relative L2 errors, training
H1 and pressure-gradient losses, their weight, learning rate, best epoch, dataset counts and ID hashes, encoding
settings, hyperparameters, and the paired checkpoint artifact. Open the sweep in
W&B and ask ARIA to compare runs by `permeability/best_relative_log_supremum`,
including its median/p90, the Frobenius diagnostic, the validation L2 metrics,
and the pressure-gradient weight and learning-rate/weight-decay settings.

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
