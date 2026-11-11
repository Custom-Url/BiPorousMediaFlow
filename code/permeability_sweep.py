"""Paired TFNO sweep ranked by held-out macro-permeability error."""

import argparse
import hashlib
import inspect
import json
import math
import os
import time
import uuid
from pathlib import Path

import torch
import wandb
import yaml
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.losses.data_losses import H1Loss
from neuralop.models.fno import FNO, TFNO
from torch.utils.data import DataLoader, TensorDataset

from utils.brinkman_amitex import load_betamap, load_velomap
from utils.helpers import compute_laplacian_fd_2d, recon_pressure_grad_2d_fd

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = ("[0,1]", "[1,0]")
METRIC = "permeability/best_relative_log_supremum"


def load_fields(directory):
    """Read every sample in a dataset, in a reproducible order."""
    directory = Path(directory)
    paths = sorted((directory / "mesh").glob("mID_*.vtk"))
    if not paths:
        raise ValueError(f"No mID_*.vtk files found in {directory / 'mesh'}")
    identifiers = [path.stem[4:] for path in paths]
    inputs, targets = [], []
    for identifier in identifiers:
        beta = load_betamap(str(directory), identifier)
        beta = beta.clone()
        beta[beta > 0] = torch.log10(beta[beta > 0])
        velocity = load_velomap(str(directory), identifier, velo_dir=1)
        inputs.append(beta[:, :, 0].unsqueeze(0).float())
        targets.append(velocity[:, :, :, 0].float())
    return identifiers, torch.stack(inputs), torch.stack(targets)


def rotate_fields(x, y):
    """Use the same 90-degree rotation as load_stokesbrinkman."""
    rotated_x = torch.rot90(x, k=1, dims=(-2, -1))
    rotated_y = torch.rot90(y, k=1, dims=(-2, -1))
    rotated_y = torch.stack((-rotated_y[:, 1], rotated_y[:, 0]), dim=1)
    return rotated_x, rotated_y


def fitted_normalizer(data, enabled):
    if not enabled:
        return None
    normalizer = UnitGaussianNormalizer(dim=list(range(data.ndim)))
    normalizer.fit(data)
    return normalizer


def make_model():
    # Match the TFNO architecture in train.py on the source branch.
    parameters = inspect.signature(FNO.__init__).parameters
    if "projection_channels" in parameters:
        projection = {"projection_channels": 64}
    elif "projection_channel_ratio" in parameters:
        projection = {"projection_channel_ratio": 2}
    else:
        raise RuntimeError("Unsupported neuralop FNO constructor: cannot configure 64 projection channels")
    return TFNO(
        n_modes=(32, 32),
        in_channels=1,
        out_channels=2,
        hidden_channels=32,
        factorization="tucker",
        rank=0.42,
        **projection,
    )


def permeability_matrix(inputs, velocities, device):
    """Reconstruct K as in evaluate.py, with [0,1] and [1,0] as columns."""
    columns = []
    for x, velocity in zip(inputs, velocities):
        x = x.to(device)
        beta = torch.where(x > 0, torch.pow(10.0, x), x)[0]
        gradient = recon_pressure_grad_2d_fd(
            beta, velocity.to(device), L=[1, 1], fix_x_bnd=True, fix_y_bnd=True
        )
        columns.append(-gradient.mean(dim=(0, 2, 3)))
    matrix = torch.stack(columns, dim=1)
    if not torch.isfinite(matrix).all():
        raise ValueError("Non-finite reconstructed pressure gradient")
    return torch.linalg.solve(matrix, torch.eye(2, device=device))


def pressure_gradient(log_beta, velocity):
    beta = torch.where(log_beta > 0, torch.pow(10.0, log_beta), log_beta)
    if not torch.isfinite(beta).all():
        raise ValueError("Non-finite decoded beta in pressure-gradient loss")
    if log_beta.ndim != 4 or log_beta.shape[1] != 1 or velocity.ndim != 4 or velocity.shape[1] != 2:
        raise ValueError("Expected beta [B,1,H,W] and velocity [B,2,H,W]")
    if log_beta.shape[0] != velocity.shape[0] or log_beta.shape[-2:] != velocity.shape[-2:]:
        raise ValueError("Beta and velocity batch/spatial dimensions differ")
    height, width = velocity.shape[-2:]
    gradient = compute_laplacian_fd_2d(
        velocity, dx=1 / height, dy=1 / width, fix_x_bnd=True, fix_y_bnd=True
    ) - beta * velocity
    if not torch.isfinite(gradient).all():
        raise ValueError("Non-finite pressure gradient in training loss")
    return gradient


PRESSURE_ABSOLUTE_FLOOR = 1e-12
PRESSURE_DATASET_FLOOR_FRACTION = 0.01
PRESSURE_MEAN_FIELD_FRACTION = 0.1


def pressure_gradient_losses(log_beta, prediction, target, normalization="legacy", reference_scale=None):
    """Return summed per-sample field/mean losses and detached denominator statistics."""
    predicted_gradient = pressure_gradient(log_beta, prediction)
    with torch.no_grad():
        target_gradient = pressure_gradient(log_beta, target)
    pred_mean = predicted_gradient.mean(dim=(-2, -1))
    target_mean = target_gradient.mean(dim=(-2, -1))
    field_norm = torch.linalg.vector_norm(target_gradient.flatten(start_dim=1), dim=1)
    mean_norm = torch.linalg.vector_norm(target_mean, dim=1)
    field_error_norm = torch.linalg.vector_norm(
        (predicted_gradient - target_gradient).flatten(start_dim=1), dim=1
    )
    if normalization == "legacy":
        if (field_norm <= 0).any() or (mean_norm <= 0).any():
            raise ValueError("Zero target pressure gradient or mean in legacy training loss")
        field_denom, mean_denom = field_norm, mean_norm
        numerator = field_error_norm
    elif normalization == "stable":
        if reference_scale is None or not math.isfinite(reference_scale) or reference_scale < 0:
            raise ValueError("Stable pressure normalization requires a finite non-negative training reference scale")
        spatial_factor = math.sqrt(target.shape[-2] * target.shape[-1])
        field_rms = field_norm / spatial_factor
        floor = max(PRESSURE_ABSOLUTE_FLOOR, PRESSURE_DATASET_FLOOR_FRACTION * reference_scale)
        field_denom = field_rms.clamp_min(floor)
        mean_denom = torch.maximum(mean_norm, (PRESSURE_MEAN_FIELD_FRACTION * field_rms).clamp_min(floor))
        numerator = field_error_norm / spatial_factor
    else:
        raise ValueError(f"Unknown pressure normalization: {normalization}")
    field_error = numerator / field_denom
    mean_error = torch.linalg.vector_norm(pred_mean - target_mean, dim=1) / mean_denom
    return field_error.sum(), mean_error.sum(), {
        "target_field_rms": (field_norm / math.sqrt(target.shape[-2] * target.shape[-1])).detach(),
        "target_mean_norm": mean_norm.detach(),
        "field_denominator": field_denom.detach(),
        "mean_denominator": mean_denom.detach(),
        "mean_clamped": (mean_denom > mean_norm).detach().float(),
    }


def pressure_reference_scale(x, y, device, batch_size):
    """Median target-gradient spatial RMS over the training set for one direction."""
    scales = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            raw_x, raw_y = x[start:start + batch_size], y[start:start + batch_size]
            gradient = pressure_gradient(raw_x.to(device), raw_y.to(device))
            scale = torch.linalg.vector_norm(gradient.flatten(start_dim=1), dim=1)
            scales.append((scale / math.sqrt(raw_y.shape[-2] * raw_y.shape[-1])).cpu())
    if not scales:
        raise ValueError("Training data is empty")
    return torch.cat(scales).median().item()


def gradient_diagnostics(components, model):
    """Gradient norms and pairwise cosine similarities on all trainable model parameters."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    vectors = {}
    for name, loss in components.items():
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        vectors[name] = [g.detach() if g is not None else torch.zeros_like(p)
                         for p, g in zip(parameters, gradients)]
    result = {}
    norms = {}
    for name, vector in vectors.items():
        norm = torch.sqrt(sum((g.abs() ** 2).sum() for g in vector))
        if not torch.isfinite(norm):
            raise ValueError(f"Non-finite {name} gradient norm")
        norms[name] = norm
        result[f"grad_norm_{name}"] = norm.item()
    names = list(vectors)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            valid = norms[name] > 0 and norms[other] > 0
            result[f"grad_cosine_defined_{name}_{other}"] = int(valid)
            if valid:
                dot = sum((a.conj() * b).real.sum() for a, b in zip(vectors[name], vectors[other]))
                result[f"grad_cosine_{name}_{other}"] = (dot / (norms[name] * norms[other])).item()
    return result


LOG_METRIC_RELATIVE_FLOOR = 1e-12


def relative_log_supremum(predicted, truth):
    """Relative log-supremum error with a relative floor for exact-zero entries.

    The paper's expression is undefined at K=0. This preserves the expression
    for nonzero entries and treats matching structural zeros equally; a zero vs
    nonzero mismatch remains penalized.
    """
    predicted, truth = predicted.abs().to(torch.float64), truth.abs().to(torch.float64)
    if not torch.isfinite(predicted).all() or not torch.isfinite(truth).all():
        raise ValueError("Permeability matrices contain non-finite entries")
    scale = torch.maximum(predicted.amax(), truth.amax())
    if scale <= 0:
        raise ValueError("Relative log-supremum metric requires a nonzero permeability scale")
    floor = scale * LOG_METRIC_RELATIVE_FLOOR
    predicted = predicted.clamp_min(floor)
    truth = truth.clamp_min(floor)
    log_truth = torch.log(truth)
    denominator = log_truth.abs().amax()
    if denominator <= torch.finfo(log_truth.dtype).eps:
        raise ValueError("Relative log-supremum denominator is zero")
    return (torch.log(predicted) - log_truth).abs().amax() / denominator


def count_log_metric_floor_entries(predicted, truth):
    scale = torch.maximum(predicted.abs().amax(), truth.abs().amax())
    if scale <= 0:
        return int(predicted.numel())
    floor = scale * LOG_METRIC_RELATIVE_FLOOR
    return int(((predicted.abs() <= floor) | (truth.abs() <= floor)).sum().item())


def evaluate_pair(models, normalizers, fields, device, sample_callback=None):
    """Evaluate permeability errors, including the paper's relative log-supremum J."""
    x1, y1 = fields
    x0, y0 = rotate_fields(x1, y1)
    log_sup_errors, frobenius_errors, l2_errors = [], [], [[], []]
    log_floor_entries = 0
    for model in models:
        model.eval()
    with torch.no_grad():
        for i in range(len(x1)):
            raw_x = (x0[i], x1[i])
            raw_y = (y0[i], y1[i])
            predictions = []
            for d in range(2):
                x = raw_x[d].unsqueeze(0).to(device)
                in_norm, out_norm = normalizers[d]
                if in_norm is not None:
                    x = in_norm.transform(x)
                pred = models[d](x)
                if out_norm is not None:
                    pred = out_norm.inverse_transform(pred)
                pred = pred[0]
                target = raw_y[d].to(device)
                denominator = torch.linalg.vector_norm(target)
                if not torch.isfinite(pred).all() or denominator <= 0:
                    raise ValueError(f"Invalid prediction or zero target for validation sample {i}")
                l2_errors[d].append((torch.linalg.vector_norm(pred - target) / denominator).item())
                predictions.append(pred)
            try:
                true_k = permeability_matrix(raw_x, raw_y, device)
                pred_k = permeability_matrix(raw_x, predictions, device)
            except torch.linalg.LinAlgError as exc:
                raise ValueError(f"Singular permeability matrix for validation sample {i}") from exc
            denominator = torch.linalg.matrix_norm(true_k.abs())
            frobenius_error = torch.linalg.matrix_norm(pred_k.abs() - true_k.abs()) / denominator
            log_sup_error = relative_log_supremum(pred_k, true_k)
            log_floor_entries += count_log_metric_floor_entries(pred_k, true_k)
            if not torch.isfinite(frobenius_error):
                raise ValueError(f"Non-finite permeability error for validation sample {i}")
            frobenius_errors.append(frobenius_error.item())
            log_sup_errors.append(log_sup_error.item())
            if sample_callback is not None:
                sample_callback(i, raw_x, raw_y, predictions, true_k, pred_k)
    if not log_sup_errors:
        raise ValueError("Validation data is empty")
    log_quantiles = torch.tensor(log_sup_errors, dtype=torch.float64).quantile(
        torch.tensor([0.5, 0.9], dtype=torch.float64)
    )
    return {
        "relative_log_supremum": sum(log_sup_errors) / len(log_sup_errors),
        "relative_log_supremum_median": log_quantiles[0].item(),
        "relative_log_supremum_p90": log_quantiles[1].item(),
        "relative_frobenius": sum(frobenius_errors) / len(frobenius_errors),
        "log_metric_floor_entries": log_floor_entries,
        "log_metric_relative_floor": LOG_METRIC_RELATIVE_FLOOR,
        "relative_l2_01": sum(l2_errors[0]) / len(log_sup_errors),
        "relative_l2_10": sum(l2_errors[1]) / len(log_sup_errors),
        "samples": len(log_sup_errors),
    }


def normalizer_metadata(normalizer):
    if normalizer is None:
        return None
    return {
        "mean": normalizer.mean.detach().cpu(),
        "std": normalizer.std.detach().cpu(),
        "eps": normalizer.eps,
    }


def restore_normalizer(metadata, device):
    if metadata is None:
        return None
    return UnitGaussianNormalizer(
        mean=metadata["mean"], std=metadata["std"], eps=metadata["eps"]
    ).to(device)


def plot_validation(models, normalizers, fields, ids, device, output_dir, sample_count):
    """Save field comparisons and permeability plots from the same validation pass."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    true_matrices, predicted_matrices = [], []

    def save_sample(index, raw_x, raw_y, predictions, true_k, pred_k):
        true_matrices.append(true_k.detach().cpu())
        predicted_matrices.append(pred_k.detach().cpu())
        if index >= sample_count:
            return
        for direction in range(2):
            x = raw_x[direction].detach().cpu()
            y = raw_y[direction].detach().cpu()
            prediction = predictions[direction].detach().cpu()
            fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), constrained_layout=True)
            images = [x[0], y[0], prediction[0], y[1], prediction[1]]
            titles = ["log10(beta)", "u1 ground truth", "u1 predicted",
                      "u2 ground truth", "u2 predicted"]
            for j, (ax, image, title) in enumerate(zip(axes, images, titles)):
                kwargs = {"cmap": "gray"} if j == 0 else {}
                if j in (1, 2):
                    kwargs.update(vmin=y[0].min().item(), vmax=y[0].max().item())
                elif j in (3, 4):
                    kwargs.update(vmin=y[1].min().item(), vmax=y[1].max().item())
                im = ax.imshow(image.numpy(), **kwargs)
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, orientation="horizontal")
            fig.suptitle(f"Validation sample {ids[index]}, loading {DIRECTIONS[direction]}")
            path = output_dir / f"sample_{index:03d}_{'01' if direction == 0 else '10'}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)

    result = evaluate_pair(models, normalizers, fields, device, sample_callback=save_sample)
    truth = torch.stack(true_matrices).abs().numpy()
    prediction = torch.stack(predicted_matrices).abs().numpy()
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for row in range(2):
        for column in range(2):
            ax = axes[row, column]
            true_values, pred_values = truth[:, row, column], prediction[:, row, column]
            valid = (true_values > 0) & (pred_values > 0)
            omitted = int((~valid).sum())
            if valid.any():
                x, y = true_values[valid], pred_values[valid]
                lower, upper = min(x.min(), y.min()), max(x.max(), y.max())
                if lower == upper:
                    lower, upper = lower / 2, upper * 2
                ax.loglog(x, y, ".", markersize=4)
                ax.loglog([lower, upper], [lower, upper], "-k", linewidth=1)
                ax.set_xlim(lower, upper)
                ax.set_ylim(lower, upper)
                ax.set_aspect("equal", adjustable="box")
            else:
                ax.text(0.5, 0.5, "No positive pairs", ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_xlabel(f"Ground truth |K{row + 1}{column + 1}|")
            ax.set_ylabel(f"Predicted |K{row + 1}{column + 1}|")
            ax.set_title(f"{int(valid.sum())} plotted; {omitted} zero pairs omitted")
    path = output_dir / "K_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved permeability and {2 * min(sample_count, len(ids))} field plots to {output_dir}")
    return result


def evaluate_checkpoint(path, validation_dir, plot_dir=None, plot_samples=3):
    """Re-evaluate a trusted paired checkpoint using its saved normalizers."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format") != "bpmf-permeability-pair-v1":
        raise ValueError(f"{path} is not a paired permeability sweep checkpoint")
    if (tuple(checkpoint["directions"]) != DIRECTIONS
            or len(checkpoint["model_state_dicts"]) != len(DIRECTIONS)
            or len(checkpoint["normalizers"]) != len(DIRECTIONS)
            or any(len(pair) != 2 for pair in checkpoint["normalizers"])):
        raise ValueError("Checkpoint does not contain the expected two ordered models and normalizers")
    models = [make_model().to(device) for _ in DIRECTIONS]
    for model, state in zip(models, checkpoint["model_state_dicts"]):
        model.load_state_dict(state)
    normalizers = [
        tuple(restore_normalizer(item, device) for item in pair)
        for pair in checkpoint["normalizers"]
    ]
    ids, x, y = load_fields(validation_dir)
    expected_hash = checkpoint["config"]["validation_ids_sha256"]
    if dataset_hash(ids) != expected_hash:
        raise ValueError("Validation sample IDs differ from those used to select this checkpoint")
    if plot_dir is None:
        result = evaluate_pair(models, normalizers, (x, y), device)
    else:
        result = plot_validation(models, normalizers, (x, y), ids, device, plot_dir, plot_samples)
    print(json.dumps(result, indent=2))


def dataset_hash(identifiers):
    return hashlib.sha256("\n".join(identifiers).encode()).hexdigest()


WINNER_SETTINGS = {
    "learning_rate": 0.00314485937, "weight_decay": 3.20830984e-6,
    "lr_decay_gamma": 0.5, "lr_decay_step": 25, "batch_size": 16, "epochs": 100,
}


def controlled_config(checkpoint, train_ids, validation_ids, field_weight, mean_weight, normalization):
    if checkpoint.get("format") != "bpmf-permeability-pair-v1" or checkpoint.get("run_id") != "z6a3nv6k":
        raise ValueError("Controlled comparison requires the trusted z6a3nv6k best_pair.pt checkpoint")
    cfg = checkpoint["config"]
    for key, value in WINNER_SETTINGS.items():
        if key not in cfg or not math.isclose(float(cfg[key]), value, rel_tol=1e-8, abs_tol=0):
            raise ValueError(f"Winner checkpoint {key} does not match the recorded settings")
    for key, expected in (("architecture", "TFNO"), ("n_modes", [32, 32]),
                          ("hidden_channels", 32), ("rank", 0.42), ("seed", 0),
                          ("factorization", "tucker"), ("directions", list(DIRECTIONS))):
        if cfg.get(key) != expected:
            raise ValueError(f"Winner checkpoint {key} does not match the recorded architecture or seed")
    if dataset_hash(train_ids) != cfg["train_ids_sha256"] or dataset_hash(validation_ids) != cfg["validation_ids_sha256"]:
        raise ValueError("Dataset IDs do not match the winner checkpoint")
    for key in ("encode_input", "encode_output"):
        if not isinstance(cfg.get(key), bool):
            raise ValueError(f"Winner checkpoint does not contain a valid {key} setting")
    result = dict({key: cfg[key] for key in WINNER_SETTINGS}, pressure_field_weight=field_weight,
                  pressure_mean_weight=mean_weight, pressure_normalization=normalization,
                  comparison_baseline_run="z6a3nv6k")
    loss_settings(result)
    return result, {"data": {"encode_input": cfg["encode_input"], "encode_output": cfg["encode_output"]}}


def loss_settings(cfg):
    if "pressure_field_weight" in cfg and "pressure_mean_weight" in cfg:
        field_weight = float(cfg["pressure_field_weight"])
        mean_weight = float(cfg["pressure_mean_weight"])
        normalization = cfg.get("pressure_normalization", "stable")
    elif "pressure_gradient_weight" in cfg:
        field_weight = mean_weight = float(cfg["pressure_gradient_weight"])
        normalization = "legacy"
    else:
        raise ValueError("Sweep has no pressure loss weights; create a new sweep")
    if not all(math.isfinite(w) and w >= 0 for w in (field_weight, mean_weight)):
        raise ValueError("Pressure weights must be finite and non-negative")
    if normalization not in ("stable", "legacy"):
        raise ValueError("pressure_normalization must be stable or legacy")
    return field_weight, mean_weight, normalization


def train_run(train_data, val_data, base, deadline, output_dir, validation_interval, data_root,
              run_config=None, wandb_options=None, gradient_interval=10):
    with wandb.init(config=run_config, **(wandb_options or {})) as run:
        cfg = run.config
        field_weight, mean_weight, normalization = loss_settings(cfg)
        pressure_enabled = bool(field_weight or mean_weight)
        torch.manual_seed(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x1, y1 = train_data[1:]
        train_fields = (rotate_fields(x1, y1), (x1, y1))
        normalizers, models, optimizers, schedulers, loaders = [], [], [], [], []
        for x, y in train_fields:
            in_norm = fitted_normalizer(x, base["data"]["encode_input"])
            out_norm = fitted_normalizer(y, base["data"]["encode_output"])
            normalizers.append(tuple(n.to(device) if n is not None else None for n in (in_norm, out_norm)))
            model = make_model().to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay)
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=int(cfg.lr_decay_step), gamma=float(cfg.lr_decay_gamma)
            )
            models.append(model)
            optimizers.append(optimizer)
            schedulers.append(scheduler)
            loaders.append(DataLoader(TensorDataset(x, y), batch_size=int(cfg.batch_size), shuffle=True))

        scales = [None, None]
        if pressure_enabled and normalization == "stable":
            scales = [pressure_reference_scale(x, y, device, int(cfg.batch_size))
                      for x, y in train_fields]
        run.config.update({
            "architecture": "TFNO", "n_modes": [32, 32], "hidden_channels": 32,
            "factorization": "tucker", "rank": 0.42, "directions": list(DIRECTIONS),
            "train_samples": len(x1), "validation_samples": len(val_data[1]),
            "train_ids_sha256": dataset_hash(train_data[0]),
            "validation_ids_sha256": dataset_hash(val_data[0]),
            "encode_input": base["data"]["encode_input"],
            "encode_output": base["data"]["encode_output"],
            "validation_metric_definition": (
                "mean per-sample max(abs(log(abs(K_pred))-log(abs(K_true)))) "
                "/ max(abs(log(abs(K_true)))); exact/tiny entries are floored "
                "at max(abs(K_pred),abs(K_true))*1e-12"
            ),
            "training_loss": "H1 + pressure_field_weight * field + pressure_mean_weight * mean",
            "resolved_pressure_field_weight": field_weight,
            "resolved_pressure_mean_weight": mean_weight,
            "resolved_pressure_normalization": normalization,
            "pressure_reference_scales": scales,
            "pressure_gradient_diagnostic_interval": gradient_interval,
            "pressure_gradient_loss_definition": (
                "sum per-sample pressure-gradient field and mean errors; stable mode uses "
                "training-reference and mean/field denominator floors"
            ),
            "seed": 0, "validation_interval": validation_interval,
            "train_directory": str(data_root / "R64"),
            "validation_directory": str(data_root / "R64_unseen"),
            "domain_lengths": [1, 1],
        })
        loss_fn = H1Loss(d=2)
        best = math.inf
        best_path = Path(output_dir) / run.id / "best_pair.pt"
        if best_path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing checkpoint: {best_path}")
        best_path.parent.mkdir(parents=True, exist_ok=True)
        completed_epochs = 0
        for epoch in range(1, int(cfg.epochs) + 1):
            epoch_losses, epoch_diagnostics = [], [{}, {}]
            for d in range(2):
                models[d].train()
                totals, samples = {"h1": 0.0, "pressure_field": 0.0,
                                   "pressure_mean": 0.0, "total": 0.0}, 0
                denominator_stats = {}
                in_norm, out_norm = normalizers[d]
                for batch_index, (x, y) in enumerate(loaders[d]):
                    x, y = x.to(device), y.to(device)
                    raw_x, raw_y = x, y
                    if in_norm is not None:
                        x = in_norm.transform(x)
                    if out_norm is not None:
                        y = out_norm.transform(y)
                    optimizers[d].zero_grad(set_to_none=True)
                    prediction = models[d](x)
                    h1 = loss_fn(prediction, y)
                    if pressure_enabled:
                        physical_prediction = (out_norm.inverse_transform(prediction)
                                               if out_norm is not None else prediction)
                        field, mean, stats = pressure_gradient_losses(
                            raw_x, physical_prediction, raw_y, normalization, scales[d]
                        )
                        if normalization == "legacy" and field_weight == mean_weight:
                            loss = h1 + field_weight * (field + mean)
                        else:
                            loss = h1
                            if field_weight:
                                loss = loss + field_weight * field
                            if mean_weight:
                                loss = loss + mean_weight * mean
                        totals["pressure_field"] += field.item()
                        totals["pressure_mean"] += mean.item()
                        for key, values in stats.items():
                            denominator_stats.setdefault(key, []).append(values.cpu())
                        if batch_index == 0 and (epoch == 1 or epoch % gradient_interval == 0):
                            components = {"h1": h1}
                            if field_weight:
                                components["pressure_field"] = field
                            if mean_weight:
                                components["pressure_mean"] = mean
                            diagnostics = gradient_diagnostics(components, models[d])
                            for name, weight in (("pressure_field", field_weight), ("pressure_mean", mean_weight)):
                                if weight:
                                    diagnostics[f"grad_norm_weighted_{name}"] = diagnostics[f"grad_norm_{name}"] * weight
                            epoch_diagnostics[d] = diagnostics
                    else:
                        loss = h1
                    if not torch.isfinite(loss):
                        raise ValueError(f"Non-finite training loss at epoch {epoch}, direction {DIRECTIONS[d]}")
                    loss.backward()
                    optimizers[d].step()
                    totals["h1"] += h1.item()
                    totals["total"] += loss.item()
                    samples += len(x)
                metrics_for_direction = {key: value / samples for key, value in totals.items()}
                for key, values in denominator_stats.items():
                    values = torch.cat(values).float()
                    if key == "mean_clamped":
                        metrics_for_direction[f"{key}_fraction"] = values.mean().item()
                    else:
                        metrics_for_direction[f"{key}_min"] = values.min().item()
                        metrics_for_direction[f"{key}_median"] = values.median().item()
                        metrics_for_direction[f"{key}_max"] = values.max().item()
                epoch_losses.append(metrics_for_direction)
                schedulers[d].step()
            completed_epochs = epoch
            timed_out = time.monotonic() >= deadline
            metrics = {"epoch": epoch, "train/learning_rate": optimizers[0].param_groups[0]["lr"]}
            for d, suffix in enumerate(("01", "10")):
                metrics[f"train/h1_{suffix}"] = epoch_losses[d]["h1"]
                metrics[f"train/total_loss_{suffix}"] = epoch_losses[d]["total"]
                if pressure_enabled:
                    metrics[f"train/pressure_field_{suffix}"] = epoch_losses[d]["pressure_field"]
                    metrics[f"train/pressure_mean_{suffix}"] = epoch_losses[d]["pressure_mean"]
                    metrics[f"train/weighted_pressure_field_{suffix}"] = field_weight * epoch_losses[d]["pressure_field"]
                    metrics[f"train/weighted_pressure_mean_{suffix}"] = mean_weight * epoch_losses[d]["pressure_mean"]
                    for key, value in epoch_losses[d].items():
                        if key.startswith(("target_", "field_denominator_", "mean_denominator_", "mean_clamped_")):
                            metrics[f"train/{key}_{suffix}"] = value
                    for key, value in epoch_diagnostics[d].items():
                        metrics[f"train/{key}_{suffix}"] = value
            if epoch % validation_interval == 0 or epoch == int(cfg.epochs) or timed_out:
                result = evaluate_pair(models, normalizers, val_data[1:], device)
                score = result["relative_log_supremum"]
                metrics.update({
                    "permeability/relative_log_supremum": score,
                    "permeability/relative_log_supremum_median": result["relative_log_supremum_median"],
                    "permeability/relative_log_supremum_p90": result["relative_log_supremum_p90"],
                    "permeability/relative_frobenius": result["relative_frobenius"],
                    "permeability/log_metric_floor_entries": result["log_metric_floor_entries"],
                    "permeability/log_metric_relative_floor": result["log_metric_relative_floor"],
                    "validation/relative_l2_01": result["relative_l2_01"],
                    "validation/relative_l2_10": result["relative_l2_10"],
                    "validation/samples": result["samples"],
                })
                if score < best:
                    best = score
                    torch.save({
                        "format": "bpmf-permeability-pair-v1",
                        "epoch": epoch, "score": score, "directions": DIRECTIONS,
                        "model_type": "tfno", "model_state_dicts": [m.state_dict() for m in models],
                        "normalizers": [[normalizer_metadata(n) for n in pair] for pair in normalizers],
                        "config": dict(run.config), "run_id": run.id,
                    }, best_path)
                    run.summary["best_epoch"] = epoch
                metrics[METRIC] = best
            run.log(metrics)
            if timed_out:
                break
        if not best_path.exists():
            raise RuntimeError("Run ended without a validated checkpoint")
        run.summary["completed_epochs"] = completed_epochs
        run.summary["stopped_by_time_budget"] = time.monotonic() >= deadline
        run.summary[METRIC] = best
        artifact = wandb.Artifact(f"permeability-pair-{run.id}", type="model", metadata={
            "best_epoch": run.summary["best_epoch"], "relative_log_supremum": best,
            "directions": list(DIRECTIONS), "format": "bpmf-permeability-pair-v1",
        })
        artifact.add_file(str(best_path))
        run.log_artifact(artifact, aliases=["best"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("BPMF_DATA_ROOT"),
                        required=os.environ.get("BPMF_DATA_ROOT") is None,
                        help="Directory containing R64 and R64_unseen")
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--output-dir", default=str(ROOT / "save_dir" / "permeability_sweep"))
    parser.add_argument("--entity", default=None, help="W&B entity; defaults to the current account")
    parser.add_argument("--project", default="BPMF")
    parser.add_argument("--sweep-id", help="Resume an existing sweep instead of creating one")
    parser.add_argument("--evaluate", metavar="CHECKPOINT",
                        help="Evaluate a saved best_pair.pt instead of running a sweep")
    parser.add_argument("--plots", action="store_true",
                        help="With --evaluate, save permeability and velocity comparison plots")
    parser.add_argument("--plot-dir", help="Plot output directory (default: plots beside checkpoint)")
    parser.add_argument("--plot-samples", type=int,
                        help="Number of validation samples to plot per direction (default: 3)")
    parser.add_argument("--compare-baseline", metavar="CHECKPOINT",
                        help="Run a controlled comparison against the trusted z6a3nv6k checkpoint")
    parser.add_argument("--field-weight", type=float, help="Controlled comparison pressure-field weight")
    parser.add_argument("--mean-weight", type=float, help="Controlled comparison pressure-mean weight")
    parser.add_argument("--pressure-normalization", choices=["stable", "legacy"], default="stable")
    parser.add_argument("--gradient-interval", type=int, default=10,
                        help="Gradient diagnostics every N epochs and epoch 1 (default: 10)")
    args = parser.parse_args()
    if args.hours <= 0 or not math.isfinite(args.hours) or args.validation_interval <= 0:
        parser.error("--hours and --validation-interval must be positive")
    if args.gradient_interval <= 0:
        parser.error("--gradient-interval must be positive")
    if args.compare_baseline:
        if args.evaluate or args.sweep_id or args.field_weight is None or args.mean_weight is None:
            parser.error("--compare-baseline requires both weights and cannot be combined with --evaluate or --sweep-id")
    elif args.field_weight is not None or args.mean_weight is not None or args.pressure_normalization != "stable":
        parser.error("Comparison loss options require --compare-baseline")
    if args.plot_samples is not None and args.plot_samples <= 0:
        parser.error("--plot-samples must be positive")
    if (args.plots or args.plot_dir is not None or args.plot_samples is not None) and not args.evaluate:
        parser.error("Plot options require --evaluate")
    if (args.plot_dir is not None or args.plot_samples is not None) and not args.plots:
        parser.error("--plot-dir and --plot-samples require --plots")
    data_root = Path(args.data_root).expanduser().resolve()
    train_dir, validation_dir = (data_root / "R64").resolve(), (data_root / "R64_unseen").resolve()
    if train_dir == validation_dir:
        parser.error("Training and validation directories must be different")
    if args.evaluate:
        plot_dir = None
        if args.plots:
            plot_dir = Path(args.plot_dir) if args.plot_dir else Path(args.evaluate).parent / "plots"
        evaluate_checkpoint(args.evaluate, validation_dir, plot_dir, args.plot_samples or 3)
        return
    deadline = time.monotonic() + args.hours * 3600
    print(f"Loading all training samples from {train_dir} and validation samples from {validation_dir}", flush=True)
    train_data, val_data = load_fields(train_dir), load_fields(validation_dir)
    if args.compare_baseline:
        checkpoint = torch.load(args.compare_baseline, map_location="cpu", weights_only=False)
        comparison, base = controlled_config(
            checkpoint, train_data[0], val_data[0], args.field_weight, args.mean_weight,
            args.pressure_normalization,
        )
        if args.validation_interval != checkpoint["config"]["validation_interval"]:
            parser.error("--validation-interval must match the winner checkpoint")
        output_dir = ROOT / "save_dir" / "permeability_comparison"
        train_run(train_data, val_data, base, math.inf, output_dir, args.validation_interval,
                  data_root, run_config=comparison,
                  wandb_options={"entity": args.entity, "project": args.project,
                                 "id": uuid.uuid4().hex, "resume": "never",
                                 "group": "controlled-permeability-loss"},
                  gradient_interval=args.gradient_interval)
        return
    with (ROOT / "config" / "BPMF_config.yaml").open(encoding="utf-8") as file:
        base = yaml.safe_load(file)["default"]
    with (ROOT / "config" / "permeability_sweep.yaml").open(encoding="utf-8") as file:
        sweep_config = yaml.safe_load(file)
    if time.monotonic() >= deadline:
        raise TimeoutError("Time budget expired while loading data; no sweep was started")
    sweep_id = args.sweep_id or wandb.sweep(sweep_config, project=args.project, entity=args.entity)
    print(f"Sweep: {sweep_id}; budget {args.hours:g} hours from process start", flush=True)
    while time.monotonic() < deadline:
        errors = []

        def run_one():
            try:
                train_run(train_data, val_data, base, deadline, args.output_dir,
                          args.validation_interval, data_root, gradient_interval=args.gradient_interval)
            except Exception as exc:
                errors.append(exc)
                raise

        wandb.agent(
            sweep_id,
            function=run_one,
            project=args.project, entity=args.entity, count=1,
        )
        if errors:
            raise RuntimeError("Sweep run failed; inspect the W&B run and terminal logs") from errors[0]


if __name__ == "__main__":
    main()
