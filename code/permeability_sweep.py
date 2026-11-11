"""Paired TFNO sweep ranked by held-out macro-permeability error."""

import argparse
import hashlib
import inspect
import json
import math
import os
import time
from pathlib import Path

import torch
import wandb
import yaml
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.losses.data_losses import H1Loss
from neuralop.models.fno import FNO, TFNO
from torch.utils.data import DataLoader, TensorDataset

from utils.brinkman_amitex import load_betamap, load_velomap
from utils.helpers import recon_pressure_grad_2d_fd

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


def train_run(train_data, val_data, base, deadline, output_dir, validation_interval, data_root):
    with wandb.init() as run:
        cfg = run.config
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
            "training_loss": "H1", "seed": 0, "validation_interval": validation_interval,
            "train_directory": str(data_root / "R64"),
            "validation_directory": str(data_root / "R64_unseen"),
            "domain_lengths": [1, 1],
        })
        loss_fn = H1Loss(d=2)
        best = math.inf
        best_path = Path(output_dir) / run.id / "best_pair.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        completed_epochs = 0
        for epoch in range(1, int(cfg.epochs) + 1):
            epoch_losses = []
            for d in range(2):
                models[d].train()
                total, samples = 0.0, 0
                in_norm, out_norm = normalizers[d]
                for x, y in loaders[d]:
                    x, y = x.to(device), y.to(device)
                    if in_norm is not None:
                        x = in_norm.transform(x)
                    if out_norm is not None:
                        y = out_norm.transform(y)
                    optimizers[d].zero_grad(set_to_none=True)
                    loss = loss_fn(models[d](x), y)
                    if not torch.isfinite(loss):
                        raise ValueError(f"Non-finite training loss at epoch {epoch}, direction {DIRECTIONS[d]}")
                    loss.backward()
                    optimizers[d].step()
                    total += loss.item()
                    samples += len(x)
                epoch_losses.append(total / samples)
                schedulers[d].step()
            completed_epochs = epoch
            timed_out = time.monotonic() >= deadline
            metrics = {"epoch": epoch, "train/h1_01": epoch_losses[0], "train/h1_10": epoch_losses[1],
                       "train/learning_rate": optimizers[0].param_groups[0]["lr"]}
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
    args = parser.parse_args()
    if args.hours <= 0 or not math.isfinite(args.hours) or args.validation_interval <= 0:
        parser.error("--hours and --validation-interval must be positive")
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
                          args.validation_interval, data_root)
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
