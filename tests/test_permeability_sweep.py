import sys
import tempfile
import time
import unittest
import json
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import permeability_sweep as sweep


class ConstantModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer("value", value.reshape(1, 2, 1, 1))

    def forward(self, x):
        return self.value.expand(x.shape[0], -1, x.shape[-2], x.shape[-1])


class Config(dict):
    def __getattr__(self, key):
        return self[key]


class FakeRun:
    def __init__(self):
        self.id = "test-run"
        self.config = Config(learning_rate=0.002, weight_decay=1e-5, epochs=2,
                             batch_size=2, lr_decay_step=1, lr_decay_gamma=0.5,
                             pressure_gradient_weight=0.0)
        self.summary = {}
        self.history = []
        self.artifacts = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def log(self, metrics):
        self.history.append(metrics)

    def log_artifact(self, artifact, aliases):
        self.artifacts.append((artifact, aliases))


class FakeArtifact:
    def __init__(self, name, type, metadata):
        self.name, self.type, self.metadata = name, type, metadata
        self.files = []

    def add_file(self, path):
        self.files.append(path)


class PermeabilitySweepTests(unittest.TestCase):
    def test_actual_tfno_forward(self):
        model = sweep.make_model().eval()
        with torch.no_grad():
            output = model(torch.ones(1, 1, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 2, 64, 64))

    def test_relative_error_and_normalizers(self):
        x = torch.ones(2, 1, 8, 8)
        y = torch.zeros(2, 2, 8, 8)
        y[:, 0], y[:, 1] = 1, 0.25
        x0, y0 = sweep.rotate_fields(x, y)
        fields = ((x0, y0), (x, y))
        normalizers, models = [], []
        for raw_x, raw_y in fields:
            input_norm = sweep.fitted_normalizer(raw_x, True)
            output_norm = sweep.fitted_normalizer(raw_y, True)
            normalizers.append((input_norm, output_norm))
            models.append(ConstantModel(output_norm.transform(raw_y[:1])[0, :, 0, 0]))
        result = sweep.evaluate_pair(models, normalizers, (x, y), torch.device("cpu"))
        self.assertEqual(result["samples"], 2)
        self.assertAlmostEqual(result["relative_frobenius"], 0, places=5)
        self.assertAlmostEqual(result["relative_l2_01"], 0, places=5)
        restored = sweep.restore_normalizer(sweep.normalizer_metadata(normalizers[1][1]), "cpu")
        torch.testing.assert_close(restored.inverse_transform(models[1](x)), y)

    def test_singular_prediction_is_not_silently_skipped(self):
        x = torch.ones(1, 1, 8, 8)
        y = torch.zeros(1, 2, 8, 8)
        y[:, 0] = 1
        models = [ConstantModel(torch.tensor([0.0, 0.0])),
                  ConstantModel(torch.tensor([1.0, 0.0]))]
        with self.assertRaisesRegex(ValueError, "Singular permeability"):
            sweep.evaluate_pair(models, [(None, None)] * 2, (x, y), torch.device("cpu"))

    def test_relative_frobenius_value(self):
        x = torch.ones(1, 1, 8, 8)
        y = torch.zeros(1, 2, 8, 8)
        y[:, 0], y[:, 1] = 1, 0.2
        models = [ConstantModel(torch.tensor([0.2, 2.0])),
                  ConstantModel(torch.tensor([2.0, 0.2]))]
        result = sweep.evaluate_pair(models, [(None, None)] * 2, (x, y), torch.device("cpu"))
        self.assertGreater(result["relative_frobenius"], 0)
        self.assertTrue(torch.isfinite(torch.tensor(result["relative_log_supremum"])))

    def test_paper_log_supremum_definition(self):
        truth = torch.tensor([[2.0, 0.5], [0.25, 4.0]])
        prediction = truth.clone()
        prediction[0, 0] *= 2
        expected = torch.log(torch.tensor(2.0)) / torch.log(torch.tensor(4.0))
        actual = sweep.relative_log_supremum(prediction, truth)
        self.assertAlmostEqual(actual.item(), expected.item(), places=6)

    def test_exact_zero_entries_are_handled_with_relative_floor(self):
        truth = torch.tensor([[2.0, 0.0], [0.25, 4.0]])
        prediction = truth.clone()
        self.assertAlmostEqual(
            sweep.relative_log_supremum(prediction, truth).item(), 0.0, places=7
        )
        prediction[0, 1] = 0.5
        error = sweep.relative_log_supremum(prediction, truth)
        self.assertTrue(torch.isfinite(error))
        self.assertGreater(error.item(), 0.0)
        self.assertEqual(sweep.count_log_metric_floor_entries(prediction, truth), 1)

    def test_pressure_loss_matches_reconstruction_and_backpropagates(self):
        torch.manual_seed(7)
        raw_x = torch.rand(2, 1, 8, 8) * 2
        raw_x[:, :, 0, 0] = 0
        raw_y = torch.rand(2, 2, 8, 8)
        prediction = (raw_y * 1.1 + 0.01).requires_grad_()
        field, mean = sweep.pressure_gradient_losses(raw_x, prediction, raw_y)
        expected_field, expected_mean = 0.0, 0.0
        for i in range(len(raw_x)):
            beta = torch.where(raw_x[i, 0] > 0, 10 ** raw_x[i, 0], raw_x[i, 0])
            gt = sweep.recon_pressure_grad_2d_fd(beta, raw_y[i], L=[1, 1],
                                                 fix_x_bnd=True, fix_y_bnd=True)
            pred = sweep.recon_pressure_grad_2d_fd(beta, prediction[i], L=[1, 1],
                                                   fix_x_bnd=True, fix_y_bnd=True)
            expected_field += torch.linalg.vector_norm(pred - gt) / torch.linalg.vector_norm(gt)
            expected_mean += torch.linalg.vector_norm((pred - gt).mean(dim=(-2, -1))) / \
                             torch.linalg.vector_norm(gt.mean(dim=(-2, -1)))
        torch.testing.assert_close(field, expected_field)
        torch.testing.assert_close(mean, expected_mean)
        (field + mean).backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(torch.linalg.vector_norm(prediction.grad).item(), 0)
        zero_field, zero_mean = sweep.pressure_gradient_losses(raw_x, raw_y, raw_y)
        self.assertEqual(zero_field.item(), 0)
        self.assertEqual(zero_mean.item(), 0)

    def test_pressure_loss_optimization_improves_synthetic_prediction(self):
        x = torch.ones(1, 1, 8, 8)
        target = torch.ones(1, 2, 8, 8)
        scale = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = torch.optim.SGD([scale], lr=0.1)
        initial = sum(sweep.pressure_gradient_losses(x, target * scale, target)).item()
        optimizer.zero_grad()
        loss = sum(sweep.pressure_gradient_losses(x, target * scale, target))
        loss.backward()
        optimizer.step()
        final = sum(sweep.pressure_gradient_losses(x, target * scale, target)).item()
        self.assertLess(final, initial)

    def test_pressure_loss_rejects_zero_and_nonfinite_reference(self):
        x = torch.ones(1, 1, 8, 8)
        y = torch.zeros(1, 2, 8, 8)
        with self.assertRaisesRegex(ValueError, "Zero target pressure gradient"):
            sweep.pressure_gradient_losses(x, y, y)
        x[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite decoded beta"):
            sweep.pressure_gradient_losses(x, y, y)

    def test_best_checkpoint_and_lr_decay(self):
        run = FakeRun()
        x = torch.rand(4, 1, 8, 8) + 1
        y = torch.rand(4, 2, 8, 8) + 1
        data = (["a", "b", "c", "d"], x, y)
        results = [
            {"relative_log_supremum": 0.2, "relative_log_supremum_median": 0.2,
             "relative_log_supremum_p90": 0.3, "relative_frobenius": 0.4,
             "log_metric_floor_entries": 0, "log_metric_relative_floor": 1e-12,
             "relative_l2_01": 0.3, "relative_l2_10": 0.4, "samples": 4},
            {"relative_log_supremum": 0.3, "relative_log_supremum_median": 0.3,
             "relative_log_supremum_p90": 0.4, "relative_frobenius": 0.5,
             "log_metric_floor_entries": 0, "log_metric_relative_floor": 1e-12,
             "relative_l2_01": 0.3, "relative_l2_10": 0.4, "samples": 4},
        ]
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(sweep.wandb, "init", return_value=run), \
                 mock.patch.object(sweep.wandb, "Artifact", FakeArtifact), \
                 mock.patch.object(sweep, "make_model", side_effect=lambda: torch.nn.Conv2d(1, 2, 1)), \
                 mock.patch.object(sweep, "evaluate_pair", side_effect=results):
                sweep.train_run(data, data, {"data": {"encode_input": True, "encode_output": True}},
                                time.monotonic() + 60, directory, 1, Path("/data"))
            path = Path(directory) / run.id / "best_pair.pt"
            checkpoint = torch.load(path, weights_only=False)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["score"], 0.2)
            self.assertEqual(checkpoint["directions"], sweep.DIRECTIONS)
            self.assertEqual(len(checkpoint["model_state_dicts"]), 2)
            self.assertAlmostEqual(run.history[-1]["train/learning_rate"], 0.0005)
            self.assertEqual(run.history[-1]["train/h1_01"], run.history[-1]["train/total_loss_01"])
            self.assertNotIn("train/pressure_field_01", run.history[-1])
            self.assertEqual(run.summary[sweep.METRIC], 0.2)
            self.assertEqual(run.artifacts[0][1], ["best"])

    def test_real_model_trains_and_checkpoint_reloads(self):
        torch.manual_seed(4)
        run = FakeRun()
        run.config["epochs"] = 1
        run.config["pressure_gradient_weight"] = 0.1
        x = torch.rand(2, 1, 64, 64) + 1
        y = torch.zeros(2, 2, 64, 64)
        y[:, 0], y[:, 1] = 1.0, 0.2
        data = (["a", "b"], x, y)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(sweep.wandb, "init", return_value=run), \
                 mock.patch.object(sweep.wandb, "Artifact", FakeArtifact):
                sweep.train_run(data, data, {"data": {"encode_input": True, "encode_output": True}},
                                time.monotonic() + 120, directory, 1, Path("/data"))
            path = Path(directory) / run.id / "best_pair.pt"
            for suffix in ("01", "10"):
                history = run.history[0]
                self.assertAlmostEqual(
                    history[f"train/total_loss_{suffix}"],
                    history[f"train/h1_{suffix}"] + 0.1 * (
                        history[f"train/pressure_field_{suffix}"] + history[f"train/pressure_mean_{suffix}"]
                    ), places=5
                )
            checkpoint = torch.load(path, weights_only=False)
            self.assertEqual(checkpoint["config"]["pressure_gradient_weight"], 0.1)
            with mock.patch.object(sweep, "load_fields", return_value=data), \
                 mock.patch("builtins.print") as output:
                sweep.evaluate_checkpoint(path, "/data/R64_unseen")
            result = json.loads(output.call_args.args[0])
            self.assertAlmostEqual(
                result["relative_log_supremum"], run.summary[sweep.METRIC], places=6
            )
            with mock.patch.object(sweep, "load_fields", return_value=data), \
                 mock.patch("builtins.print") as output:
                sweep.evaluate_checkpoint(path, "/data/R64_unseen", Path(directory) / "plots", 1)
            plot_result = json.loads(output.call_args.args[0])
            self.assertEqual(result, plot_result)
            self.assertTrue((Path(directory) / "plots" / "K_comparison.png").is_file())

    def test_old_sweep_requires_new_sweep_configuration(self):
        run = FakeRun()
        del run.config["pressure_gradient_weight"]
        with mock.patch.object(sweep.wandb, "init", return_value=run):
            with self.assertRaisesRegex(ValueError, "start a new sweep"):
                sweep.train_run(None, None, {}, 0, "/tmp", 1, Path("/data"))

    def test_training_physics_uses_raw_inputs_and_decoded_outputs(self):
        torch.manual_seed(9)
        x = torch.rand(2, 1, 8, 8) * 2 + 0.1
        y = torch.rand(2, 2, 8, 8) + 2
        data = (["a", "b"], x, y)
        validation = {"relative_log_supremum": 0.2, "relative_log_supremum_median": 0.2,
                      "relative_log_supremum_p90": 0.3, "relative_frobenius": 0.4,
                      "log_metric_floor_entries": 0, "log_metric_relative_floor": 1e-12,
                      "relative_l2_01": 0.3, "relative_l2_10": 0.4, "samples": 2}

        class RecordingModel(torch.nn.Conv2d):
            def __init__(self):
                super().__init__(1, 2, 1)

            def forward(self, inputs):
                result = super().forward(inputs)
                self.last_prediction = result.detach().clone()
                return result

        for enabled in (False, True):
            with self.subTest(normalization=enabled), tempfile.TemporaryDirectory() as directory:
                run = FakeRun()
                run.config.update(epochs=1, pressure_gradient_weight=0.1)
                models, observed = [], []
                pressure_loss = sweep.pressure_gradient_losses

                def make_model():
                    model = RecordingModel()
                    models.append(model)
                    return model

                def record_loss(raw_x, pred, raw_y):
                    observed.append((raw_x.detach().clone(), pred.detach().clone(), raw_y.detach().clone()))
                    return pressure_loss(raw_x, pred, raw_y)

                with mock.patch.object(sweep.wandb, "init", return_value=run), \
                     mock.patch.object(sweep.wandb, "Artifact", FakeArtifact), \
                     mock.patch.object(sweep, "make_model", side_effect=make_model), \
                     mock.patch.object(sweep, "evaluate_pair", return_value=validation), \
                     mock.patch.object(sweep, "pressure_gradient_losses", side_effect=record_loss):
                    sweep.train_run(data, data, {"data": {"encode_input": enabled, "encode_output": enabled}},
                                    time.monotonic() + 60, directory, 1, Path("/data"))
                for d, (raw_x, physical_pred, raw_y) in enumerate(observed):
                    expected_x, expected_y = (sweep.rotate_fields(x, y), (x, y))[d]
                    for j in range(len(raw_x)):
                        index = next(i for i in range(len(expected_x)) if torch.equal(raw_x[j], expected_x[i]))
                        torch.testing.assert_close(raw_y[j], expected_y[index])
                    output_norm = sweep.fitted_normalizer(expected_y, enabled)
                    expected_pred = (output_norm.inverse_transform(models[d].last_prediction)
                                     if enabled else models[d].last_prediction)
                    torch.testing.assert_close(physical_pred, expected_pred)

    def test_checkpoint_plots_use_saved_normalizers_and_validation_data(self):
        x = torch.ones(2, 1, 8, 8)
        y = torch.zeros(2, 2, 8, 8)
        y[:, 0], y[:, 1] = 1, 0.25
        data = (["sample-a", "sample-b"], x, y)
        fields = (sweep.rotate_fields(x, y), (x, y))
        normalizers, models = [], []
        for raw_x, raw_y in fields:
            input_norm = sweep.fitted_normalizer(raw_x, True)
            output_norm = sweep.fitted_normalizer(raw_y, True)
            normalizers.append((input_norm, output_norm))
            models.append(ConstantModel(output_norm.transform(raw_y[:1])[0, :, 0, 0]))
        checkpoint = {
            "format": "bpmf-permeability-pair-v1",
            "directions": sweep.DIRECTIONS,
            "model_state_dicts": [m.state_dict() for m in models],
            "normalizers": [[sweep.normalizer_metadata(n) for n in pair] for pair in normalizers],
            "config": {"validation_ids_sha256": sweep.dataset_hash(data[0])},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_pair.pt"
            torch.save(checkpoint, path)
            plot_dir = Path(directory) / "plots"
            with mock.patch.object(sweep, "make_model", side_effect=lambda: ConstantModel(torch.zeros(2))), \
                 mock.patch.object(sweep, "load_fields", return_value=data), \
                 mock.patch("builtins.print") as output:
                sweep.evaluate_checkpoint(path, "/data/R64_unseen", plot_dir, 1)
            result = json.loads(output.call_args.args[0])
            self.assertEqual(result["samples"], 2)
            self.assertAlmostEqual(result["relative_log_supremum"], 0, places=5)
            self.assertEqual({p.name for p in plot_dir.iterdir()},
                             {"K_comparison.png", "sample_000_01.png", "sample_000_10.png"})
            import matplotlib.image as mpimg
            for image in plot_dir.iterdir():
                self.assertGreater(mpimg.imread(image).var(), 0.001)
            checkpoint["config"]["validation_ids_sha256"] = sweep.dataset_hash(["other"])
            torch.save(checkpoint, path)
            with mock.patch.object(sweep, "load_fields", return_value=data), \
                 mock.patch.object(sweep, "make_model", side_effect=lambda: ConstantModel(torch.zeros(2))):
                with self.assertRaisesRegex(ValueError, "Validation sample IDs differ"):
                    sweep.evaluate_checkpoint(path, "/data/R64_unseen", plot_dir)

    def test_invalid_pair_directions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_pair.pt"
            torch.save({"format": "bpmf-permeability-pair-v1",
                        "directions": list(reversed(sweep.DIRECTIONS)),
                        "model_state_dicts": [{}, {}], "normalizers": [[None, None]] * 2}, path)
            with self.assertRaisesRegex(ValueError, "expected two ordered models"):
                sweep.evaluate_checkpoint(path, "/data/R64_unseen")

    def test_zero_permeability_entries_are_counted_in_plot(self):
        x = torch.ones(1, 1, 8, 8)
        y = torch.zeros(1, 2, 8, 8)
        y[:, 0], y[:, 1] = 1, 0.25
        x0, y0 = sweep.rotate_fields(x, y)
        models = [ConstantModel(y0[0, :, 0, 0]), ConstantModel(y[0, :, 0, 0])]
        matrix = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        import matplotlib.pyplot as plt
        real_subplots = plt.subplots
        figures = []

        def capture_figure(*args, **kwargs):
            fig, axes = real_subplots(*args, **kwargs)
            figures.append((fig, axes))
            return fig, axes

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(sweep, "permeability_matrix", return_value=matrix), \
                 mock.patch.object(plt, "subplots", side_effect=capture_figure):
                result = sweep.plot_validation(models, [(None, None)] * 2, (x, y),
                                               ["sample"], "cpu", directory, 1)
            self.assertEqual(result["log_metric_floor_entries"], 2)
            axes = figures[-1][1]
            self.assertEqual(axes[0, 1].get_title(), "0 plotted; 1 zero pairs omitted")
            self.assertEqual(axes[0, 1].texts[0].get_text(), "No positive pairs")
            self.assertTrue((Path(directory) / "K_comparison.png").is_file())


if __name__ == "__main__":
    unittest.main()
