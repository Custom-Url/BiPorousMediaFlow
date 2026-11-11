import sys
import tempfile
import time
import unittest
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
                             batch_size=2, lr_decay_step=1, lr_decay_gamma=0.5)
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
            self.assertEqual(run.summary[sweep.METRIC], 0.2)
            self.assertEqual(run.artifacts[0][1], ["best"])

    def test_real_model_trains_and_checkpoint_reloads(self):
        torch.manual_seed(4)
        run = FakeRun()
        run.config["epochs"] = 1
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
            with mock.patch.object(sweep, "load_fields", return_value=data), \
                 mock.patch("builtins.print") as output:
                sweep.evaluate_checkpoint(path, "/data/R64_unseen")
            import json
            result = json.loads(output.call_args.args[0])
            self.assertAlmostEqual(
                result["relative_log_supremum"], run.summary[sweep.METRIC], places=6
            )


if __name__ == "__main__":
    unittest.main()
