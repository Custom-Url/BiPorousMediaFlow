import unittest
from unittest.mock import patch

import torch

from brinkman_amitex import (
    load_stokesbrinkman,
    macro_permeability_from_latent,
    velocity_to_permeability_latent,
)
from utils.helpers import recon_pressure_grad_2d_fd


class PermeabilityLatentTests(unittest.TestCase):
    def test_matches_existing_gradient_and_macro_permeability(self):
        torch.manual_seed(7)
        beta = torch.rand(3, 1, 8, 10, dtype=torch.float64) + 1
        velocities = [torch.randn(3, 2, 8, 10, dtype=torch.float64) for _ in range(2)]
        latents = [velocity_to_permeability_latent(beta, u, L=(2, 3)) for u in velocities]

        gradients = [
            torch.cat([
                recon_pressure_grad_2d_fd(
                    10**beta[i], u[i], L=[2, 3],
                    fix_x_bnd=True, fix_y_bnd=True,
                )
                for i in range(len(beta))
            ])
            for u in velocities
        ]
        for latent, gradient in zip(latents, gradients):
            torch.testing.assert_close(latent, -gradient)
        matrix = torch.stack([-g.mean(dim=(-2, -1)) for g in gradients], dim=-1)
        expected = torch.linalg.solve(matrix, torch.eye(2, dtype=beta.dtype).expand_as(matrix))
        torch.testing.assert_close(macro_permeability_from_latent(*latents), expected)

    def test_constant_fields_and_singular_matrix(self):
        beta = torch.full((1, 1, 5, 6), 2.0)
        u1 = torch.ones(1, 2, 5, 6)
        u1[:, 1] = 0
        u2 = torch.flip(u1, dims=[1])
        z1 = velocity_to_permeability_latent(beta, u1, mu=2)
        z2 = velocity_to_permeability_latent(beta, u2, mu=2)
        torch.testing.assert_close(macro_permeability_from_latent(z1, z2), torch.eye(2)[None] / 50)
        with self.assertRaises(RuntimeError):
            macro_permeability_from_latent(z1, z1)

    def test_loader_transforms_each_oriented_train_and_test_sample(self):
        betamap = torch.arange(1, 37, dtype=torch.float32).reshape(6, 6, 1)
        velocity = torch.arange(72, dtype=torch.float32).reshape(2, 6, 6, 1) / 10
        for condition in ("[1,0]", "[0,1]", "both"):
            with self.subTest(condition=condition), \
                    patch("brinkman_amitex.glob.glob", return_value=["mID_a.vtk", "mID_b.vtk"]), \
                    patch("brinkman_amitex.load_betamap", return_value=betamap), \
                    patch("brinkman_amitex.load_velomap", return_value=velocity):
                train, tests, _ = load_stokesbrinkman(
                    "unused", n_train=1, n_tests=1, batch_size=2,
                    test_batch_sizes=[2], train_resolution=6, loading_cond=condition,
                )
                x = torch.log10(betamap[..., 0])[None, None]
                u = velocity[..., 0][None]
                if condition == "[0,1]":
                    x = torch.rot90(x, 1, (-2, -1))
                    u = torch.rot90(u, 1, (-2, -1))
                    u = torch.stack((-u[:, 1], u[:, 0]), dim=1)
                elif condition == "both":
                    x = torch.cat((x, x.transpose(-2, -1)))
                    u = torch.cat((u, u.transpose(-2, -1)))
                expected = velocity_to_permeability_latent(x, u)
                for loader in (train, tests[6]):
                    samples = list(loader.dataset)
                    torch.testing.assert_close(torch.stack([s["x"] for s in samples]), x)
                    torch.testing.assert_close(torch.stack([s["y"] for s in samples]), expected)

    def test_output_normalizer_is_fitted_to_latent(self):
        beta = torch.full((6, 6, 1), 10.0)
        velocity = torch.arange(72, dtype=torch.float32).reshape(2, 6, 6, 1) / 10
        with patch("brinkman_amitex.glob.glob", return_value=["mID_a.vtk", "mID_b.vtk"]), \
                patch("brinkman_amitex.load_betamap", return_value=beta), \
                patch("brinkman_amitex.load_velomap", return_value=velocity):
            train, _, processor = load_stokesbrinkman(
                "unused", n_train=1, n_tests=1, batch_size=1,
                test_batch_sizes=[1], encode_output=True,
            )
        latent = train.dataset[0]["y"].unsqueeze(0)
        normalizer = processor.out_normalizer
        torch.testing.assert_close(normalizer.mean, latent.mean().reshape(1, 1, 1, 1))
        torch.testing.assert_close(
            normalizer.inverse_transform(normalizer.transform(latent)), latent
        )


if __name__ == "__main__":
    unittest.main()
