import torch
import sys
import datetime
from neuralop.models.fno import TFNO, FNO

import wandb
import argparse

from neuralop.training.trainer import Trainer
from neuralop.training.callbacks import CheckpointCallback, BasicLoggerCallback
from neuralop.utils import count_model_params
from neuralop.losses.data_losses import LpLoss, H1Loss
from utils.brinkman_amitex import load_stokesbrinkman
from custom_loss import (
    BetaULoss,
    MacroPermeabilityLoss,
    H1BetaULoss,
    H2BetaLoss,
    H2Loss,
    BetaULoss,
    H1BetaULaplacianLoss,
    H1LaplacianLoss,
)

device = "cuda"
torch.manual_seed(0)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return "True"
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return "False"
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# parser
parser = argparse.ArgumentParser(description="Dual Scale Flow Training")
parser.add_argument("--epochs", type=int, help="number of total epochs to run")
parser.add_argument("-b", "--batch-size", type=int, default=16)
parser.add_argument("--test-batch-sizes", nargs="+", type=int, default=[64])
parser.add_argument("--train_resolution", type=int, default=64)
parser.add_argument("--test_resolutions", nargs="+", type=int, default=[64])
parser.add_argument("--lr", type=float)
parser.add_argument("--wd", type=float)
parser.add_argument("--scheduler-step-size", type=int, default=40)
parser.add_argument("--scheduler-gamma", type=float, default=0.8)
parser.add_argument("--model-type", type=str)
parser.add_argument("--n-train", type=int, default=700)
parser.add_argument("--n-tests", nargs="+", type=int, default=[100])
parser.add_argument("--positional-encoding", type=str2bool, default=True)
parser.add_argument("--encode-input", type=str2bool, default=False)
parser.add_argument("--encode-output", type=str2bool, default=False)
parser.add_argument("--encoding", type=str, default="channel-wise")
parser.add_argument("--losses", type=str, default="l2h1")
parser.add_argument("--lam1", type=float, default=1.0)
parser.add_argument("--lam2", type=float, default=1.0)
parser.add_argument("--loading-cond", type=str, default="[1,0]")


if __name__ == "__main__":
    # init parser
    args = parser.parse_args()
    # load the data
    root_dir = r"<data_root_dir>"
    train_loader, test_loaders, data_processor = load_stokesbrinkman(
        root_dir=root_dir,
        n_train=args.n_train,
        n_tests=args.n_tests,
        batch_size=args.batch_size,
        test_batch_sizes=args.test_batch_sizes,
        test_resolutions=args.test_resolutions,
        train_resolution=args.train_resolution,
        grid_boundaries=[[0, 1], [0, 1]],
        positional_encoding=args.positional_encoding,
        encode_input=args.encode_input,
        encode_output=args.encode_output,
        encoding=args.encoding,
        loading_cond=args.loading_cond,
    )
    data_processor = data_processor.to(device)

    # create a tensorised FNO model
    if args.model_type == "tfno":
        model = TFNO(
            n_modes=(64, 64),
            in_channels=3,
            out_channels=2,
            hidden_channels=32,
            projection_channels=64,
            factorization="tucker",
            rank=0.42,
        )
    # create an FNO model
    elif args.model_type == "fno":
        model = FNO(
            n_modes=(64, 64),
            in_channels=3,
            out_channels=2,
            hidden_channels=32,
            projection_channels=64,
        )
    else:
        raise ValueError(f"Model type {args.model_type} not recognised.")
    model = model.to(device)

    n_params = count_model_params(model)
    print(f"\nOur model has {n_params} parameters.")
    sys.stdout.flush()

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma
    )

    # creating the losses
    if args.losses == "h1":
        l2loss = LpLoss(d=2, p=2, reduce_dims=[0, 1])
        h1loss = H1Loss(d=2, reduce_dims=[0, 1], fix_x_bnd=True, fix_y_bnd=True)
        train_loss = h1loss
        eval_losses = {"h1": h1loss, "l2": l2loss}
    elif args.losses == "l2":
        train_loss = LpLoss(d=2, p=2, reduce_dims=[0, 1], reductions="mean")
        eval_losses = {"l2": LpLoss(d=2, p=2, reduce_dims=[0, 1], reductions="mean")}
    elif args.losses == "macro_perm":
        train_loss = MacroPermeabilityLoss()
        eval_losses = {"macro_perm": MacroPermeabilityLoss()}
    elif args.losses == "beta_u":
        train_loss = BetaULoss()
        eval_losses = {"beta_u": BetaULoss()}
    elif args.losses == "h1beta_u":
        train_loss = H1BetaULoss(args.lam1)
        eval_losses = {"h1beta_u": H1BetaULoss(args.lam1)}
    elif args.losses == "h2":
        train_loss = H2Loss(reduce_dims=[0, 1])
        eval_losses = {"h2": H2Loss(reduce_dims=[0, 1])}
    elif args.losses == "h2beta":
        train_loss = H2BetaLoss(lam=args.lam1)
        eval_losses = {"h2beta": H2BetaLoss(lam=args.lam1)}
    elif args.losses == "h1betau_laplacian":
        train_loss = H1BetaULaplacianLoss(lam1=args.lam1, lam2=args.lam2)
        eval_losses = {
            "h1betau_laplacian": H1BetaULaplacianLoss(lam1=args.lam1, lam2=args.lam2)
        }
    elif args.losses == "h1_laplacian":
        train_loss = H1LaplacianLoss(lam=args.lam1)
        eval_losses = {"h1_laplacian": H1LaplacianLoss(lam=args.lam1)}
    else:
        raise ValueError(f"Losses {args.losses} not recognised.")

    # %%
    train_loss_arr = []
    eval_losses_arr = []
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss}")
    print(f"\n * Test: {eval_losses}")
    sys.stdout.flush()

    # Create the trainer
    save_dir = "<checkpoint_save_dir>"
    DATE = datetime.datetime.utcnow().strftime(
        "%Y-%m-%dT%H%M%S.%fZ"
    )  # more precise for sweeps
    # log into wandb
    wandb.login()
    wandb.init(
        project="<wandb_project>",
        name="<wandb_run_name>",
        config=args,
    )
    trainer = Trainer(
        model=model,
        n_epochs=args.epochs,
        device=device,
        callbacks=[
            CheckpointCallback(
                save_dir=save_dir,
                save_best=False,
                save_interval=10,
                save_optimizer=True,
                save_scheduler=True,
            ),
            BasicLoggerCallback(),
        ],
        data_processor=data_processor,
        wandb_log=True,
        log_test_interval=3,
        use_distributed=False,
        verbose=True,
    )

    # Train the model
    train = True
    if train is True:
        trainer.train(
            train_loader=train_loader,
            test_loaders=test_loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            regularizer=None,
            training_loss=train_loss,
            eval_losses=eval_losses,
        )

        ## save the model
        checkpoint = {
            "epoch": float("inf"),
            "valid_loss_min": 0,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(
            checkpoint, f"<checkpoint_save_dir>/{DATE}_last.pt"
        )
    wandb.finish()
