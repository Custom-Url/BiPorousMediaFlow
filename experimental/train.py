import os
import torch
import sys
import datetime
from pathlib import Path

import yaml
from neuralop.models.fno import TFNO, FNO

import wandb

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig


from neuralop.training.trainer import Trainer
from neuralop.utils import count_model_params
from neuralop.losses.data_losses import LpLoss, H1Loss
from neuralop.training import setup
from brinkman_amitex import load_stokesbrinkman

device = "cuda"
torch.manual_seed(0)

# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./BPMF_config.yaml", config_name="default", config_folder="../BiPorousMediaFlow/config"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="../config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

# Set-up distributed communication, if using
device, is_logger = setup(config)

# Set up WandB logging
wandb_init_args = None
if config.wandb.log and is_logger:
 wandb.login(key="4e31870235d5cf5bcbe8fa12a916362a75a51d9e")
 if config.wandb.name:
     wandb_name = config.wandb.name
 else:
     wandb_name = "_".join(
         f"{var}"
         for var in [
             config_name,
             config.fno2d.n_layers,
             config.fno2d.n_modes_width,
             config.fno2d.n_modes_height,
             config.fno2d.hidden_channels,
             config.fno2d.factorization,
             config.fno2d.rank,
             config.patching.levels,
             config.patching.padding,
         ]
     )
 wandb_init_args = dict(
     config=config,
     name=wandb_name,
     group=config.wandb.group,
     project=config.wandb.project,
     entity=config.wandb.entity,
 )
 if config.wandb.sweep:
     for key in wandb.config.keys():
         config.params[key] = wandb.config[key]

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print config to screen
if config.verbose:
 pipe.log()
 sys.stdout.flush()


# Define sweep configuration
sweep_config = {
    "method": "bayes",
    "metric": {"name": "f1", "goal": "maximize"},
    "parameters": {}
}


# Add parameters from the config file to the sweep configuration
for param_name, param_values in config.wandb.params.items():
    # Detect if parameter is defined as discrete or as a range
    if "values" in param_values:
        # Discrete list of possible values
        sweep_config["parameters"][param_name] = {"values": param_values["values"]}
    elif "min" in param_values and "max" in param_values:
        # Continuous range
        sweep_config["parameters"][param_name] = {
            "min": param_values["min"],
            "max": param_values["max"]
        }
    else:
        raise ValueError(f"Parameter '{param_name}' must define either 'values' or 'min'/'max'.")


# Initialize a new sweep
sweep_id = wandb.sweep(sweep_config, project="BPMF")

def train():
    wandb.init(project="BPMF")
    cfg = wandb.config


   # Update config values with those stored in cfg
    config.data.n_train = cfg.n_train
    n_epochs = cfg.n_epochs
    config.opt.learning_rate = cfg.learning_rate
    config.opt.weight_decay = cfg.weight_decay

    config.fno2d.data_channels = config.fno2d.in_channels

    run_name = "_".join(
    f"{var}"
    for var in [
     "BPMF",
     n_epochs,
     config.data.n_train,
     config.opt.learning_rate,
     config.opt.weight_decay,
    ]
    )
    wandb.run.name = run_name
    

    # load the data
    root_dir = r"/mnt/data2/yc2634/ML/StokesBrinkman/R64"
    train_loader, test_loaders, data_processor = load_stokesbrinkman(
        root_dir=root_dir,
        n_train=cfg.n_train,
        n_tests=config.data.n_tests,
        batch_size=config.data.batch_size,
        test_batch_sizes=config.data.test_batch_sizes,
        test_resolutions=config.data.test_resolutions,
        train_resolution=config.data.train_resolution,
        grid_boundaries=[[0, 1], [0, 1]],
        positional_encoding=config.data.positional_encoding,
        encode_input=config.data.encode_input,
        encode_output=config.data.encode_output,
        loading_cond=cfg.loading_cond,
    )
    data_processor = data_processor.to(device)

    


# in_channels is the data channel count: neuralop's FNO adds the 2 grid channels itself
    # create a tensorised FNO model
    if cfg.model_type == "tfno":
        model = TFNO(
            n_modes=(32, 32),
            in_channels=1,
            out_channels=2,
            hidden_channels=32,
            projection_channels=64,
            factorization="tucker",
            rank=0.42,
        )
    # create an FNO model
    elif cfg.model_type == "fno":
        model = FNO(
            n_modes=(32, 32),
            in_channels=1,
            out_channels=2,
            hidden_channels=32,
            projection_channels=64,
        )
    else:
        raise ValueError(f"Model type {cfg.model_type} not recognised.")

    model = model.to(device)

    print("CONFIG:", config)
    
    n_params = count_model_params(model)
    print(f"\nOur model has {n_params} parameters.")
    sys.stdout.flush()

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )

    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    train_loss = h1loss
    eval_losses = {"h1": h1loss, "l2": l2loss}

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
    save_dir = "save_dir"
    DATE = datetime.datetime.utcnow().strftime(
        "%Y-%m-%dT%H%M%S.%fZ"
    )  # more precise for sweeps



    trainer = Trainer(
        model=model,
        n_epochs=config.opt.n_epochs,
        data_processor=data_processor,
        device=device,
        log_output=config.wandb.log_output,
        use_distributed=config.distributed.use_distributed,
        verbose=config.verbose,
        wandb_log = config.wandb.log
    )

    # Log parameter count
    if is_logger:
        n_params = count_model_params(model)

        if config.verbose:
            print(f"\nn_params: {n_params}")
            sys.stdout.flush()

        if config.wandb.log:
            to_log = {"n_params": n_params}
            if config.n_params_baseline is not None:
                to_log["n_params_baseline"] = (config.n_params_baseline,)
                to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
                to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
            wandb.log(to_log)
            wandb.watch(model)


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
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            checkpoint, f"{save_dir}/{DATE}_last.pt"
        )
        print(f"Saved checkpoint {save_dir}/{DATE}_last.pt (loading_cond={cfg.loading_cond})")

# Run the sweep
wandb.agent(sweep_id, function=train)


if config.wandb.log and is_logger:
 wandb.finish()
