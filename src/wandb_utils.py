import os

import wandb
import wandb.wandb_run

from src import config


def get_wandb_logger(
    trainer_config: config.TrainerConfig, training_date: str
) -> wandb.wandb_run.Run:
    """Initialize wandb logger

    Args:
        trainer_config (config.TrainerConfig): trainer config
        training_date (str):  date of training run

    Raises:
        ValueError: If failed to login to wandb

    Returns:
        loggers.WandbLogger: wandb logger
    """
    is_logged_in = wandb.login(
        key=os.environ.get("WANDB_API_KEY"),
    )
    if not is_logged_in:
        raise ValueError("Failed to login to wandb")
    model_name = f"{training_date}"
    project = trainer_config.project_name
    if trainer_config.is_local:
        project = "debug-" + project

    run = wandb.init(
        dir=config.WandbConfig.WANDB_LOG_PATH,
        project=project,
        entity=config.WandbConfig.WANDB_ENTITY,
        name=model_name,
        config=trainer_config.model_dump(),
    )

    if run is None:
        raise ValueError("Failed to initialize wandb run")
    else:
        return run
