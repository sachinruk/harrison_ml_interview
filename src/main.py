import datetime

import pandas as pd
import typer
import lightning as L
from loguru import logger

from src import (
    config,
    data,
    segmentation_model,
    trainer,
    wandb_utils,
)

app = typer.Typer()


@app.command()
def train(trainer_config_json: str = "{}", training_date: str = "") -> None:
    trainer_config = config.TrainerConfig.model_validate_json(trainer_config_json)
    L.seed_everything(trainer_config.seed)

    if training_date == "":
        now = datetime.datetime.now()
        training_date = now.strftime("%Y%m%d-%H%M%S")
    _ = wandb_utils.get_wandb_logger(trainer_config, training_date)
    logger.info("Initialized wandb logger")

    # Get the base model and transforms
    encoder, image_transforms = segmentation_model.get_base_model_and_transforms(
        trainer_config.segmentation_model_config
    )

    df = pd.read_csv(config.DataConfig.pets_csv)
    logger.info("Data loaded from CSV")
    train_df, valid_df = data.get_train_valid_split(df, trainer_config)
    logger.info("Train-validation split completed")
    train_dl, valid_dl = data.get_dataloaders(train_df, valid_df, image_transforms, trainer_config)
    logger.info("Data loaders created")

    model = segmentation_model.UNet(
        encoder=encoder,
        segmentation_model_config=trainer_config.segmentation_model_config,
    )
    logger.info("Model initialized")

    logger.info("Training started")
    trainer.train(
        model=model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        trainer_config=trainer_config,
        image_transforms=image_transforms,
    )


# Entry point for the CLI app
if __name__ == "__main__":
    app()
