import lightning as L
from lightning.pytorch.loggers import WandbLogger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryPrecision, BinaryRecall
import wandb

from src import config
from src import segmentation_model
from src import losses
from src import metrics


class LightningModule(L.LightningModule):
    def __init__(
        self,
        model: segmentation_model.UNet,
        learning_rate: float,
        loss_fn: nn.Module,
        image_transforms: config.ImageTransforms,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.image_transforms = image_transforms
        self.valid_precision = metrics.AveragePrecision()
        self.valid_recall = metrics.AverageRecall()

    def common_step(
        self, x: tuple[torch.Tensor, torch.Tensor], prefix: str, batch_idx: int
    ) -> torch.Tensor:
        image, mask = x
        out = self.model(image)
        loss = self.loss_fn(out, mask)
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True)

        if prefix == "valid":
            # Log metrics for validation
            self.valid_precision.update(out, (mask > config.TRIMAP_THRESHOLD).long())
            self.valid_recall.update(out, (mask > config.TRIMAP_THRESHOLD).long())

            self.log(
                f"{prefix}_precision",
                self.valid_precision,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{prefix}_recall",
                self.valid_recall,
                on_step=False,
                on_epoch=True,
            )
            if batch_idx == 0:
                table = wandb.Table(columns=["image", "mask", "prediction"])

                for i in range(len(image)):
                    # Undo your dataset-level normalization / resizing, etc.
                    img_vis = self.image_transforms.inverse_transform(image[i])

                    # Add one row per sample
                    table.add_data(wandb.Image(img_vis), wandb.Image(mask[i]), wandb.Image(out[i]))

                wandb.log({f"{prefix}_epoch_{self.current_epoch}": table})

        return loss

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss = self.common_step(batch, prefix="train", batch_idx=batch_idx)

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        _ = self.common_step(batch, prefix="valid", batch_idx=batch_idx)

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler._LRScheduler]]:
        """
        Use a lower learning rate for the encoder (backbone) and the base
        learning rate for the rest of the network.  Keep the OneCycleLR
        scheduler, providing a separate `max_lr` for each param‑group.
        """
        # Param‑group 1: encoder / backbone ─ use lr / 10
        encoder_params = self.model.encoder.parameters()

        # Param‑group 2: everything else ─ use lr
        decoder_params = (
            list(self.model.center.parameters())
            + list(self.model.up_blocks.parameters())
            + list(self.model.seg_head.parameters())
        )

        optimizer = torch.optim.Adam(
            [
                {"params": encoder_params, "lr": self.learning_rate / 10},
                {"params": decoder_params, "lr": self.learning_rate},
            ]
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.learning_rate / 10, self.learning_rate],
            total_steps=self.trainer.estimated_stepping_batches,
        )

        return [optimizer], [scheduler]


def train(
    model: segmentation_model.UNet,
    train_dl: DataLoader,
    valid_dl: DataLoader,
    trainer_config: config.TrainerConfig,
    image_transforms: config.ImageTransforms,
):
    loss_fn = losses.get_loss_fn(trainer_config.loss_fn_name)
    lightning_module = LightningModule(
        model, trainer_config.learning_rate, loss_fn, image_transforms
    )
    logger = WandbLogger()
    trainer = L.Trainer(
        max_epochs=10 if trainer_config.is_local else trainer_config.num_epochs,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        gradient_clip_val=1.0,
        precision=32,
        num_sanity_val_steps=0,
        logger=logger,
        enable_progress_bar=trainer_config.is_local,
        log_every_n_steps=100,
        # limit_train_batches=20 if trainer_config.is_local else 1.0,
        # limit_val_batches=3 if trainer_config.is_local else 1.0,
    )
    trainer.fit(lightning_module, train_dl, valid_dl)
