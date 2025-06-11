import dataclasses
import pathlib
from typing import ClassVar

import pydantic
from torchvision import transforms


@dataclasses.dataclass
class DataConfig:
    pets_csv: pathlib.Path = pathlib.Path("./data/cats_and_dogs/pets_dataset_info.csv")


@dataclasses.dataclass
class ImageTransforms:
    train_transform: transforms.Compose
    val_transform: transforms.Compose
    inverse_transform: transforms.Normalize


@dataclasses.dataclass
class WandbConfig:
    WANDB_LOG_PATH: pathlib.Path = pathlib.Path("/tmp/wandb_logs")
    WANDB_LOG_PATH.mkdir(parents=True, exist_ok=True)
    WANDB_ENTITY: str = "sachinruk"


class SegModelConfig(pydantic.BaseModel):
    out_indices: tuple[int, ...] = (0, 1, 2, 3, 4)  # Encoder output indices
    embedding_dim: int = 768
    hidden_size: int = 128
    num_layers: int = 3
    model_name: str = "mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k"
    decoder_channels: list[int] = [256, 128, 64, 32, 16]
    num_classes: int = 1  # Number of output channels (1 for a single-channel mask)
    pretrained: bool = True


class TrainerConfig(pydantic.BaseModel):
    seed: int = 42
    batch_size: int = 32
    accumulate_grad_batches: int = 1
    num_workers: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = 10
    test_size: float = 0.1

    project_name: str = "pets_segmentation"
    is_local: bool = True

    segmentation_model_config: SegModelConfig = SegModelConfig()

    class Config:
        extra = "forbid"
        protected_namespaces = ()
