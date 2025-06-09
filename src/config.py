import dataclasses
import pathlib
from typing import ClassVar

import pydantic


@dataclasses.dataclass
class DataConfig:
    pets_csv: pathlib.Path = pathlib.Path("./data/cats_and_dogs/pets_dataset_info.csv")


@dataclasses.dataclass
class WandbConfig:
    WANDB_LOG_PATH: pathlib.Path = pathlib.Path("/tmp/wandb_logs")
    WANDB_LOG_PATH.mkdir(parents=True, exist_ok=True)
    WANDB_ENTITY: str = "sachinruk"


class SegModelConfig(pydantic.BaseModel):
    embedding_dim: int = 768
    hidden_size: int = 128
    num_layers: int = 3


class TrainerConfig(pydantic.BaseModel):
    seed: int = 42
    batch_size: int = 32
    accumulate_grad_batches: int = 1
    num_workers: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = 10
    test_size: float = 0.1

    model_name: str = "mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k"
    project_name: str = "pets_segmentation"
    is_local: bool = True

    segmentation_model_config: SegModelConfig = SegModelConfig()

    class Config:
        extra = "forbid"
        protected_namespaces = ()
