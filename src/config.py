import dataclasses
import pathlib
from typing import ClassVar

import pydantic


@dataclasses.dataclass
class DataConfig:
    columns: ClassVar[list[str]] = [
        "questions",
        "concepts",
        "responses",
        "timestamps",
        "selectmasks",
        "is_repeat",
    ]

    train_data_path: pathlib.Path = pathlib.Path("data/XES3G5M/kc_level/train_valid_sequences.csv")
    test_data_path: pathlib.Path = pathlib.Path("data/XES3G5M/kc_level/test.csv")
    questions_embeddings_path: pathlib.Path = pathlib.Path(
        "data/XES3G5M/metadata/embeddings/qid2content_emb.json"
    )
    concepts_embeddings_path: pathlib.Path = pathlib.Path(
        "data/XES3G5M/metadata/embeddings/cid2content_emb.json"
    )


@dataclasses.dataclass
class WandbConfig:
    WANDB_LOG_PATH: pathlib.Path = pathlib.Path("/tmp/wandb_logs")
    WANDB_LOG_PATH.mkdir(parents=True, exist_ok=True)
    WANDB_ENTITY: str = "sachinruk"


class KCModelConfig(pydantic.BaseModel):
    embedding_dim: int = 768
    hidden_size: int = 128
    num_layers: int = 3
    num_heads: int = 8
    use_lstm: bool = True


class TrainerConfig(pydantic.BaseModel):
    seed: int = 42
    batch_size: int = 32
    accumulate_grad_batches: int = 1
    num_workers: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = 10
    test_size: float = 0.1

    model_name: str = "kc_model"
    project_name: str = "kc_project"
    is_local: bool = True

    kc_model_config: KCModelConfig = KCModelConfig()

    class Config:
        extra = "forbid"
        protected_namespaces = ()
