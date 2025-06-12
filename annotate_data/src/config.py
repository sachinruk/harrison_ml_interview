import dataclasses


@dataclasses.dataclass
class ObjectDectionModel:
    model_name: str = "microsoft/Florence-2-base-ft"
    prompt: str = "<OD>"
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 8


CSV_FILE: str = "data/cats_and_dogs/pets_dataset_info.csv"
