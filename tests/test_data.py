import pathlib

import pytest
import torch
from torchvision import transforms
from timm.data.transforms import RandomResizedCropAndInterpolation, MaybeToTensor

from src import data


@pytest.fixture(scope="session")
def sample_data():
    return {
        "image_path": [
            pathlib.Path("data/cats_and_dogs/data/86791a73-9fdb-5e29-affa-0fc8a29c4429/image.jpg"),
            pathlib.Path("data/cats_and_dogs/data/bb9f3704-2b30-57c2-b433-a1b73a3bf39a/image.jpg"),
            pathlib.Path("data/cats_and_dogs/data/4acd0a75-6353-5a9a-828e-a7f013d080ab/image.jpg"),
            pathlib.Path("data/cats_and_dogs/data/8c5508d1-6185-580e-af96-de6325ee482a/image.jpg"),
            pathlib.Path("data/cats_and_dogs/data/72f95e57-0ade-5937-a039-a0a4c18f5a3d/image.jpg"),
        ],
        "mask_path": [
            pathlib.Path("data/cats_and_dogs/data/86791a73-9fdb-5e29-affa-0fc8a29c4429/mask.jpg"),
            pathlib.Path("data/cats_and_dogs/data/bb9f3704-2b30-57c2-b433-a1b73a3bf39a/mask.jpg"),
            pathlib.Path("data/cats_and_dogs/data/4acd0a75-6353-5a9a-828e-a7f013d080ab/mask.jpg"),
            pathlib.Path("data/cats_and_dogs/data/8c5508d1-6185-580e-af96-de6325ee482a/mask.jpg"),
            pathlib.Path("data/cats_and_dogs/data/72f95e57-0ade-5937-a039-a0a4c18f5a3d/mask.jpg"),
        ],
    }


@pytest.fixture(scope="session")
def image_transform():
    return transforms.Compose(
        [
            RandomResizedCropAndInterpolation(
                size=(256, 256),
                scale=(0.08, 1.0),
                ratio=(0.75, 1.3333),
                interpolation="bicubic",
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=0
            ),
            MaybeToTensor(),
            transforms.Normalize(
                mean=torch.tensor([0.4850, 0.4560, 0.4060]),
                std=torch.tensor([0.2290, 0.2240, 0.2250]),
            ),
        ]
    )


def test_dataset_creation(sample_data, image_transform):
    dataset = data.ImageMasksDataset(
        image_paths=sample_data["image_path"],
        mask_paths=sample_data["mask_path"],
        image_transform=image_transform,
    )

    assert len(dataset) == len(sample_data["image_path"])
    assert (
        len(dataset.mask_transform.transforms) == len(image_transform.transforms) - 2
    )  # Exclude ColorJitter and Normalize

    img, mask = dataset[0]

    assert img.shape == (3, 256, 256)  # Assuming RGB images resized to 256x256
    assert mask.shape == (1, 256, 256)  # Assuming single-channel masks resized to 256x256
    assert len(mask[(mask > 0) & (mask < 1)]) > 0  # Ensure mask has some non-zero values
