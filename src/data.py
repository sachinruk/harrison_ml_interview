import ast
import pathlib
import random

from loguru import logger
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src import config


class ImageMasksDataset(Dataset):
    """Dataset that keeps image & mask aligned under random transforms."""

    def __init__(
        self,
        image_paths: list[pathlib.Path],
        mask_paths: list[pathlib.Path],
        image_transform: transforms.Compose,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_transform = image_transform
        self.mask_transform = transforms.Compose(
            [
                [
                    transform
                    for transform in image_transform.transforms
                    if not isinstance(transform, transforms.ColorJitter)
                ]
            ]
        )  # Mask transform is the same as image, but without ColorJitter

    def __len__(self):
        return len(self.image_paths)

    def _apply_with_seed(
        self, pil_img: Image.Image, transform: transforms.Compose, seed: int
    ) -> torch.Tensor:
        """Utility: set RNG seed, apply transform, reset not needed."""
        if transform is None:
            return pil_img
        random_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        random.seed(seed)
        torch.manual_seed(seed)
        out = transform(pil_img)
        random.setstate(random_state)
        torch.random.set_rng_state(torch_state)
        return out

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  # 1-channel

        seed = torch.randint(0, 2**32, ()).item()

        # 1) apply *shared* geometric transform
        img = self._apply_with_seed(img, self.image_transform, seed)
        mask = self._apply_with_seed(mask, self.mask_transform, seed).long()

        return img, mask


def get_train_test_split(
    df: pd.DataFrame,
    trainer_config: config.TrainerConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df["image_path"] = df["Sample_ID"].map(
        lambda x: pathlib.Path(f"./data/cats_and_dogs/data/{x}/image.jpg")
    )
    df["mask_path"] = df["Sample_ID"].map(
        lambda x: pathlib.Path(f"./data/cats_and_dogs/data/{x}/mask.jpg")
    )
    df["Breed"] = df["Breed"].apply(ast.literal_eval)

    # Split the DataFrame into train and test sets
    train_df, test_df = train_test_split(
        df,
        test_size=trainer_config.test_size,
        random_state=trainer_config.seed,
        stratify=df["Breed"].map(len),
    )
    logger.info(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")

    return train_df, test_df


def get_dataloaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_transform: config.ImageTransforms,
    trainer_config: config.TrainerConfig,
) -> tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders from a DataFrame."""
    # Create datasets
    train_dataset = ImageMasksDataset(
        image_paths=train_df["image_path"].tolist(),
        mask_paths=train_df["mask_path"].tolist(),
        image_transform=image_transform.train_transform,
    )

    test_dataset = ImageMasksDataset(
        image_paths=test_df["image_path"].tolist(),
        mask_paths=test_df["mask_path"].tolist(),
        image_transform=image_transform.val_transform,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        num_workers=trainer_config.num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=False,
        num_workers=trainer_config.num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
