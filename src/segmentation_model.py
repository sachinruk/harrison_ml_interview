import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from timm.models.layers import ConvNormAct  # same conv-BN-activation block used inside timm

from src import config


def get_inverse_transform(
    transfrom: transforms.Compose,
) -> transforms.Normalize:
    """
    Get the inverse transform for the given transform.
    This is used to convert the model output back to the original image space.
    """
    if not isinstance(transfrom, transforms.Compose):
        raise ValueError("Expected a Compose transform")

    # Extract mean and std from the last Normalize transform
    for t in reversed(transfrom.transforms):
        if isinstance(t, transforms.Normalize):
            return transforms.Normalize(
                mean=-t.mean / t.std,
                std=1.0 / t.std,
            )
    raise ValueError("No Normalize transform found in the provided Compose transform")


def get_base_model_and_transforms(
    segmentation_model_config: config.SegModelConfig,
) -> tuple[nn.Module, config.ImageTransforms]:
    """
    Get the base model and transforms for the segmentation model.
    """
    encoder = timm.create_model(
        segmentation_model_config.model_name,
        pretrained=segmentation_model_config.pretrained,
        features_only=True,
        out_indices=segmentation_model_config.out_indices,
    )
    model_config = timm.data.resolve_data_config({}, model=encoder)
    train_transform = timm.data.create_transform(**model_config, is_training=True)
    val_transform = timm.data.create_transform(**model_config, is_training=False)

    image_transforms = config.ImageTransforms(
        train_transform=train_transform,
        val_transform=val_transform,
        inverse_transform=get_inverse_transform(val_transform),
    )
    return encoder, image_transforms


class UpBlock(nn.Module):
    """Upsample → concat skip → double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = ConvNormAct(out_ch + skip_ch, out_ch, kernel_size=3)
        self.conv2 = ConvNormAct(out_ch, out_ch, kernel_size=3)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor (B, in_ch, H/2, W/2)
            skip: skip connection tensor (B, skip_ch, H, W)
        Returns:
            torch.Tensor: output tensor (B, out_ch, H, W)
        """
        x = self.up(x)
        # handle odd input sizes
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)  # (B, in_ch + skip_ch, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ShiftedSigmoid(nn.Module):
    """Shifted sigmoid activation function."""

    def __init__(self, low: float = -0.1, high: float = 1.1):
        super().__init__()
        self.low = low
        self.high = high
        self.scale = high - low

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * self.scale + self.low


class Squasher(nn.Module):
    """Piece-wise-linear ‘squash’:
    x            for 0 ≤ x ≤ 1
    alpha·x          for x < 0
    1 + alpha(x–1)   for x > 1
    """

    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha  # make this a nn.Parameter if you want it learnable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # keep the in-range part; everything else becomes the ‘excess’
        clamped = x.clamp(0.0, 1.0)  # or F.hardtanh(x, 0.0, 1.0)
        return clamped + self.alpha * (x - clamped)


class UNet(nn.Module):
    """
    UNet with a timm backbone.
    • encoder_name – any model that supports features_only=True (MobileNet-V4, V3, V2, ResNet, ConvNeXt…)
    • num_classes   – number of output channels (1 for a single-channel mask)
    """

    def __init__(
        self,
        encoder: nn.Module,
        segmentation_model_config: config.SegModelConfig,
    ):
        super().__init__()

        # ----- Encoder ------------------------------------------------------
        self.encoder = encoder
        enc_chs = self.encoder.feature_info.channels()  # [C1, C2, C3, C4, C5]

        # ----- Bridge (bottom of the “U”) -----------------------------------
        self.center = nn.Sequential(
            ConvNormAct(enc_chs[-1], enc_chs[-1], 3),
            ConvNormAct(
                enc_chs[-1], segmentation_model_config.decoder_channels[0], 3
            ),  # shrink to decoder’s first width
        )

        # ----- Decoder ------------------------------------------------------
        # You can tune this list – it controls decoder widths at each scale
        decoder_channels = segmentation_model_config.decoder_channels

        self.up_blocks = nn.ModuleList()
        for i in range(
            len(decoder_channels) - 1
        ):  # four up-sampling steps (bring C5 → C1 resolution)
            self.up_blocks.append(
                UpBlock(decoder_channels[i], enc_chs[-(i + 2)], decoder_channels[i + 1])
            )

        # ----- Segmentation head & final resize -----------------------------
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[-1], decoder_channels[-1], kernel_size=2, stride=2),
            ConvNormAct(decoder_channels[-1], decoder_channels[-1], kernel_size=3),
            ConvNormAct(decoder_channels[-1], segmentation_model_config.num_classes, kernel_size=3),
        )
        self.final_activation = ShiftedSigmoid()

    def forward(self, x):
        h, w = x.shape[-2:]  # remember input size
        feats = self.encoder(x)  # list: [C1, C2, C3, C4, C5]

        x = self.center(feats[-1])  # start from deepest feature (C5)
        for i, up in enumerate(self.up_blocks):  # go C5→C4→…→C1
            x = up(x, feats[-(i + 2)])

        # x shape: (B, decoder_channels[-1], H/2, W/2)
        x = self.seg_head(x)  # (B, num_classes, H, W)
        if x.shape[-2:] != (h, w):
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return self.final_activation(x)  # (B, num_classes, H, W)
