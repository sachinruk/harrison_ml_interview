import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import ConvNormAct  # same conv-BN-activation block used inside timm

from src import config


def get_base_model_and_transforms(
    segmentation_model_config: config.SegModelConfig,
) -> tuple[nn.Module, config.ImageTransforms]:
    """
    Get the base model and transforms for the segmentation model.
    """
    encoder = timm.create_model(
        segmentation_model_config.model_name,
        pretrained=True,
        features_only=True,
        out_indices=segmentation_model_config.out_indices,
    )
    config = timm.data.resolve_data_config({}, model=encoder)
    train_transform = timm.data.create_transform(**config, is_training=True)
    val_transform = timm.data.create_transform(**config, is_training=False)

    image_transforms = config.ImageTransforms(
        train_transform=train_transform,
        val_transform=val_transform,
    )
    return encoder, image_transforms


class UpBlock(nn.Module):
    """Upsample → concat skip → double conv."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = ConvNormAct(out_ch + skip_ch, out_ch, kernel_size=3)
        self.conv2 = ConvNormAct(out_ch, out_ch, kernel_size=3)

    def forward(self, x, skip):
        x = self.up(x)
        # handle odd input sizes
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TimmUNet(nn.Module):
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
            ConvNormAct(enc_chs[-1], 256, 3),  # shrink to decoder’s first width
        )

        # ----- Decoder ------------------------------------------------------
        # You can tune this list – it controls decoder widths at each scale
        decoder_channels = segmentation_model_config.decoder_channels

        self.up_blocks = nn.ModuleList()
        for i in range(4):  # four up-sampling steps (bring C5 → C1 resolution)
            self.up_blocks.append(
                UpBlock(decoder_channels[i], enc_chs[-(i + 2)], decoder_channels[i + 1])
            )

        # ----- Segmentation head & final resize -----------------------------
        self.seg_head = nn.Conv2d(
            decoder_channels[-1], segmentation_model_config.num_classes, kernel_size=1
        )

    def forward(self, x):
        h, w = x.shape[-2:]  # remember input size
        feats = self.encoder(x)  # list: [C1, C2, C3, C4, C5]

        x = self.center(feats[-1])  # start from deepest feature (C5)
        for i, up in enumerate(self.up_blocks):  # go C5→C4→…→C1
            x = up(x, feats[-(i + 2)])

        x = self.seg_head(x)  # logits (B, 1, H/2, W/2)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x  # (B, 1, H, W)


# ---------------------------------------------------------------------------
# Quick smoke test
if __name__ == "__main__":
    model = TimmUNet("mobilenetv4_l", num_classes=1, pretrained=True)  # or any timm backbone
    dummy = torch.randn(1, 3, 256, 256)
    out = model(dummy)
    print(out.shape)  # torch.Size([1, 1, 256, 256])
