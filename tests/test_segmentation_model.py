import torch

import src.segmentation_model as segmentation_model
import src.config as config


def test_segmentation_model():
    # Test the TimmUNet class initialization
    encoder, _ = segmentation_model.get_base_model_and_transforms(
        config.SegModelConfig(pretrained=False)
    )

    model = segmentation_model.TimmUNet(
        encoder=encoder, segmentation_model_config=config.SegModelConfig()
    )

    assert model is not None, "Model initialization failed"

    # Test the forward pass with a dummy input
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    output = model(dummy_input)

    assert output.shape == (
        1,
        1,
        224,
        224,
    ), "Output shape mismatch"
