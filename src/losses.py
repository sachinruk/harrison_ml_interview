import torch
import torch.nn.functional as F


class L1PlusTVLoss(torch.nn.Module):
    """L1 reconstruction loss + isotropic TV smoothness."""

    def __init__(self, tv_weight: float = 1e-2):
        super().__init__()
        self.tv_weight = tv_weight  # tune in {1e-3 … 1e-1}

    @staticmethod
    def _total_variation(x: torch.Tensor) -> torch.Tensor:
        dh = torch.abs(x[..., 1:, :] - x[..., :-1, :]).mean()
        dw = torch.abs(x[..., :, 1:] - x[..., :, :-1]).mean()
        return dh + dw  # isotropic TV

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred, target)
        tv = self._total_variation(pred)
        return l1 + self.tv_weight * tv


def get_loss_fn(loss_fn_name: str) -> torch.nn.Module:
    """Get the loss function by name."""
    if loss_fn_name == "l1":
        return torch.nn.L1Loss()
    elif loss_fn_name == "l1_plus_tv":
        return L1PlusTVLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")
