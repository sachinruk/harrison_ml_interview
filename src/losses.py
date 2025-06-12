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


class TriMapPlusTVLoss(torch.nn.Module):
    """L1 reconstruction loss + isotropic TV smoothness."""

    def __init__(
        self,
        background_weight: float = 0.0,
        foreground_weight: float = 1.0,
        uncertain_weight: float = 0.0,
        tv_weight: float = 1e-2,
    ):
        super().__init__()
        self.background_weight = background_weight
        self.foreground_weight = foreground_weight
        self.uncertain_weight = uncertain_weight
        self.tv_weight = tv_weight

    @staticmethod
    def _total_variation(x: torch.Tensor) -> torch.Tensor:
        dh = torch.abs(x[..., 1:, :] - x[..., :-1, :]).mean()
        dw = torch.abs(x[..., :, 1:] - x[..., :, :-1]).mean()
        return dh + dw  # isotropic TV

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred, target)
        foreground_mask = target == 1
        background_mask = target == 0
        uncertain_mask = ~(foreground_mask | background_mask)
        l1 += self.foreground_weight * F.l1_loss(pred[foreground_mask], target[foreground_mask])
        l1 += self.background_weight * F.l1_loss(pred[background_mask], target[background_mask])
        l1 += self.uncertain_weight * F.l1_loss(pred[uncertain_mask], target[uncertain_mask])
        tv = self._total_variation(pred)
        return l1 + self.tv_weight * tv


def get_loss_fn(loss_fn_name: str) -> torch.nn.Module:
    """Get the loss function by name."""
    if loss_fn_name == "l1":
        return torch.nn.L1Loss()
    elif loss_fn_name == "l1_plus_tv":
        return L1PlusTVLoss()
    elif loss_fn_name == "trimap_plus_tv":
        return TriMapPlusTVLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")
