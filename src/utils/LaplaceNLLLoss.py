import torch
import torch.nn as nn


class LaplaceNLLLoss(nn.Module):
    """
    Laplace negative log-likelihood loss.
    Input pred: [..., 2*C] where last dim = [loc, scale] concatenated.
    Input target: [..., C].
    Formula: log(2 * scale) + |target - loc| / scale
    """

    def __init__(self, eps: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError(f'{self.reduction} is not a valid reduction')
