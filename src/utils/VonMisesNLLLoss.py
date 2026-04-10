import math

import torch
import torch.nn as nn


def _eval_poly(y, coef):
    coef = list(coef)
    result = coef.pop()
    while coef:
        result = coef.pop() + y * result
    return result


_I0_COEF_SMALL = [1.0, 3.5156229, 3.0899424,
                  1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2]
_I0_COEF_LARGE = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2,
                  -0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2]


def _log_modified_bessel_fn(x, order=0):
    assert order == 0 or order == 1

    y = (x / 3.75)
    y = y * y
    small = _eval_poly(y, _I0_COEF_SMALL)
    small = small.log()

    y = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y, _I0_COEF_LARGE).log()

    result = torch.where(x < 3.75, small, large)
    return result


class VonMisesNLLLoss(nn.Module):

    def __init__(self, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super(VonMisesNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loc, conc = pred.chunk(2, dim=-1)
        conc = conc.clone()
        with torch.no_grad():
            conc.clamp_(min=self.eps)
        nll = -conc * torch.cos(target - loc) + math.log(2 * math.pi) + _log_modified_bessel_fn(conc, order=0)
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError(f'{self.reduction} is not a valid value for reduction')
