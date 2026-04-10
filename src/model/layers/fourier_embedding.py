import math
from typing import List, Optional

import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    """Learnable Fourier embedding for continuous features.

    Each input dimension is independently embedded via learnable sinusoidal
    frequencies, then summed and projected.  This captures periodicity
    (angles) and multi-scale structure (distances) better than a plain
    linear layer.

    Adapted from DONUT (Knoche et al., ICCV 2025).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_freq_bands: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = nn.Embedding(input_dim, num_freq_bands)
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(input_dim)
            ]
        )
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, continuous_inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            continuous_inputs: [..., input_dim]  arbitrary batch dims.
        Returns:
            [..., hidden_dim]
        """
        # [..., input_dim, 1] * [input_dim, num_freq_bands] → [..., input_dim, num_freq_bands]
        x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
        # [..., input_dim, num_freq_bands*2+1]
        x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)

        embs: List[torch.Tensor] = []
        for i in range(self.input_dim):
            embs.append(self.mlps[i](x[..., i, :]))  # [..., hidden_dim]

        x = torch.stack(embs, dim=0).sum(dim=0)  # [..., hidden_dim]
        x = self.to_out(x)
        return x
