# Codex Prompt: Migrate DONUT Autoregressive Decoder to FINet (Mamba-based)

## Context

**FINet** (`/Users/wardenliu/Develop/Projects/FINet/`) is a trajectory prediction model for autonomous driving that uses Mamba (state space model) as its core sequence processor. Its current decoder (`TimeDecoder`) generates all 60 future timesteps in parallel using Cross-Attention + bidirectional Mamba.

**DONUT** (`/Users/wardenliu/Develop/Projects/DONUT/`) is a decoder-only trajectory prediction model that uses tokenization (10 timesteps -> 1 token) and autoregressive prediction with 4 types of attention (Temporal, Road, Social, Mode).

**Goal**: Replace FINet's `TimeDecoder` with a DONUT-style autoregressive decoder, but use **unidirectional Mamba** as the temporal processor instead of DONUT's TemporalAttention. Keep the entire FINet encoder pipeline untouched.

## Key Design Decisions (already confirmed)

- **S (Social) interaction**: NOT included in the decoder. `spatial_mamba` already handles agent interactions.
- **Heading prediction**: NOT included. FINet only predicts (x,y) positions. No von Mises loss.
- **Local coordinate transform**: NOT included. Stay in ego-centric frame like FINet.
- **Over-prediction**: NOT included. Simplify to direct prediction only.
- **T (Temporal)**: Unidirectional Mamba (`bimamba=False`)
- **R (Road/Scene)**: Cross-Attention to scene encoding (reuse FINet's `Cross_Block`)
- **M (Mode)**: Self-attention across 6 modes with `mode_emb` + `time_emb`
- **Two stages**: Proposer (cumsum deltas) -> Refiner (residual correction)
- **Tokenization**: `t_per_tok=10`, so 60 future steps = 6 autoregressive steps

## Architecture Overview

### What stays UNCHANGED:
- `src/model/model_forecast.py`: `__init__` encoder parts (hist_embed_mamba, lane_embed, pos_embed, spatial_mamba, dense_predictor, decoder0, decoder1, samba_blocks1/2, self.tokens)
- `src/model/model_forecast.py`: `forward()` up to and including `spatial_mamba()` call
- `src/model/model_forecast.py`: `spatial_mamba()` method entirely
- `src/model/trainer_forecast.py`: `cal_loss()` entirely (output dict interface preserved)
- All data pipeline, metrics, utils
- `src/model/layers/transformer_blocks.py`
- `src/model/layers/mamba/vim_mamba.py`

### What gets REPLACED:
- `src/model/layers/time_decoder.py` -> rewrite with new decoder
- `src/model/model_forecast.py`: the 4 lines after `spatial_mamba()` that do linspace expansion + time_decoder call

## Hyperparameters

From config:
```
embed_dim = 128
future_steps = 60
num_modes = 6
t_per_tok = 10          # NEW - timesteps per token
num_pred_tokens = 6     # = future_steps / t_per_tok
dec_layer_1 = 4         # Mamba blocks for proposer's temporal processing
dec_layer_2 = 4         # Mamba blocks for refiner's temporal processing
num_heads = 8
drop_path = 0.2
```

## File Changes

### 1. NEW FILE: `src/model/layers/donut_decoder.py`

Create this file with the following classes:

#### 1.1 `SimpleTokenizer`

Converts `t_per_tok` timesteps of predicted positions into a single token embedding.

```python
class SimpleTokenizer(nn.Module):
    """
    Converts t_per_tok position steps into one token feature vector.
    Input: positions [B, num_modes, t_per_tok, 2]
    Output: token features [B, num_modes, embed_dim]
    """
    def __init__(self, embed_dim=128, t_per_tok=10):
        # Step 1: compute per-step features from positions
        #   - position delta: pos[..., 1:, :] - pos[..., :-1, :]  -> [B, M, t_per_tok-1, 2]
        #   - velocity (norm of delta)                              -> [B, M, t_per_tok-1, 1]
        #   - raw position (relative to first in window)            -> [B, M, t_per_tok-1, 2]
        #   Total feature dim per step: 5
        #
        # Step 2: embed per-step features
        #   Linear(5, embed_dim) -> LayerNorm -> ReLU -> Linear(embed_dim, embed_dim)
        #
        # Step 3: aggregate across t_per_tok-1 steps
        #   Flatten [t_per_tok-1, embed_dim] -> Linear((t_per_tok-1)*embed_dim, embed_dim)
        #
        # Step 4: final projection
        #   Linear(embed_dim, embed_dim) -> LayerNorm -> ReLU -> Linear(embed_dim, embed_dim)

        self.step_embed = nn.Sequential(
            nn.Linear(5, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.aggregate = nn.Linear((t_per_tok - 1) * embed_dim, embed_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, positions):
        # positions: [B, M, t_per_tok, 2]
        # Compute deltas
        deltas = positions[..., 1:, :] - positions[..., :-1, :]    # [B, M, t_per_tok-1, 2]
        velocity = torch.linalg.norm(deltas, dim=-1, keepdim=True) # [B, M, t_per_tok-1, 1]
        rel_pos = positions[..., 1:, :] - positions[..., :1, :]    # [B, M, t_per_tok-1, 2]
        features = torch.cat([deltas, velocity, rel_pos], dim=-1)  # [B, M, t_per_tok-1, 5]

        x = self.step_embed(features)                              # [B, M, t_per_tok-1, embed_dim]
        x = x.flatten(-2, -1)                                      # [B, M, (t_per_tok-1)*embed_dim]
        x = self.aggregate(x)                                      # [B, M, embed_dim]
        x = self.out_proj(x)                                       # [B, M, embed_dim]
        return x
```

#### 1.2 `SimpleDetokenizer`

Converts a token feature back into `t_per_tok` position deltas and scale parameters.

```python
class SimpleDetokenizer(nn.Module):
    """
    Converts token feature into t_per_tok position predictions and Laplace scale.
    Input: token [B, num_modes, embed_dim]
    Output: pos_delta [B, num_modes, t_per_tok, 2], scale [B, num_modes, t_per_tok, 2]
    """
    def __init__(self, embed_dim=128, t_per_tok=10):
        self.t_per_tok = t_per_tok
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.pos_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, t_per_tok * 2)
        )
        self.scale_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, t_per_tok * 2)
        )

    def forward(self, x):
        # x: [B, M, embed_dim]
        x = self.shared(x)
        pos_delta = self.pos_head(x).reshape(*x.shape[:2], self.t_per_tok, 2)    # [B, M, t_per_tok, 2]
        scale = F.elu(self.scale_head(x), alpha=1.0) + 1.0 + 1e-4
        scale = scale.reshape(*x.shape[:2], self.t_per_tok, 2)                    # [B, M, t_per_tok, 2]
        return pos_delta, scale
```

#### 1.3 `ModeAttention`

Self-attention across the 6 modes for the same agent, with learnable mode and time embeddings.

```python
class ModeAttention(nn.Module):
    """
    Self-attention among modes at each autoregressive step.
    Input: x [B, num_modes, embed_dim]
    Output: x [B, num_modes, embed_dim]
    """
    def __init__(self, embed_dim=128, num_modes=6, num_pred_steps=6, num_heads=8, drop_path=0.2):
        self.mode_emb = nn.Embedding(num_modes, embed_dim)
        self.time_emb = nn.Embedding(num_pred_steps + 1, embed_dim)  # 0=history, 1-6=pred steps
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, pred_step):
        # x: [B, M, embed_dim]
        # Add mode embedding
        x = x + self.mode_emb.weight[None, :x.shape[1]]   # broadcast [1, M, D]
        # Add time embedding
        x = x + self.time_emb(torch.tensor(pred_step, device=x.device, dtype=torch.long))

        # Self-attention among modes
        x_norm = self.norm(x)
        attn_out = self.attn(x_norm, x_norm, x_norm)[0]
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path2(self.mlp(x))
        return x
```

#### 1.4 `AutoregressiveStage`

One autoregressive stage (used for both proposer and refiner).

```python
class AutoregressiveStage(nn.Module):
    """
    One stage of autoregressive decoding: Mamba(T) + CrossAttn(R) + ModeAttn(M)
    repeated for num_pred_tokens steps, each producing t_per_tok timesteps.
    """
    def __init__(self, embed_dim=128, t_per_tok=10, num_modes=6,
                 num_pred_tokens=6, num_mamba_layers=4, num_heads=8,
                 drop_path=0.2, is_refiner=False):
        self.t_per_tok = t_per_tok
        self.num_pred_tokens = num_pred_tokens
        self.num_modes = num_modes
        self.is_refiner = is_refiner

        # Tokenizer
        self.tokenizer = SimpleTokenizer(embed_dim, t_per_tok)

        # T: Unidirectional Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            create_block(d_model=embed_dim, layer_idx=i, drop_path=drop_path,
                         bimamba=False, rms_norm=True)
            for i in range(num_mamba_layers)
        ])
        self.mamba_norm = RMSNorm(embed_dim, eps=1e-5)
        self.mamba_drop_path = DropPath(drop_path)

        # R: Cross-attention to scene encoding
        self.cross_attn = Cross_Block(dim=embed_dim, num_heads=num_heads,
                                       drop_path=drop_path)

        # M: Mode attention
        self.mode_attn = ModeAttention(embed_dim, num_modes, num_pred_tokens,
                                        num_heads, drop_path)

        # Detokenizer
        self.detokenizer = SimpleDetokenizer(embed_dim, t_per_tok)

        # Pi head (mode probability): applied on max-pooled features
        self.pi_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

        # If refiner, need feature fusion MLP
        if is_refiner:
            self.feature_fuse = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim),
                nn.ReLU(), nn.Linear(embed_dim, embed_dim)
            )

    def forward(self, mode_tokens, ego_feat, scene_encoding, scene_mask,
                proposed_positions=None, proposer_feats=None):
        """
        Args:
            mode_tokens: [B, M, D] from spatial_mamba (initial mode features)
            ego_feat: [B, D] ego agent's encoded feature (history summary)
            scene_encoding: [B, N_scene, D] scene context (agents + lanes)
            scene_mask: [B, N_scene] padding mask for cross-attention
            proposed_positions: [B, M, 60, 2] proposer output (only for refiner)
            proposer_feats: list of [B, M, D] per-step features from proposer

        Returns:
            y_hat: [B, M, 60, 2] predicted trajectories
            pi: [B, M] mode logits
            scal: [B, M, 60, 2] Laplace scale
            step_feats: list of [B, M, D] per-step features (for refiner input)
        """
        B = mode_tokens.shape[0]
        M = self.num_modes

        # History summary as initial temporal context
        # Expand ego_feat to all modes: [B, D] -> [B, M, 1, D]
        hist_token = ego_feat.unsqueeze(1).unsqueeze(2).expand(B, M, 1, -1)
        token_seq_list = [hist_token]   # Will accumulate tokens for Mamba

        all_positions = []
        all_scales = []
        step_feats = []
        anchor_pos = None   # Last predicted position, for cumulative offset

        for step in range(self.num_pred_tokens):
            # --- Tokenize ---
            if step == 0:
                tok = mode_tokens   # [B, M, D] from spatial_mamba
            else:
                # Use previous step's predicted positions
                tok = self.tokenizer(prev_positions)  # [B, M, D]

            # If refiner: fuse proposer features
            if self.is_refiner and proposer_feats is not None:
                tok = tok + self.feature_fuse(proposer_feats[step])

            # Append to temporal sequence
            token_seq_list.append(tok.unsqueeze(2))  # [B, M, 1, D]

            # --- T: Unidirectional Mamba on accumulated sequence ---
            seq = torch.cat(token_seq_list, dim=2)    # [B, M, step+2, D]
            seq_flat = seq.reshape(B * M, -1, seq.shape[-1])  # [B*M, step+2, D]
            residual = None
            for blk in self.mamba_blocks:
                seq_flat, residual = blk(seq_flat, residual)
            fused_add_norm_fn = rms_norm_fn
            seq_flat = fused_add_norm_fn(
                self.mamba_drop_path(seq_flat),
                self.mamba_norm.weight, self.mamba_norm.bias,
                eps=self.mamba_norm.eps, residual=residual,
                prenorm=False, residual_in_fp32=True
            )
            tok = seq_flat[:, -1].reshape(B, M, -1)   # [B, M, D] last token

            # --- R: Cross-attention to scene ---
            # Reshape for Cross_Block: [B, M, D] -> treated as [B, M, D]
            tok = self.cross_attn(tok, scene_encoding, key_padding_mask=scene_mask)
            # Output: [B, M, D]

            # --- M: Mode interaction ---
            tok = self.mode_attn(tok, step + 1)  # pred_step 1-indexed
            # Output: [B, M, D]

            # Save features for refiner
            step_feats.append(tok)

            # --- Detokenize ---
            pos_delta, scale = self.detokenizer(tok)  # [B, M, t_per_tok, 2] each

            # --- Convert deltas to absolute positions ---
            if self.is_refiner and proposed_positions is not None:
                # Refiner: output residual corrections on top of proposed
                start = step * self.t_per_tok
                end = start + self.t_per_tok
                prop_chunk = proposed_positions[:, :, start:end, :]  # [B, M, t_per_tok, 2]
                positions = prop_chunk + pos_delta
            else:
                # Proposer: cumulative sum of deltas
                cum_delta = torch.cumsum(pos_delta, dim=2)  # [B, M, t_per_tok, 2]
                if anchor_pos is None:
                    # First step: relative to origin (ego center)
                    positions = cum_delta
                else:
                    positions = anchor_pos.unsqueeze(2) + cum_delta  # offset from last pos

            all_positions.append(positions)
            all_scales.append(scale)

            # Update anchor and previous positions for next step
            anchor_pos = positions[:, :, -1, :].detach()         # [B, M, 2]
            prev_positions = positions.detach()                   # [B, M, t_per_tok, 2]

        # --- Assemble output ---
        y_hat = torch.cat(all_positions, dim=2)   # [B, M, 60, 2]
        scal = torch.cat(all_scales, dim=2)       # [B, M, 60, 2]

        # Pi: mode probabilities from max-pooled features across steps
        feat_stack = torch.stack(step_feats, dim=2)  # [B, M, num_pred_tokens, D]
        feat_pool = feat_stack.max(dim=2)[0]          # [B, M, D]
        pi = self.pi_head(feat_pool).squeeze(-1)      # [B, M]

        return y_hat, pi, scal, step_feats
```

#### 1.5 `DonutMambaDecoder` (top-level decoder)

```python
class DonutMambaDecoder(nn.Module):
    """
    DONUT-style autoregressive decoder with Mamba temporal core.
    Two stages: proposer -> refiner.
    Drop-in replacement for FINet's TimeDecoder.
    """
    def __init__(self, embed_dim=128, t_per_tok=10, num_modes=6,
                 future_steps=60, dec_layer_1=4, dec_layer_2=4,
                 num_heads=8, drop_path=0.2):
        num_pred_tokens = future_steps // t_per_tok  # = 6

        self.proposer = AutoregressiveStage(
            embed_dim=embed_dim, t_per_tok=t_per_tok, num_modes=num_modes,
            num_pred_tokens=num_pred_tokens, num_mamba_layers=dec_layer_1,
            num_heads=num_heads, drop_path=drop_path, is_refiner=False
        )
        self.refiner = AutoregressiveStage(
            embed_dim=embed_dim, t_per_tok=t_per_tok, num_modes=num_modes,
            num_pred_tokens=num_pred_tokens, num_mamba_layers=dec_layer_2,
            num_heads=num_heads, drop_path=drop_path, is_refiner=True
        )

    def forward(self, mode_tokens, ego_feat, scene_encoding, scene_mask):
        """
        Args:
            mode_tokens: [B, 6, 128] from spatial_mamba
            ego_feat: [B, 128] ego feature from scene encoding
            scene_encoding: [B, 173, 128] full scene features
            scene_mask: [B, 173] key_padding_mask (True = masked/invalid)

        Returns: (dense_pred, y_hat, pi, mode_features, new_y_hat, new_pi, scal, scal_new)
            Same interface as original TimeDecoder.forward()
        """
        # Stage 1: Proposer
        y_hat, pi, scal, prop_feats = self.proposer(
            mode_tokens, ego_feat, scene_encoding, scene_mask
        )

        # Stage 2: Refiner (residual correction)
        new_y_hat, new_pi, scal_new, _ = self.refiner(
            mode_tokens, ego_feat, scene_encoding, scene_mask,
            proposed_positions=y_hat.detach(),
            proposer_feats=prop_feats
        )

        # Return same interface as original TimeDecoder
        dense_pred = None
        mode_features = None  # Not used downstream
        return dense_pred, y_hat, pi, mode_features, new_y_hat, new_pi, scal, scal_new
```

### 2. MODIFY: `src/model/model_forecast.py`

#### 2.1 Replace TimeDecoder import

```python
# REMOVE:
from .layers.time_decoder import TimeDecoder

# ADD:
from .layers.donut_decoder import DonutMambaDecoder
```

#### 2.2 Replace `__init__` TimeDecoder instantiation

```python
# REMOVE (line ~97):
self.time_decoder = TimeDecoder(dec_layer_1=dec_layer_1, dec_layer_2=dec_layer_2)

# ADD:
self.t_per_tok = 10  # NEW hyperparameter
self.time_decoder = DonutMambaDecoder(
    embed_dim=embed_dim,
    t_per_tok=self.t_per_tok,
    num_modes=6,
    future_steps=future_steps,
    dec_layer_1=dec_layer_1,
    dec_layer_2=dec_layer_2,
    num_heads=num_heads,
    drop_path=drop_path,
)
```

#### 2.3 Replace forward() decoder call

In `forward()`, replace the 4 lines after `spatial_mamba()` call:

```python
# REMOVE (lines ~341-346):
ep_embedding = torch.linspace(0, 1, steps=self.future_steps).view(1, 1, -1, 1).to(mode.device) * mode.unsqueeze(2)
mode = x_encoder[:,:1].unsqueeze(1) + ep_embedding

dense_predict, y_hat, pi, x_mode, new_y_hat, new_pi, scal, scal_new = \
    self.time_decoder(mode, x_encoder, mask=~key_valid_mask)

# ADD:
ego_feat = x_encoder[:, 0]  # [B, 128] ego agent feature as history summary
dense_predict, y_hat, pi, x_mode, new_y_hat, new_pi, scal, scal_new = \
    self.time_decoder(mode, ego_feat, x_encoder, mask=~key_valid_mask)
```

Note: `mode` here is `fut_tok` from `spatial_mamba()` with shape `[B, 6, 128]`.

### 3. MODIFY: `conf/config.yaml`

Add new hyperparameter (optional, since it's hardcoded to 10 in the model):

```yaml
# After dec_layer_2 line, add:
t_per_tok: 10
```

### 4. NO CHANGES needed to:
- `src/model/trainer_forecast.py` - output dict interface is preserved
- `src/model/layers/transformer_blocks.py` - Cross_Block reused as-is
- `src/model/layers/mamba/vim_mamba.py` - create_block reused as-is
- Any data pipeline files

## Tensor Shape Trace (full forward pass through new decoder)

```
Input from spatial_mamba:
  mode_tokens:    [B, 6, 128]    # 6 mode tokens
  ego_feat:       [B, 128]       # ego agent summary
  scene_encoding: [B, 173, 128]  # 48 agents + 125 lanes
  scene_mask:     [B, 173]       # padding mask

=== Proposer (6 autoregressive steps) ===

hist_token = ego_feat -> [B, 6, 1, 128]  # expanded to all modes
token_seq_list = [hist_token]

Step 0:
  tok = mode_tokens                           [B, 6, 128]
  token_seq_list.append(tok.unsqueeze(2))  -> list has 2 items
  seq = cat(token_seq_list, dim=2)            [B, 6, 2, 128]
  seq_flat = reshape                          [B*6, 2, 128]
  -> Mamba (4 blocks, unidirectional)      -> [B*6, 2, 128]
  -> RMSNorm                               -> [B*6, 2, 128]
  tok = seq_flat[:, -1].reshape               [B, 6, 128]
  -> Cross_Block(tok, scene_encoding)      -> [B, 6, 128]
  -> ModeAttention(tok, step=1)            -> [B, 6, 128]
  -> Detokenize                            -> pos_delta [B, 6, 10, 2], scale [B, 6, 10, 2]
  -> cumsum(pos_delta)                     -> positions [B, 6, 10, 2]

Step 1:
  tok = tokenizer(prev_positions [B,6,10,2])  [B, 6, 128]
  seq = cat(token_seq_list)                   [B, 6, 3, 128]
  seq_flat                                    [B*6, 3, 128]
  ... same processing ...
  positions = anchor_pos + cumsum(delta)      [B, 6, 10, 2]

... Steps 2-5 similar, seq grows to [B*6, 7, 128] max ...

y_hat = cat(all_positions)                    [B, 6, 60, 2]
scal = cat(all_scales)                        [B, 6, 60, 2]
pi = pi_head(max_pool(step_feats))            [B, 6]

=== Refiner (6 autoregressive steps) ===

Same structure, but:
  - tok = tok + feature_fuse(proposer_feats[step])  # fuse proposer features
  - positions = proposed_chunk + pos_delta           # residual correction
  - proposed_positions = y_hat.detach()              # from proposer

new_y_hat: [B, 6, 60, 2]
new_pi:    [B, 6]
scal_new:  [B, 6, 60, 2]

=== Final output dict (same as before) ===
{
    "y_hat":          [B, 6, 60, 2],
    "pi":             [B, 6],
    "scal":           [B, 6, 60, 2],
    "new_y_hat":      [B, 6, 60, 2],
    "new_pi":         [B, 6],
    "scal_new":       [B, 6, 60, 2],
    "y_hat_others":   [B, 47, 60, 2],
    "ep_offsets":     [[B, 2], [B, 2]],
    "dense_predict":  None,
}
```

## Import Dependencies for `donut_decoder.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from .transformer_blocks import Cross_Block
from .mamba.vim_mamba import create_block
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, rms_norm_fn
except ImportError:
    RMSNorm, rms_norm_fn = None, None
# Fallback import if needed:
from src.model.models_mamba import RMSNorm, rms_norm_fn
```

## Verification Plan

1. **Shape check**: After implementing, run a single forward pass with dummy data and verify all output tensor shapes match the expected dict above.

2. **Loss compatibility**: Verify `cal_loss()` in `trainer_forecast.py` runs without error on the new output dict.

3. **Training smoke test**: Run `python train.py` for ~100 iterations and verify:
   - Loss decreases
   - No NaN/Inf values
   - GPU memory usage is reasonable

4. **Comparison**: The new decoder should produce 6 modes that differentiate (check that `y_hat[:, i]` differs from `y_hat[:, j]` for i != j after a few training steps).
