# FINet_dev: Tokenized Autoregressive Motion Forecasting with Mamba

## Overview

This project is an experimental fusion of **FINet** and **DONUT** for Argoverse 2 motion forecasting.

The current design keeps:

- **FINet-style scene encoding and Mamba-based interaction**
- **DONUT-style tokenized autoregressive decoding**
- **Two-stage prediction: proposer -> refiner**

The decoder is no longer the original DONUT `TRSM` stack. The current runtime path is:

```text
T -> BiMamba -> M
```

where:

- `T`: relation-aware temporal attention over history tokens and generated future tokens
- `BiMamba`: FINet-style scene/mode interaction over `[sorted scene tokens ; mode tokens]`
- `M`: relation-aware mode attention

There is **no separate runtime `R` block** anymore. Scene interaction is handled by decoder BiMamba and by a refiner-only future-token refinement head.

## Current Architecture

### 1. Scene Encoder

The encoder produces scene tokens but does not run FINet's original two-round spatial BiMamba encoder.

#### Agent branch

- Input history features per agent:
  - `x_positions_diff`
  - `x_velocity_diff`
  - valid mask
  - `x_heading_diff`
  - `cos(heading)`
  - `sin(heading)`
- Shape: `[B, N, 50, 7]`
- Processing:
  - `MLP(7 -> 64 -> 128)`
  - `4 x` uni-Mamba blocks
  - take the last hidden state
- Output:
  - `actor_feat [B, N, 128]`

#### Lane branch

- Each lane polyline is centered by its lane center
- Encoded by `LaneEmbeddingLayer`
- Output:
  - `lane_feat [B, L, 128]`

#### Scene tokens

- Concatenate actor and lane tokens:
  - `x_encoder = [actor_feat ; lane_feat]`
- Add:
  - actor type embedding
  - lane type embedding
  - position/orientation embedding from
    - `(center_x, center_y, cos(angle), sin(angle))`
- Final shape:
  - `x_encoder [B, N_scene, 128]`

This `x_encoder` is the shared scene memory for proposer and refiner.

### 2. FINet-style Mode Initialization

The decoder still uses FINet-like mode initialization:

- `decoder0(ego_token)` predicts:
  - a coarse endpoint offset `ep_offset_1`
  - an endpoint token bias `ep_tok_1`
- `ego_token` is copied into 6 modes
- a learned per-mode embedding is added
- the first mode is further biased by `ep_tok_1`

This produces:

- `mode_tokens [B, 6, 128]`

## Tokenization Strategy

### History tokens

The focal ego history is explicitly chunked:

- `50` history steps
- `t_per_tok = 10`
- `5` history chunks in total

Stage usage:

- proposer uses the first `4` chunks
- refiner uses the last `4` chunks

Each chunk is tokenized by `SimpleTokenizer` after converting the chunk into a local frame anchored at the chunk's final position and heading.

### Future tokens

Prediction is autoregressive over future tokens:

- `future_steps = 60`
- `t_per_tok = 10`
- `6` future tokens total

Each autoregressive step predicts one token, corresponding to `10` future timesteps.

## 3. Proposer

The proposer is the first autoregressive stage.

### 3.1 History-conditioned initialization

Before rollout, proposer history tokens are passed through a lightweight history adapter:

- tokenized history chunks
- `1 x` uni-Mamba history adapter
- RMSNorm
- pooled stage-specific history state

The resulting history state is projected back into mode space and added to the initial `mode_tokens`.

So proposer does not start from a pure learned mode embedding. It starts from:

```text
mode_tokens + proposer_history_bias
```

### 3.2 Autoregressive rollout

The proposer runs for `6` autoregressive steps.

At each step:

1. Build the current token
   - step 0: use the initialized mode token
   - later steps: tokenize the previous predicted chunk

2. Temporal sequence assembly
   - sequence starts with history tokens
   - then previously generated future tokens
   - metadata is tracked alongside token features:
     - token position
     - token heading
     - token time index

3. Apply `[T -> BiMamba -> M] x num_repetitions`

#### T: relation-aware temporal attention

The current token attends to the temporal token sequence with explicit relative features:

- distance
- relative direction
- relative heading
- relative token time

These 4 values are Fourier-embedded and injected into attention.

This is the DONUT-style temporal idea, but implemented as a dense PyTorch attention instead of sparse graph attention.

#### BiMamba: scene/mode interaction

Scene tokens are reordered before each BiMamba application.

Sorting center:

- initial proposer center:
  - `ego_center + ep_offset_1`
- later steps:
  - `pi`-weighted average of the current predicted mode endpoints

Only **scene tokens** are sorted.

Then the actual BiMamba input sequence is:

```text
[sorted scene tokens ; current mode tokens]
```

BiMamba updates both:

- scene tokens
- mode tokens

This is the main FINet-style scene interaction mechanism in the decoder.

#### M: relation-aware mode attention

Mode attention now uses explicit mode geometry.

For each mode, the current chunk is temporarily decoded to obtain:

- chunk-end position
- chunk-end heading

Mode-to-mode relative features are then computed:

- distance
- relative direction
- relative heading

These are Fourier-embedded and injected into mode attention, giving mode interaction a spatial meaning instead of relying only on feature similarity.

### 3.3 Detokenization

After the final repetition of each step, the token is detokenized into:

- position deltas
- Laplace scale
- heading deltas
- von Mises concentration

Outputs are then converted back to global trajectories.

For proposer:

- positions are cumulative deltas in token-local frame
- headings are cumulative `0.3 * tanh(delta_heading)` in token-local frame

### 3.4 Uncertainty parameterization

The current implementation follows DONUT-style assembly semantics:

- position scale is accumulated across autoregressive steps
- heading concentration is accumulated in inverse space and then inverted again

This avoids the earlier concentration mismatch bug.

## 4. Refiner

The refiner is the second autoregressive stage.

It differs from proposer in four ways:

1. It uses the shifted history window
   - proposer: chunks `0..3`
   - refiner: chunks `1..4`

2. It receives proposer outputs as refinement targets
   - `proposed_positions`
   - `proposed_headings`

3. It receives proposer hidden states
   - `proposer_feats`
   - these are fused into the current token at each AR step

4. It has an extra future-token refinement head after rollout

### 4.1 Residual refinement

For refiner, the predicted chunk is not decoded from scratch.

Instead:

- proposer chunk is converted into the current local frame
- the refiner predicts residual corrections on top of that local proposal

### 4.2 Future-token refinement head

After the refiner finishes its 6-step autoregressive rollout, the full future-token sequence is refined once more:

1. stack all step features into
   - `feat_stack [B, 6, 6, 128]`
2. flatten mode and time into future queries
3. run one scene cross-attention from future queries to scene tokens
4. reshape back to `[B * 6, 6, 128]`
5. run one uni-Mamba across the 6 future tokens
6. detokenize the refined token sequence again

This branch imports the useful part of FINet's decoder:

- scene information refresh over future queries
- temporal propagation over the future sequence

without abandoning the DONUT-style tokenized autoregressive backbone.

## Summary of the Current Forward Flow

```text
AV2 input
  -> agent history encoder (uni-Mamba)
  -> lane encoder
  -> scene token assembly
  -> FINet-style mode initialization
  -> ego history chunking
  -> proposer:
       history adapter
       6-step AR rollout with [T -> BiMamba -> M]
       detokenize to proposer trajectory
  -> proposer endpoint / proposer features
  -> refiner:
       shifted history adapter
       6-step AR rollout with [T -> BiMamba -> M]
       residual refinement on proposer trajectory
       future-token refinement head
       detokenize to final trajectory
```

## What Changed Relative to Earlier Versions

Compared with earlier FINet_dev variants, the current codebase has already incorporated these fixes:

- fixed heading concentration assembly to match DONUT semantics
- restored multi-token ego history
- gave proposer and refiner different history windows
- added lightweight stage-specific history encoding
- removed the old runtime `R` block from the active decoder path
- upgraded `T` to relation-aware temporal attention
- upgraded `M` to relation-aware mode attention
- added a refiner-only future-token refinement branch

## Training Defaults

Current defaults in [conf/config.yaml](conf/config.yaml):

```yaml
batch_size: 160
lr: 2e-4
weight_decay: 1e-4
warmup_epochs: 5
gradient_clip_val: 1.0
gradient_clip_algorithm: norm
epochs: 60
t_per_tok: 10
dec_layer_1: 2
dec_layer_2: 2
```

Important:

- `batch_size` is **per-device batch size**
- the repo currently defaults to `gpus: 1`

## Gradient Monitoring

The trainer now logs gradient-health diagnostics to help identify instability:

- `train/global_grad_norm`
- `train/proposer_bimamba_grad_norm`
- `train/refiner_bimamba_grad_norm`
- `train/grad_clip_indicator`
- `train/grad_clip_ratio`

`grad_clip_indicator` is logged on both step and epoch. Its epoch mean can be used as an approximate clip frequency.

## File Guide

### Main files

- `src/model/model_forecast.py`
  - top-level model
  - scene encoder
  - mode initialization
  - proposer/refiner wiring
- `src/model/layers/donut_decoder.py`
  - tokenizer / detokenizer
  - relation-aware `T`
  - BiMamba decoder core
  - relation-aware `M`
  - proposer/refiner stages
  - future-token refinement
- `src/model/trainer_forecast.py`
  - losses
  - metrics
  - gradient diagnostics
- `src/model/layers/coordinate_transforms.py`
  - local/global trajectory transforms
- `src/model/layers/fourier_embedding.py`
  - Fourier feature embedding for continuous geometry inputs

## Open Issues

The main unresolved items are now:

1. Further hyperparameter tuning
   - especially validating `batch_size / lr / warmup / clip` on the actual training environment

2. Deciding whether an explicit road-geometry module is still necessary
   - the current design intentionally relies on BiMamba and refiner query-scene refinement instead of a standalone runtime `R`

## Installation

```bash
conda create -n FINet python=3.10
conda activate FINet
pip install -r requirements.txt
cd mamba_modules/causal-conv1d
pip install -v --no-build-isolation .
cd ../mamba
pip install -v --no-build-isolation .
```

## Data Preparation

Download Argoverse 2 and preprocess it:

```bash
python preprocess.py --data_root=/path/to/data_root -p
```

The default processed data root in the config is:

```text
data/processed
```

## Training and Evaluation

```bash
# Train
python train.py

# Validation / evaluation
python eval.py

# Test for submission
python eval.py gpus=1 test=true
```

## References

```bibtex
@inproceedings{li2025future,
  title={Future-Aware Interaction Network For Motion Forecasting},
  author={Li, Shijie and Liu, Chunyu and Xu, Xun and Yeo, Si Yong and Yang, Xulei},
  booktitle={ICCV},
  year={2025}
}

@inproceedings{knoche2025donut,
  title={{DONUT: A Decoder-Only Model for Trajectory Prediction}},
  author={Knoche, Markus and de Geus, Daan and Leibe, Bastian},
  booktitle={ICCV},
  year={2025}
}
```
