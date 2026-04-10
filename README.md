# FINet + DONUT Decoder: Mamba-based Autoregressive Trajectory Prediction

## Overview

This project is a modified version of **FINet** (Future-Aware Interaction Network), where the original parallel `TimeDecoder` has been replaced with a **DONUT-style autoregressive decoder** that uses **unidirectional Mamba** as the temporal core.

### Base Projects

- **FINet** (ICCV 2025): Mamba-based trajectory prediction with future-aware spatial interaction.
- **DONUT** (ICCV 2025): Decoder-only autoregressive trajectory prediction with tokenization and 4-type attention (Temporal/Road/Social/Mode).

### Motivation

FINet's original decoder generates all 60 future timesteps in parallel (Cross-Attention + bidirectional Mamba). DONUT demonstrates that autoregressive token-level prediction with explicit interaction types achieves strong results. This project combines the strengths of both: FINet's efficient Mamba encoder with DONUT's structured autoregressive decoder.

## Architecture

### Encoder (FINet, with heading-enriched input)

```
Raw Input (AV2)
    |
    +-- Agent History [B, N, 50, 7]
    |     (pos_diff_x, pos_diff_y, vel_diff, valid_mask,
    |      heading_diff, cos(heading), sin(heading))
    |     -> hist_embed_mlp -> Unidirectional Mamba (4 blocks) -> actor_feat [B, N, 128]
    |
    +-- Lane Map -> LaneEmbeddingLayer (Conv1d) -> lane_feat [B, M, 128]
    |
    +-- Position/Type Embeddings -> scene context x_encoder [B, 173, 128]
```

### Forward Flow: Round1 -> Proposer -> Round2 -> Refiner

The two-round spatial Mamba is split by the Proposer, enabling the Refiner to receive endpoint-aware scene encoding:

```
x_encoder [B, 173, 128]
    |
    === Round 1: spatial_mamba_round1 ===
    decoder0 predicts coarse endpoint -> sort scene tokens -> biMamba (4 blocks)
    -> x_encoder_r1 [B, 173, 128] + mode_r1 [B, 6, 128]
    |
    === Proposer (6 AR steps, using Round 1 encoding) ===
    -> y_hat [B, 6, 60, 2] + heading_hat [B, 6, 60] + pi [B, 6]
    |
    Pi-weighted endpoint = sum(softmax(pi) * y_hat[:,:,-1,:])  [detached]
    |
    === Round 2: spatial_mamba_round2 ===
    Re-sort by Proposer endpoint -> biMamba (2 blocks)
    -> x_encoder_r2 [B, 173, 128] + mode_r2 [B, 6, 128]
    |
    === Refiner (6 AR steps, using Round 2 encoding) ===
    + feature fusion from proposer
    + residual correction in local coordinate frame
    -> new_y_hat [B, 6, 60, 2] + new_heading_hat [B, 6, 60] + new_pi [B, 6]
```

### Autoregressive Decoder Detail

Each AR step within Proposer/Refiner:

```
For step = 1..6 (each producing 10 timesteps):
    Tokenize (Fourier Embedding, 8-dim input)
      -> pos_delta, velocity, rel_pos, heading, heading_delta, head_vs_motion
      |
    Mamba-T: unidirectional, causal on accumulated token sequence
      |
    CrossAttn-R: attend to scene encoding (agents + lanes)
      |
    ModeAttn-M: self-attention among 6 modes with mode/time embeddings
      |
    Detokenize -> pos_delta + heading_delta + scale + concentration
                  (normal t_per_tok=10 steps + over-prediction t_per_tok=10 steps)
      |
    Local coordinate transform:
      Proposer: cumsum(pos_delta), cumsum(0.3*tanh(head_delta)) -> local_to_global
      Refiner:  proposed_chunk(local) + residual -> local_to_global
      |
    Feed back to next step (detached)
```

### Interaction Design

| Type | Implementation | Role |
|------|---------------|------|
| **T (Temporal)** | Unidirectional Mamba (`bimamba=False`) | Causal temporal modeling over token history |
| **R (Road/Scene)** | `Cross_Block` (MultiheadAttention) | Query scene context (agents + lanes) |
| **S (Social)** | Not included | Already handled by `spatial_mamba` in encoder |
| **M (Mode)** | Self-attention + `mode_emb` + `time_emb` | Differentiate and interact among 6 modes |

### DONUT Features Adopted

| Feature | DONUT | This Project |
|---------|-------|-------------|
| Heading prediction | von Mises distribution | Adopted (VonMisesNLLLoss) |
| Coordinate system | Per-token local (rotate/translate) | Adopted (global_to_local / local_to_global) |
| Over-prediction | Auxiliary shifted prediction | Adopted (shift=t_per_tok, GT offset 10 steps) |
| Fourier Embedding | Learnable sinusoidal for continuous features | Adopted (8-dim tokenizer input) |
| Uncertainty | Cumulative scale/concentration | Adopted (scale cumsum + conc inverse cumsum) |
| Social attention | Graph-based radius search | Skipped (encoder handles it) |

### Detach Strategy (Cascade R-CNN style)

| Data | Detached? | Reason |
|------|-----------|--------|
| Proposer endpoint (for Round 2 sorting) | Yes | Prevent Refiner from manipulating sort order |
| x_encoder_r1 (into Round 2) | No | Shared backbone, both losses optimize encoder |
| y_hat (into Refiner) | Yes | Standard two-stage practice |
| heading_hat (into Refiner) | Yes | Consistent with position detach |
| proposer_feats (into Refiner) | No | Allow gradient flow for feature fusion |

## File Structure

### New Files

- `src/model/layers/coordinate_transforms.py` — `wrap_angle`, `global_to_local`, `local_to_global`
- `src/model/layers/fourier_embedding.py` — Learnable Fourier Embedding (8-dim -> sin/cos -> per-feature MLP -> sum -> 128-dim)
- `src/utils/VonMisesNLLLoss.py` — Von Mises NLL loss for heading prediction (adapted from DONUT)

### Modified Files

- `src/datamodule/av2_dataset.py`:
  - Preserves future heading GT as `target_heading` [N, 60]
  - Computes historical `x_heading_diff` [N, 50]
  - Updated collate_fn for new keys

- `src/model/model_forecast.py`:
  - Encoder input expanded from 4-dim to 7-dim (+ heading_diff, cos/sin heading)
  - `spatial_mamba()` split into `spatial_mamba_round1()` + `spatial_mamba_round2()`
  - `forward()` restructured: Round1 -> Proposer -> Round2 -> Refiner
  - Proposer uses Round 1 encoding, Refiner uses Round 2 encoding
  - `decoder1` removed; Round 2 sorts by pi-weighted Proposer endpoint

- `src/model/layers/donut_decoder.py`:
  - `SimpleTokenizer`: 8-dim features with Fourier Embedding (pos + heading features)
  - `SimpleDetokenizer`: outputs position + heading + scale + concentration, with over-prediction support
  - `AutoregressiveStage`: per-token local coordinate transforms, heading prediction, over-prediction
  - `DonutMambaDecoder`: returns dict with all outputs (normal + over-prediction)

- `src/model/trainer_forecast.py`:
  - Added `VonMisesNLLLoss` for heading regression
  - `cal_loss`: heading reg loss (proposer + refiner) + over-prediction shifted loss
  - `compute_cls_nll`: joint position + heading NLL for mode classification
  - Diagnostics: heading MAE in degrees

### Unchanged

- `src/model/layers/transformer_blocks.py` — `Cross_Block` reused as-is
- `src/model/layers/mamba/vim_mamba.py` — `create_block` reused as-is
- `src/metrics/` — All metric classes unchanged (only use y_hat and pi)

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `embed_dim` | 128 | Feature dimension throughout |
| `t_per_tok` | 10 | Timesteps aggregated per token |
| `future_steps` | 60 | Prediction horizon (6s @ 10Hz) |
| `num_modes` | 6 | Multi-modal trajectory hypotheses |
| `dec_layer_1` | 4 | Mamba blocks in proposer |
| `dec_layer_2` | 4 | Mamba blocks in refiner |
| `enc_layer_1` | 4 | Spatial Mamba blocks (round 1) |
| `enc_layer_2` | 2 | Spatial Mamba blocks (round 2) |
| `over_predict` | 1 | Over-prediction enabled (extra t_per_tok steps) |

## Loss Components

| Loss | Source | Target |
|------|--------|--------|
| `reg_loss_prop` | Proposer best-mode trajectory | Laplace NLL on GT positions |
| `reg_loss_ref` | Refiner best-mode trajectory | Laplace NLL on GT positions |
| `heading_reg_prop` | Proposer best-mode heading | Von Mises NLL on GT heading |
| `heading_reg_ref` | Refiner best-mode heading | Von Mises NLL on GT heading |
| `over_loss` | Both stages' over-prediction | Shifted GT (offset by t_per_tok steps) |
| `cls_loss` | Refiner all-mode NLL (pos + heading) | Mixture classification |
| `ep_reg_loss` | decoder0 endpoint offset | SmoothL1 on GT endpoint |
| `others_reg_loss` | Dense predictor for other agents | SmoothL1 on GT |

## Install

```bash
conda create -n FINet python=3.10
conda activate FINet
pip install -r requirements.txt
cd mamba_modules/causal-conv1d
pip install -v --no-build-isolation .
cd ../mamba
pip install -v --no-build-isolation .
```

## Data Preparation (AV2)

Download data at [Argoverse 2](https://www.argoverse.org/av2.html).

```bash
python preprocess.py --data_root=/path/to/data_root -p
```

## Training and Evaluation

```bash
# Train
python train.py

# Evaluation
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
