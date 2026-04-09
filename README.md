# FINet + DONUT Decoder: Mamba-based Autoregressive Trajectory Prediction

## Overview

This project is a modified version of **FINet** (Future-Aware Interaction Network), where the original parallel `TimeDecoder` has been replaced with a **DONUT-style autoregressive decoder** that uses **unidirectional Mamba** as the temporal core.

### Base Projects

- **FINet** (ICCV 2025): Mamba-based trajectory prediction with future-aware spatial interaction.
- **DONUT** (ICCV 2025): Decoder-only autoregressive trajectory prediction with tokenization and 4-type attention (Temporal/Road/Social/Mode).

### Motivation

FINet's original decoder generates all 60 future timesteps in parallel (Cross-Attention + bidirectional Mamba). DONUT demonstrates that autoregressive token-level prediction with explicit interaction types achieves strong results. This project combines the strengths of both: FINet's efficient Mamba encoder with DONUT's structured autoregressive decoder.

## Architecture

### Unchanged (FINet Encoder)

The entire encoder pipeline is preserved:

```
Raw Input (AV2)
    │
    ├── Agent History ─── hist_embed_mlp ─── Unidirectional Mamba (4 blocks) ─── actor_feat [B, N, 128]
    │
    ├── Lane Map ──────── LaneEmbeddingLayer (Conv1d) ──────────────────────── lane_feat [B, M, 128]
    │
    └── Position/Type Embeddings ──── scene context x_encoder [B, 173, 128]
                                           │
                                    Spatial Mamba (2 rounds)
                                    Round 1: sort by endpoint → biMamba (4 blocks)
                                    Round 2: re-sort → biMamba (2 blocks)
                                           │
                                    x_encoder [B, 173, 128]  +  mode_tokens [B, 6, 128]
```

### Replaced (Autoregressive Decoder)

The original `TimeDecoder` (linspace expansion + Cross-Attention + biMamba) is replaced by `DonutMambaDecoder`:

```
mode_tokens [B, 6, 128]   ego_feat [B, 128]   scene_encoding [B, 173, 128]
         │                      │                        │
         ▼                      ▼                        │
    ┌─── Proposer (6 autoregressive steps) ──────────────┤
    │                                                    │
    │  For step = 1..6:                                  │
    │    Tokenize (10 timesteps → 1 token)               │
    │      ↓                                             │
    │    Mamba-T: unidirectional, on accumulated seq     │
    │      ↓                                             │
    │    CrossAttn-R: attend to scene ←──────────────────┘
    │      ↓
    │    ModeAttn-M: self-attention among 6 modes
    │      ↓
    │    Detokenize → pos_delta [B,6,10,2] + scale [B,6,10,2]
    │      ↓
    │    cumsum → absolute positions
    │      ↓
    │    feed back to next step
    │
    └──→ y_hat [B, 6, 60, 2],  pi [B, 6],  scal [B, 6, 60, 2]
              │
              ▼
    ┌─── Refiner (6 autoregressive steps, same structure) ──────
    │    + feature fusion from proposer
    │    + residual correction on proposed positions
    │
    └──→ new_y_hat [B, 6, 60, 2],  new_pi [B, 6],  scal_new [B, 6, 60, 2]
```

### Interaction Design

| Type | Implementation | Role |
|------|---------------|------|
| **T (Temporal)** | Unidirectional Mamba (`bimamba=False`) | Causal temporal modeling over token history |
| **R (Road/Scene)** | `Cross_Block` (MultiheadAttention) | Query scene context (agents + lanes) |
| **S (Social)** | Not included | Already handled by `spatial_mamba` in encoder |
| **M (Mode)** | Self-attention + `mode_emb` + `time_emb` | Differentiate and interact among 6 modes |

### Simplifications vs DONUT

| Feature | DONUT | This Project |
|---------|-------|-------------|
| Heading prediction | von Mises distribution | Not included |
| Coordinate system | Per-token local (rotate/translate) | Ego-centric (no transform) |
| Over-prediction | Auxiliary shifted prediction | Not included |
| Uncertainty | Cumulative scale/concentration | Direct ELU scale |
| Social attention | Graph-based radius search | Skipped (encoder handles it) |

## File Changes

### New

- `src/model/layers/donut_decoder.py` — Contains 5 classes:
  - `SimpleTokenizer`: 10 position steps → 1 token feature
  - `SimpleDetokenizer`: token feature → 10 position deltas + Laplace scale
  - `ModeAttention`: self-attention among modes with learnable mode/time embeddings
  - `AutoregressiveStage`: one full stage (Mamba-T + CrossAttn-R + ModeAttn-M, 6 AR steps)
  - `DonutMambaDecoder`: top-level decoder (proposer + refiner)

### Modified

- `src/model/model_forecast.py`:
  - Import changed from `TimeDecoder` to `DonutMambaDecoder`
  - `__init__`: instantiates `DonutMambaDecoder` with `t_per_tok` parameter
  - `forward()`: passes `ego_feat = x_encoder[:, 0]` to decoder instead of linspace expansion
- `conf/config.yaml`: added `t_per_tok: 10`
- `conf/model/model_forecast.yaml`: added `t_per_tok: ${t_per_tok}`

### Deleted

- `src/model/layers/time_decoder.py` — Original parallel decoder

### Unchanged

- `src/model/trainer_forecast.py` — Loss functions, training loop (output dict interface preserved)
- `src/model/layers/transformer_blocks.py` — `Cross_Block` reused as-is
- `src/model/layers/mamba/vim_mamba.py` — `create_block` reused as-is
- All data pipeline, metrics, and utility files

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
