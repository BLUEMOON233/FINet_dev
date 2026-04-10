# FINet + DONUT: Mamba-based Autoregressive Trajectory Prediction

## Overview

This project combines **FINet**'s efficient Mamba encoder with **DONUT**'s structured autoregressive decoder for multi-modal trajectory prediction on Argoverse 2.

### Base Projects

- **FINet** (ICCV 2025): Mamba-based trajectory prediction with future-aware spatial interaction.
- **DONUT** (ICCV 2025): Decoder-only autoregressive trajectory prediction with tokenization and 4-type attention (Temporal/Road/Social/Mode).

### Core Idea

FINet's original architecture uses a two-round BiMamba encoder for spatial interaction, then a parallel decoder to generate all 60 timesteps at once. DONUT uses autoregressive token-level prediction with TRSM x 2 attention per step.

This project **merges the encoder's spatial BiMamba into the decoder's autoregressive loop** as the S (social/spatial) component, using a `[T-R-BiMamba-M] x 2` attention pattern per AR step. The Proposer and Refiner replace the original Round 1 and Round 2:

- **T (Temporal)**: Cross-attention -- current token queries the accumulated token sequence for temporal context
- **R (Road/Scene)**: Cross-attention -- mode tokens attend to scene encoding (agents + lanes)
- **BiMamba (Social/Spatial)**: Bidirectional Mamba scanning over sorted `[scene_tokens + mode_tokens]`, **updating all tokens** -- provides implicit social + spatial interaction, replacing both DONUT's S and part of R
- **M (Mode)**: Self-attention among 6 modes with learnable mode/time embeddings

## Architecture

### Encoder (lightweight, no BiMamba)

The encoder only extracts per-token local features. All cross-agent spatial interaction is deferred to the decoder's BiMamba.

```
Raw Input (AV2)
    |
    +-- Agent History [B, N, 50, 7]
    |     (pos_diff_x, pos_diff_y, vel_diff, valid_mask,
    |      heading_diff, cos(heading), sin(heading))
    |     -> MLP(7->64->128) -> UniMamba (4 blocks) -> actor_feat [B, N, 128]
    |     (temporal encoding only, no cross-agent interaction)
    |
    +-- Lane Map -> LaneEmbeddingLayer (Conv1d) -> lane_feat [B, M, 128]
    |
    +-- Position/Type Embeddings -> x_encoder [B, N+M, 128] -> LayerNorm

decoder0(ego_token) -> ep_offset_1 (coarse endpoint for initial sorting)
mode_tokens = ego_token x 6 + learnable_tokens + ep_tok (FINet-style init)
```

### Forward Flow: Encode -> Proposer -> Refiner

```
x_encoder [B, 173, 128]  (no spatial context yet -- agents don't see each other)
    |
    === Proposer ("new Round 1") ===
    init sort_center = ego_pos + ep_offset_1 (decoder0 coarse endpoint)
    6 AR steps, each: [T-R-BiMamba-M] x 2
    -> BiMamba builds spatial context from scratch at each step
    -> scene_encoding evolves through 6 steps (agents learn about each other)
    -> y_hat [B, 6, 60, 2] + heading_hat + pi + proposer_feats
    |
    proposer_endpoint = softmax(pi) * y_hat[:,:,-1,:] [detached]
    |
    === Refiner ("new Round 2") ===
    init sort_center = proposer_endpoint (Proposer-informed sorting)
    scene_encoding = original x_encoder (independent, fresh start)
    6 AR steps, each: [T-R-BiMamba-M] x 2
    + feature fusion from proposer (proposer_feats, with gradient)
    + residual correction on proposed trajectories (detached)
    -> new_y_hat + new_heading_hat + new_pi
```

### Decoder Detail: Each AR Step

```
For step = 0..5 (each producing 10 timesteps):

    Tokenize (Fourier Embedding, 8-dim input)
      -> pos_delta, velocity, rel_pos, heading, heading_delta, head_vs_motion
      |
    [T-R-BiMamba-M] x 2:
      |
      T: tok(Q) x token_seq(KV)  -- cross-attention
         current token queries [hist, tok_0, ..., tok_k] for temporal context
         (no causal mask needed -- seq only contains past/current steps)
      |
      R: tok(Q) x scene_encoding(KV)  -- cross-attention
         mode tokens attend to scene for road/agent information
      |
      BiMamba: sort scene by pi-weighted prediction position
               -> cat([sorted_scene, tok]) -> BiMamba bidirectional scan
               -> update BOTH scene and mode tokens (social interaction)
               -> scatter restore scene to original order
      |
      M: self-attention among 6 modes + mode/time embeddings
      |
    Write updated tok back into token_seq for next rep/step
      |
    Detokenize -> pos_delta + heading_delta + scale + concentration
                  (normal 10 steps + over-prediction 10 steps)
      |
    Local coordinate transform:
      Proposer: cumsum(pos_delta), cumsum(0.3*tanh(head_delta)) -> local_to_global
      Refiner:  proposed_chunk(local) + residual -> local_to_global
      |
    Update sort_center = softmax(pi) * positions[:,:,-1,:] for next step
```

### Design Rationale

**Why move BiMamba from encoder to decoder?**

In the original FINet, BiMamba runs in two fixed encoder rounds (Round 1 + Round 2) that are "blind" to the prediction task. By moving BiMamba into the decoder's AR loop:

1. Spatial interaction is **prediction-aware** from the first step
2. Scene tokens adapt dynamically to the evolving trajectory prediction
3. Sorting updates at every AR step (not just twice), providing finer-grained spatial locality
4. No redundant BiMamba processing (encoder BiMamba + decoder BiMamba was wasteful)
5. Proposer = "new Round 1", Refiner = "new Round 2" -- same two-stage structure, but integrated with prediction

**Why Proposer and Refiner start from the same x_encoder independently?**

- Original FINet chains Round 1 → Round 2 (6 BiMamba layers, short gradient path)
- Chaining Proposer → Refiner would create a 24-layer BiMamba gradient path -- too deep
- Independent starts let the Refiner build its own spatial understanding, unbiased by Proposer errors
- The Refiner still benefits from Proposer through: sort_center (endpoint), proposer_feats (feature fusion), proposed_positions/headings (residual targets)

### Comparison with DONUT and FINet

| | FINet (original) | DONUT | This Project |
|--|------------------|-------|-------------|
| **Encoder** | BiMamba spatial (2 rounds) | QCNet (Transformer) | Lightweight (no BiMamba) |
| **Decoder type** | Parallel BiMamba | Autoregressive Transformer | Autoregressive [T-R-BiMamba-M] |
| **Temporal (T)** | BiMamba (all 60 steps) | Transformer attention + relation emb | Cross-attention (tok→seq) |
| **Road (R)** | Cross-attention per layer | Transformer attention + relation emb | Cross-attention per rep |
| **Social (S)** | Implicit in encoder BiMamba | Explicit Transformer attention | BiMamba in decoder (scene updated) |
| **Mode (M)** | Output logits only | Transformer + embeddings | Self-attention + embeddings |
| **Depth per step** | N/A (parallel) | TRSM x 2 | [T-R-BiMamba-M] x 2 |
| **History tokens** | 1 (compressed) | 5 (tokenized from 50 steps) | 1 (compressed) |
| **Relation embedding** | None | FourierEmbed on pairwise geometry | None |
| **Heading prediction** | No | Von Mises NLL | Von Mises NLL |
| **Over-prediction** | No | Yes | Yes |
| **Position loss** | SmoothL1 | Laplace NLL | Laplace NLL |
| **Classification** | CrossEntropy | Mixture NLL | Mixture NLL |

### Features Adopted from DONUT

| Feature | Implementation |
|---------|---------------|
| Heading prediction | Von Mises distribution (VonMisesNLLLoss) |
| Per-token local coordinates | global_to_local / local_to_global per AR step |
| Over-prediction | Auxiliary shifted-GT loss (offset by t_per_tok steps) |
| Fourier Embedding | Learnable sinusoidal for 8-dim tokenizer input |
| Uncertainty modeling | Cumulative scale (position) + inverse cumulative concentration (heading) |
| Mixture NLL classification | -logsumexp(log_pi - NLL) on refiner output |

### Detach Strategy

| Data | Detached? | Reason |
|------|-----------|--------|
| Proposer endpoint (for Refiner sorting) | Yes | Prevent Refiner from manipulating sort order |
| y_hat, heading_hat (into Refiner) | Yes | Standard two-stage practice |
| proposer_feats (into Refiner) | No | Allow gradient flow for feature fusion |
| ep_offset_1 (for Proposer sorting) | Yes | Sorting is non-differentiable guidance only |

## TODO

### 1. Multi-token History (from DONUT)

**Current**: 50 history steps are compressed into 1 token via UniMamba (`ego_feat`). The T attention only sees a single history vector.

**DONUT**: Tokenizes 50 steps into 5 history tokens (10 steps each), preserving temporal structure. T attention can selectively attend to different parts of the history (recent vs. older).

**Plan**: Keep UniMamba encoding for each agent, but additionally extract 5 intermediate tokens (one per 10-step chunk) to form a richer history sequence for T. This would change `token_seq` from `[hist, tok_0, ..., tok_k]` (max 7) to `[hist_0, ..., hist_4, tok_0, ..., tok_k]` (max 11), matching DONUT.

### 2. Geometric Relation Embedding (from DONUT)

**Current**: R (CrossAttn) and BiMamba use standard attention / Mamba scanning without pairwise geometric relation information. Spatial awareness comes only from position embeddings on tokens and BiMamba's sorting order.

**DONUT**: Every attention type (T/R/S/M) injects pairwise geometric relations via FourierEmbedding:
```
relation = FourierEmbed([distance, direction, relative_heading])  -> 128d
K = K + Linear_k(relation)   # injected into Key
V = V + Linear_v(relation)   # injected into Value
```

This provides explicit "who is where, facing which direction" information that pure position embeddings cannot capture (e.g., distinguishing "car ahead approaching" vs "car behind following").

**Plan**: Add a `RelationalCrossBlock` for R attention that computes pairwise geometric relations between mode tokens and scene tokens, then injects them into K and V via FourierEmbedding. BiMamba cannot directly support relation embeddings (not attention-based), but R with relations + BiMamba with sorting provides complementary coverage.

## File Structure

### Key Files

| File | Role |
|------|------|
| `src/model/model_forecast.py` | Main model: lightweight encoder + decoder0 + Proposer/Refiner calls |
| `src/model/layers/donut_decoder.py` | `AutoregressiveStage` ([T-R-BiMamba-M]x2), `DonutMambaDecoder` (proposer+refiner) |
| `src/model/trainer_forecast.py` | Loss computation (Laplace + VonMises + Mixture NLL + over-prediction) |
| `src/model/layers/coordinate_transforms.py` | `wrap_angle`, `global_to_local`, `local_to_global` |
| `src/model/layers/fourier_embedding.py` | Learnable Fourier Embedding (8-dim -> 128-dim) |
| `src/model/layers/transformer_blocks.py` | `Cross_Block` (reused for both T and R attention) |
| `src/model/layers/mamba/vim_mamba.py` | `create_block` (BiMamba / UniMamba) |
| `src/utils/LaplaceNLLLoss.py` | Laplace NLL for position regression |
| `src/utils/VonMisesNLLLoss.py` | Von Mises NLL for heading regression |

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `embed_dim` | 128 | Feature dimension throughout |
| `t_per_tok` | 10 | Timesteps aggregated per token |
| `future_steps` | 60 | Prediction horizon (6s @ 10Hz) |
| `num_modes` | 6 | Multi-modal trajectory hypotheses |
| `dec_layer_1` | 2 | [T-R-BiMamba-M] repetitions in proposer |
| `dec_layer_2` | 2 | [T-R-BiMamba-M] repetitions in refiner |
| `over_predict` | 1 | Over-prediction enabled |
| `lr` | 0.002 | Learning rate (AdamW) |
| `weight_decay` | 0.01 | L2 regularization |
| `gradient_clip_val` | 5.0 | Gradient norm clipping |
| `batch_size` | 256 | Training batch size |
| `warmup_epochs` | 10 | Linear LR warmup |

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
