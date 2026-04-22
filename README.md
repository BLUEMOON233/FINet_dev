 # FINet_dev

 Experimental Argoverse 2 motion forecasting codebase that combines:

 - FINet-style scene encoding and Mamba interaction
 - DONUT-style tokenized autoregressive decoding
 - two-stage prediction: proposer -> refiner

 ## Current model summary

 The active decoder path is:

 ```text
 T -> BiMamba -> M
 ```

 - `T`: relation-aware temporal attention over history and generated future tokens
 - `BiMamba`: scene/mode interaction over `[sorted scene tokens ; mode tokens]`
 - `M`: relation-aware mode attention

 The runtime no longer uses a separate decoder `R` block.

 ### High-level flow

 ```text
 AV2 input
   -> agent history encoder (uni-Mamba)
   -> lane encoder
   -> scene token assembly
   -> FINet-style mode initialization
   -> ego history chunking
   -> proposer: 6-step AR rollout with [T -> BiMamba -> M]
   -> refiner: 6-step AR rollout + future-token refinement head
   -> final trajectories and uncertainties
 ```

 ### Key design points

 - shared scene memory for proposer and refiner
 - tokenized history with `t_per_tok = 10`
 - 60 future steps decoded as 6 autoregressive future tokens
 - proposer initializes from mode tokens plus history bias
 - refiner uses a shifted history window and residual refinement over proposer output
 - refiner sorting uses the proposer top-1 endpoint instead of a soft endpoint average
 - proposer and refiner share the proposer-selected target mode during regression training
 - refiner includes one extra future-token refinement pass after rollout

 ## Main files

 - `src/model/model_forecast.py`: top-level model and proposer/refiner wiring
 - `src/model/layers/donut_decoder.py`: tokenizer, detokenizer, decoder stages, refinement
 - `src/model/trainer_forecast.py`: losses, metrics, training logic
 - `src/model/layers/coordinate_transforms.py`: local/global transforms
 - `src/model/layers/fourier_embedding.py`: continuous geometry embeddings

 ## Training defaults

 Current defaults in `conf/config.yaml`:

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

 Notes:

 - `batch_size` is per-device
 - the repo currently defaults to `gpus: 1`

 ## Gradient diagnostics

 The trainer logs:

 - `train/global_grad_norm`
 - `train/proposer_bimamba_grad_norm`
 - `train/refiner_bimamba_grad_norm`
 - `train/grad_clip_indicator`
 - `train/grad_clip_ratio`

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

 ## Data preparation

 ```bash
 python preprocess.py --data_root=/path/to/data_root -p
 ```

 Default processed data root:

 ```text
 data/processed
 ```

 ## Train / eval

 ```bash
 # Train
 python train.py

 # Validation / evaluation
 python eval.py

 # Test for submission
 python eval.py gpus=1 test=true
 ```

 ## Current focus

 The main open work is still decoder-side iteration:

 1. validate shared-mode supervision and top-1 refiner sorting with short continuation runs
 2. confirm gains hold under longer training or another seed
 3. continue tuning decoder interaction and ranking behavior

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