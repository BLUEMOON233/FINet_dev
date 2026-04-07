# FINet 结构化 Future Modes 改造说明（覆盖版）

更新时间：2026-04-07

本 README 为本次改造工作的**完整说明**，已覆盖旧内容。

## 改造目标

在不推翻 FINet 主干结构的前提下，将 `future token generator` 从：

- `focal token + fixed learnable tokens`

升级为：

- `goal-conditioned + branch-conditioned + social-response-conditioned`

即结构化未来模式生成（Structured Future Modes）。

## 已完成工作

### 1) 新增 Structured Mode Generator 模块

新增文件：`src/model/layers/structured_mode_generator.py`

包含 4 个核心类：

- `GoalProposalHead`
  - 输入：focal feature + lane feature/center/attr/angle
  - 输出：`top_ng=3` 个 goals（`goal_feat / goal_scores / goal_xy`）
- `BranchPooler`
  - 基于 goal 与 lane 上下文做软分支池化
  - 输出：`branch_feat / branch_scores`
- `SocialResponseHead`
  - 基于 goal/branch + 邻车特征生成 2 个 social responses
  - 输出：`social_embed / social_logits`
- `StructuredFutureModeGenerator`
  - 组合 `h_f, g_m, b_m, s_m` 生成 `K=6` mode tokens
  - 公式：`q_m = proj([h_f, g_m, b_m, s_m])`

### 2) 接入 ModelForecast（最小侵入式）

修改文件：`src/model/model_forecast.py`

- 删除原固定 token 参数：
  - `self.tokens = nn.Parameter(torch.randn(1, 6, 128))`
- 新增：
  - `self.mode_generator = StructuredFutureModeGenerator(...)`
- 在 `spatial_mamba()` 中替换 `fut_tok` 生成逻辑：
  - 使用 `self.mode_generator(...)` 直接生成 `[B, 6, D]`
  - 第一轮 `decoder0` 的 `ep_offset_1` 已作为 goal proposal bias 接入
- 保留主干流程：
  - `decoder0 / decoder1`
  - 两轮 ARS / 两轮 samba blocks
  - `TimeDecoder` 主体
- 保持原输出接口不变，并新增辅助输出：
  - 新增：`goal_scores / goal_xy / social_logits / fallback_stats`

### 3) 训练侧日志接入

修改文件：`src/model/trainer_forecast.py`

- 新增 `_log_fallback_stats(...)`
- 在 `training_step` 和 `validation_step` 自动记录 fallback 概率日志

## Fallback 监控（新增）

为了监控 fallback 触发频率、判断新增结构模块是否真实使用了有效信息，已新增运行期统计。

### 日志前缀

- `train/fallback/*`
- `val/fallback/*`

### 统计指标

- `batch_*_prob`：当前 batch 的触发概率
- `running_*_prob`：从训练开始到当前 step 的累计触发概率

### 已监控项

- `lane_valid_mask_missing`
- `agent_valid_mask_missing`
- `lane_attr_missing`
- `lane_angles_missing`
- `x_angles_missing`
- `goal_padding_used`（lane 数不足 `top_ng`）
- `social_topk_truncated`（agent 数不足 `social_topk`）
- `no_valid_neighbors`（social 头没有可用邻车）

### 结果解释建议

- 若 `running_*_prob` 长期接近 0：说明 fallback 很少触发，结构化信息链路基本有效。
- 若某一项长期偏高：说明对应输入字段经常缺失或候选不足，建议优先排查该数据链路。

## 修改文件清单

### 新增

- `src/model/layers/structured_mode_generator.py`

### 修改

- `src/model/model_forecast.py`
- `src/model/trainer_forecast.py`
- `README.md`
