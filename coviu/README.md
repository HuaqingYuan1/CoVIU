# CoVIU

`validate_counterfactual_gain.py` 包含训练和 adapter 保存。核心 `sensitive_token_gain_loss()` 对选中 token 的增益 `g = target_log_probs - baseline_log_probs` 施加区间外惩罚：

```text
mean(ReLU(g - upper)^2 + w * ReLU(lower - g)^2)
```

`upper = tolerance`，未指定 `gain_lower_bound` 时 `lower = -tolerance`，`w = under_suppression_weight`。默认主配置目标区间为 `[-0.1, 0.1]`。

从本目录运行；共享数据和模型分别位于仓库根目录的 `data/`、`models/`，训练 YAML 路径以本目录为基准：

```bash
python scripts/build_hard_negative_retain_map.py
bash scripts/run_token_counterfactual.sh
```

构建脚本默认读取 `../data/baseline_data/forget_5/`、`retain_95/` 的训练 Parquet，用本地 CLIP 生成 `results/hard_negative_retain_map_clip_top20.json`，与主训练配置对应。可用 `--help` 指定实际输入；更改划分后须重新构建映射。

默认配置为 `configs/token_counterfactual_augmented_all_lora.yaml`。CLI 可覆盖 YAML：

```bash
bash scripts/run_token_counterfactual.sh configs/token_counterfactual_augmented_all_lora.yaml --seed 42
```

主配置为 seed42。需要其他种子时，可通过 `--seed` 指定，同时区分 `--save-adapter` 和 `--output`，避免覆盖。

本目录仅保留默认主方法配置，不包含可选变体和消融脚本。训练保存 LoRA adapter；所有评估使用 [公共评估入口](../evaluation/README.md)。从仓库根目录运行：

```bash
python evaluation/run_eval.py --method coviu
```
