# MLLMU-Bench Baselines

包含 GA、GA-Diff、NPO、Manifold、ViKeR 的训练代码。训练目标和超参数沿用原代码，模型与训练数据默认读取仓库根目录的 `models/`、`data/baseline_data/`。启动脚本会切换到本目录，YAML 中的相对路径以本目录为基准。

GA、GA-Diff、NPO、Manifold 的代码基于 MMUnlearner；ViKeR 是参考 ViKeR 方法编写的实现，并非以官方代码名义发布。各方法的遗忘前完整基础模型均使用 MLLMU-Bench 的 vanilla checkpoint，配置中的 `vanilla_dir` 应指向该模型；NPO 的 `oracle_model_id` 默认也指向该模型。准确的论文、上游仓库引用与许可信息仍需在公开发布时补齐。

从仓库根目录运行：

```bash
bash baseline/scripts/run_mllmu_baseline.sh ga
bash baseline/scripts/run_mllmu_baseline.sh gadiff
bash baseline/scripts/run_mllmu_baseline.sh npo

# Manifold 先生成 mask，再训练
bash baseline/scripts/run_mllmu_gen_mask.sh
bash baseline/scripts/run_mllmu_baseline.sh manifold

# ViKeR 先准备并配置 data/celeba/img_align_celeba 参考图片
bash baseline/scripts/run_mllmu_baseline.sh viker
```

`ga_diff` 也是 GA-Diff 的有效名称。统一入口支持指定 YAML 并覆盖参数：

```bash
bash baseline/scripts/run_mllmu_baseline.sh ga config/mllmu_ga.yaml --lr 1e-5
```

数据准备完成后可使用 `run_mllmu_baselines_all.sh` 顺序运行全部方法。训练日志在本目录 `logs/`，adapter 输出在配置指定的 `saved_model/`。

Manifold 的 mask 生成需要原始 benchmark 的 `Full_Set`，默认生成 `masks/forget10/mllmu_both_mask_weight_only.pt`；基础模型和数据划分须与训练一致。NPO 的 `oracle_model_id` 默认指向遗忘前完整模型。`data_process/CLEAR_process.py` 被多个训练入口用作共享依赖，因此保留，未包含 CLEAR 训练配置。

默认 GA、GA-Diff、Manifold、ViKeR 使用 10% 遗忘划分，NPO 使用 15%。比较前需对齐划分、基础模型和训练预算；GA 的原始梯度累积步数固定为 4。

这里不再保留独立评估器。所有方法都使用 [公共评估入口](../evaluation/README.md)：

```bash
python evaluation/run_eval.py --method ga
```
