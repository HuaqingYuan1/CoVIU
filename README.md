# CoVIU: MLLMU-Bench Unlearning

CoVIU 方法、五种 baseline 与统一的 MLLMU-Bench zero-shot 评估。此目录是可独立上传 GitHub 的代码包，不依赖外部旧代码目录；数据和模型需要另行准备。

```text
coviu_git/
├── coviu/                  # CoVIU 训练代码、配置和数据准备脚本
├── baseline/               # GA、GA-Diff、NPO、Manifold、ViKeR
├── evaluation/             # 所有方法共用的唯一评估实现
├── requirements.txt
├── .gitignore
└── README.md
```

未包含消融、旧评估入口、缓存、日志、模型、数据或整理前的归档。

## 代码与模型来源

- GA、GA-Diff、NPO、Manifold 的 baseline 代码基于 MMUnlearner，并适配本项目的实验配置。
- ViKeR 是参考 ViKeR 方法编写的实现，不作为 ViKeR 官方代码发布。
- 遗忘前的基础模型使用 MLLMU-Bench 提供的 vanilla checkpoint。配置中的 `models/LLaVA_Vanilla/` 应指向该完整模型；`models/llava-1.5-7b-hf/` 用于对应的 processor/tokenizer。
- `coviu/` 包含本项目的 CoVIU 方法，`evaluation/` 为各方法共用的评估入口。

## 环境与共享数据

建议使用 Linux / WSL、Python 3.10 和 CUDA GPU。先安装匹配本机 CUDA 的 PyTorch，再从本目录安装：

```bash
pip install -r requirements.txt
```

依赖约束根据现有代码整理，尚未通过 GPU 训练和完整评估验证，不是原实验环境锁定文件。

在本目录下准备共享数据与模型，或修改 YAML 指向实际位置：

```text
data/
├── baseline_data/          # 训练用 QA Parquet：forget_5、retain_95 等
├── mllmu-bench/            # 原始评估任务 Parquet
└── celeba/img_align_celeba/ # ViKeR 参考图片
models/
├── LLaVA_Vanilla/          # 遗忘前的完整基础模型
├── llava-1.5-7b-hf/        # processor/tokenizer
└── clip-vit-base-patch32/  # CoVIU hard-negative 映射构建
```

训练数据需要含图片数据的 `image` 和 JSON QA 列表 `metadata`，每项含 `Question`、`Answer`。评估读取原始 `Classification_Task`、`Mask_Task`、`Generation_Task` 等任务字段，两者不能直接互换。当前代码包不含原始数据到训练 QA Parquet 的完整转换入口。

## CoVIU 训练

在本目录运行：

```bash
cd coviu
python scripts/build_hard_negative_retain_map.py
bash scripts/run_token_counterfactual.sh
cd ..
```

默认主配置为 `coviu/configs/token_counterfactual_augmented_all_lora.yaml`，增益目标区间为 `[-0.1, 0.1]`，训练保存 LoRA adapter。详见 [CoVIU 说明](coviu/README.md)。

## Baseline 训练

```bash
bash baseline/scripts/run_mllmu_baseline.sh ga
bash baseline/scripts/run_mllmu_baseline.sh gadiff
bash baseline/scripts/run_mllmu_baseline.sh npo
bash baseline/scripts/run_mllmu_gen_mask.sh
bash baseline/scripts/run_mllmu_baseline.sh manifold
bash baseline/scripts/run_mllmu_baseline.sh viker
```

Manifold 需要先生成 mask，ViKeR 需要先准备参考图片。详见 [Baseline 说明](baseline/README.md)。

## 统一评估

先修改 `evaluation/config.yaml` 的基础模型、processor、数据和各方法 adapter 路径，再从本目录运行：

```bash
python evaluation/run_eval.py --method coviu
python evaluation/run_eval.py --method ga
python evaluation/run_eval.py --method gadiff
python evaluation/run_eval.py --method npo
python evaluation/run_eval.py --method manifold
python evaluation/run_eval.py --method viker
python evaluation/run_eval.py --method base
```

`base` 评估遗忘前模型；其他方法统一加载对应 adapter，默认在内存中合并后评估，不额外保存完整模型。任务、提示词、指标和随机种子设置共用一份实现。详见 [评估说明](evaluation/README.md)。

**原训练默认比例不同：CoVIU 为 5%，NPO 为 15%，其他 baseline 为 10%。正式比较前须使用相同的数据划分与基础模型重新训练，并同步调整配置中的 adapter 路径和遗忘比例。** 仅修改评估比例不会改变模型训练使用的数据。

## 不加载模型的检查

```bash
python evaluation/run_eval.py --method coviu --dry-run
python evaluation/run_eval.py --method ga --check-only
python -m unittest discover -s evaluation/tests -v
python coviu/validate_counterfactual_gain.py --self-test
```

`--dry-run` 查看解析后的配置；`--check-only` 检查本地输入路径。数学自检在未安装 PyTorch 时只验证标量部分。

## 发布

仅将 `coviu_git/` 作为仓库根目录上传。`.gitignore` 排除数据、权重、adapter、mask、日志与实验输出。此目录保留原训练目标及超参数；只调整目录、路径与公共评估入口。来源见上文；公开发布前还需补充对应论文和仓库的准确引用，并保留上游要求的版权与许可信息。本代码包未代为声明许可证。
