# Unified MLLMU-Bench Evaluation

CoVIU、GA、GA-Diff、NPO、Manifold、ViKeR 共用一套 zero-shot 评估任务、提示词、答案提取和指标。`eval.py` 提取自原 CoVIU 评估代码，`run_eval.py` 负责统一配置、模型选择和输入检查，不依赖两个训练目录中的 Python 模块。

## 使用

在仓库根目录运行：

```bash
python evaluation/run_eval.py --method coviu
python evaluation/run_eval.py --method ga
python evaluation/run_eval.py --method gadiff
python evaluation/run_eval.py --method npo
python evaluation/run_eval.py --method manifold
python evaluation/run_eval.py --method viker
python evaluation/run_eval.py --method base
```

Linux / WSL 也可运行 `bash evaluation/run_eval.sh --method coviu`。`ga_diff` 是 `gadiff` 的别名。`base` 不加载 adapter，用于评估遗忘前完整模型。

先修改 `config.yaml` 的模型、processor、数据和 adapter 路径。**YAML 及路径覆盖参数中的相对路径统一以 `evaluation/` 为基准**，不受调用目录影响；`--config` 指定的配置文件本身则相对于调用目录查找。支持绝对路径。

```bash
# 不要求数据、模型或 PyTorch，查看实际将使用的配置
python evaluation/run_eval.py --method ga --dry-run

# 检查模型目录、adapter 文件与所选评估集是否存在，不加载模型
python evaluation/run_eval.py --method ga --check-only

# 指定训练完成的 adapter，并覆盖对应数据比例
python evaluation/run_eval.py --method ga --adapter-path ../baseline/saved_model/forget5_ga --forget-ratio 5

# 仅评估指定划分
python evaluation/run_eval.py --method coviu --splits forget retain --seed 42
```

默认输出到 `evaluation/results/<method>_forget<ratio>_seed<seed>/`。同名运行会覆盖原结果；不同 checkpoint 可用 `--name coviu_epoch1` 或 `--output-dir` 指定单独目录。该目录包含运行时解析的 `evaluation_config.json`、详细任务结果、accuracy summary 和 focus metrics。

## 统一协议

- 分类与填空：分别报告图文和纯文本准确率。
- 生成：沿用原代码的 BLEU 和 ROUGE 指标及答案提取规则。
- 所有方法都使用 zero-shot 提示词，不加载 few-shot 示例。
- 使用同一份基础模型和 processor 配置；基础模型以 FP16 加载，评估前调用 `model.eval()`。
- 默认先加载基础模型，再加载 adapter 并在内存中合并，不额外保存完整模型。`model.merge_adapter` 为所有方法共用的设置。
- 所有方法使用同一随机种子和固定任务顺序，测试集图片选择沿用原代码。生成使用原 LLaVA 分支的 `do_sample=False`、`max_new_tokens=50`。

目前公共入口支持本地 LLaVA 模型和 CUDA；并未扩展为 CPU 或其他模型架构的通用评估器。其他架构的旧任务分支保留在任务函数中，不通过公共入口启用。

## 数据与公平比较

读取**原始 benchmark 任务 Parquet**，不是训练用的 `metadata` QA Parquet：

```text
data/mllmu-bench/
├── forget_5/train-00000-of-00001.parquet
├── retain_95/train-00000-of-00001.parquet
├── Test_Set/               # 测试任务 Parquet，含原代码要求的 images 等字段
└── Retain_Set/train-00000-of-00001.parquet
```

数据默认位于仓库根目录下的 `data/mllmu-bench`。`forget` / `retain` 目录中的比例由配置决定；`test` 也需要对应 forget 文件来筛选遗忘身份。`real` 使用 `Retain_Set`。zero-shot 不需要读取 `Full_Set`。

`profiles` 中暂时按现有训练配置设置：CoVIU 为 5%，NPO 为 15%，其他 baseline 为 10%。**这些默认配置不能直接作为同一比例的公平对比**；请先用一致划分训练各方法，再同时设置 adapter 路径与真实比例。只修改评估比例不会改变 adapter 的训练划分。

## 验证与迁移

依赖见 `requirements.txt`，请先安装匹配机器 CUDA 的 PyTorch；依赖约束不是经过 GPU 验证的环境锁定文件。

```bash
python -m unittest discover -s evaluation/tests -v
```

本代码包只有本目录这一套评估实现，`coviu/` 和 `baseline/` 均不包含旧评估入口。既有实验结果不会因为整理代码而自动变成统一协议下的结果，比较时请使用本入口重新评估。
