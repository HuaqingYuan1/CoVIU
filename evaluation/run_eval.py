"""Configure and run the shared MLLMU-Bench zero-shot evaluator.

All relative paths in YAML and CLI overrides are relative to evaluation/.
Heavy model dependencies are imported only after configuration validation.
"""

import argparse
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
SPLITS = ("forget", "test", "retain", "real")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--method", help="Profile in config.yaml: coviu, ga, gadiff, npo, manifold, viker, base")
    parser.add_argument("--base-model-path")
    parser.add_argument("--processor-path")
    parser.add_argument("--adapter-path")
    parser.add_argument("--forget-ratio", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--splits", nargs="+", choices=SPLITS)
    parser.add_argument("--name", help="Single directory/file label for this experiment")
    parser.add_argument("--output-dir", help="Override the full output directory")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without requiring data or GPU")
    parser.add_argument("--check-only", action="store_true", help="Validate local input paths without loading model dependencies")
    return parser


def local_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def resolve_arguments(cli):
    config_path = cli.config.expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping")
    unknown = set(cfg) - {"method", "model", "data", "eval", "output", "runtime", "profiles"}
    if unknown:
        raise ValueError("Unknown config sections: " + ", ".join(sorted(unknown)))
    model = cfg.get("model", {})
    data = cfg.get("data", {})
    protocol = cfg.get("eval", {})
    output = cfg.get("output", {})
    runtime = cfg.get("runtime", {})
    profiles = cfg.get("profiles", {})
    method = cli.method or cfg.get("method", "coviu")
    if method == "ga_diff":
        method = "gadiff"
    if method not in profiles:
        raise ValueError(f"Unknown method {method!r}; choose from: {', '.join(profiles)}")
    profile = profiles[method]
    ratio = cli.forget_ratio if cli.forget_ratio is not None else profile.get("forget_ratio", data.get("forget_ratio"))
    if isinstance(ratio, bool) or not isinstance(ratio, int) or not 0 < ratio < 100:
        raise ValueError("forget_ratio must be an integer between 1 and 99")
    seed = cli.seed if cli.seed is not None else protocol.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    splits = cli.splits if cli.splits is not None else protocol.get("splits", list(SPLITS))
    if not isinstance(splits, list) or not splits or any(s not in SPLITS for s in splits):
        raise ValueError("eval.splits must be a nonempty list of forget, test, retain, real")
    # Canonical order keeps random test-image selection independent of YAML split ordering.
    splits = [split for split in SPLITS if split in splits]
    merge_adapter = model.get("merge_adapter", True)
    if not isinstance(merge_adapter, bool):
        raise ValueError("model.merge_adapter must be a YAML boolean")
    adapter = cli.adapter_path if cli.adapter_path is not None else profile.get("adapter_path")
    base_model = cli.base_model_path or model.get("base_model_path")
    processor = cli.processor_path or model.get("processor_path")
    if not base_model or not processor or not data.get("data_root"):
        raise ValueError("model.base_model_path, model.processor_path and data.data_root are required")
    model_id = model.get("model_id", "llava-hf/llava-1.5-7b-hf")
    if not model_id.startswith("llava"):
        raise ValueError("This shared entry point currently supports LLaVA models only")
    label = cli.name or f"{method}_forget{ratio}_seed{seed}"
    if label in {".", ".."} or any(c in label for c in '/\\:*?"<>|'):
        raise ValueError("name must be a single valid filename label")
    data_root = local_path(data["data_root"])
    output_dir = local_path(cli.output_dir) if cli.output_dir else local_path(output.get("result_root", "results")) / label
    args = argparse.Namespace(
        model_id=model_id,
        processor_id=str(local_path(processor)),
        cache_path=str(local_path(base_model)),
        adapter_path=str(local_path(adapter)) if adapter else None,
        no_merge_adapter=not merge_adapter,
        save_merged_model_path=None,
        merge_only=False,
        pretrain=False,
        few_shot_data=None,
        data_split_folder=str(data_root),
        test_data=str(local_path(data["test_data"])) if data.get("test_data") else str(data_root / "Test_Set"),
        celebrity_data=str(local_path(data["celebrity_data"])) if data.get("celebrity_data") else str(data_root / "Retain_Set" / "train-00000-of-00001.parquet"),
        output_folder=str(output_dir),
        output_file=label,
        forget_ratio=ratio,
        seed=seed,
        eval_list=splits,
    )
    manifest = {
        "method": method,
        "config": str(config_path),
        "protocol": "mllmubench_zero_shot_v1",
        "few_shots_num": 0,
        "merge_adapter": merge_adapter,
        "cuda_visible_devices": cli.cuda_visible_devices if cli.cuda_visible_devices is not None else runtime.get("cuda_visible_devices"),
        "arguments": vars(args).copy(),
    }
    return args, manifest


def validate_inputs(args):
    required_dirs = [Path(args.cache_path), Path(args.processor_id)]
    if args.adapter_path:
        adapter = Path(args.adapter_path)
        required_dirs.append(adapter)
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing adapter config: {adapter / 'adapter_config.json'}")
        if not any((adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
            raise FileNotFoundError(f"Missing adapter weights under {adapter}")
    for path in required_dirs:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing local model/processor directory: {path}")
    required_files = []
    data_root = Path(args.data_split_folder)
    if "forget" in args.eval_list or "test" in args.eval_list:
        required_files.append(data_root / f"forget_{args.forget_ratio}" / "train-00000-of-00001.parquet")
    if "retain" in args.eval_list:
        required_files.append(data_root / f"retain_{100 - args.forget_ratio}" / "train-00000-of-00001.parquet")
    if "real" in args.eval_list:
        required_files.append(Path(args.celebrity_data))
    if "test" in args.eval_list:
        test = Path(args.test_data)
        if not (test.is_file() or (test.is_dir() and any(test.glob("*.parquet")))):
            raise FileNotFoundError(f"Missing test Parquet data: {test}")
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation Parquet file: {path}")


def main(argv=None):
    parser = build_parser()
    cli = parser.parse_args(argv)
    try:
        args, manifest = resolve_arguments(cli)
        if not cli.dry_run:
            validate_inputs(args)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if cli.dry_run or cli.check_only:
        return
    device = manifest["cuda_visible_devices"]
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from eval import main as evaluate

    destination = Path(args.output_folder)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "evaluation_config.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    evaluate(args)


if __name__ == "__main__":
    main()
