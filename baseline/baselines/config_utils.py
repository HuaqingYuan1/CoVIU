import argparse
import os


def load_clear_split(data_dir, split_name, forget_ratio):
    """Load a local CLEAR split using forget05/retain95 benchmark naming."""
    from datasets import load_dataset, load_from_disk

    ratio = forget_ratio if split_name == "forget" else 100 - forget_ratio
    suffix = f"{ratio:02d}" if split_name == "forget" else str(ratio)
    path = os.path.join(data_dir, f"{split_name}{suffix}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CLEAR {split_name} split not found: {path}. "
            "Expected directories such as forget05 and retain95."
        )
    try:
        return load_dataset(path, split="train")
    except (ValueError, FileNotFoundError):
        return load_from_disk(path)


def add_dataset_args(parser):
    parser.add_argument("--dataset_type", choices=("mllmu", "clear"), default="mllmu")
    parser.add_argument("--clear_data_dir", type=str, default="data/CLEAR")
    parser.add_argument("--cuda_visible_devices", type=str, default=None)


def subsample_dataset(data, fraction, max_samples, seed):
    """Apply equivalent deterministic sampling to pandas or HF datasets."""
    target = len(data)
    if fraction < 1.0:
        target = max(1, int(target * fraction))
    if max_samples > 0:
        target = min(target, max_samples)
    if target >= len(data):
        return data
    if hasattr(data, "select") and hasattr(data, "shuffle"):
        return data.shuffle(seed=seed).select(range(target))
    return data.sample(n=target, random_state=seed).reset_index(drop=True)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _flatten_yaml_config(value, output, prefix=""):
    if not isinstance(value, dict):
        raise TypeError("The YAML root must be a mapping.")
    for key, item in value.items():
        normalized_key = str(key).replace("-", "_")
        location = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            _flatten_yaml_config(item, output, location)
            continue
        if normalized_key in output:
            raise ValueError(
                f"Duplicate YAML option {normalized_key!r} at {location}."
            )
        output[normalized_key] = item


def _load_yaml_defaults(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML configuration requires PyYAML. Install it with: pip install pyyaml"
        ) from exc

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    flattened = {}
    _flatten_yaml_config(raw, flattened)
    return flattened


def parse_args_with_yaml(parser, required=()):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()

    if bootstrap_args.config:
        yaml_defaults = _load_yaml_defaults(bootstrap_args.config)
        valid_options = {
            action.dest
            for action in parser._actions
            if action.dest not in {"help", "config"}
        }
        unknown = sorted(set(yaml_defaults).difference(valid_options))
        if unknown:
            raise ValueError("Unknown YAML option(s): " + ", ".join(unknown))
        parser.set_defaults(**yaml_defaults)

    args = parser.parse_args()
    args.config = bootstrap_args.config

    missing = [name for name in required if not getattr(args, name, None)]
    if missing:
        raise ValueError(
            "Missing required option(s), either in YAML or CLI: " + ", ".join(missing)
        )
    return args
