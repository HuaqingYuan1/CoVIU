from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable


DEFAULT_GENDER_QUESTION_PATTERNS = (
    r"\bgender\b",
    r"\bsex\b",
    r"\bmale\s+or\s+female\b",
    r"\bfemale\s+or\s+male\b",
    r"\bman\s+or\s+woman\b",
    r"\bwoman\s+or\s+man\b",
)


@dataclass(frozen=True)
class Profile:
    profile_id: str
    image_value: Any
    qa_pairs: tuple[tuple[str, str], ...]
    variant_image_values: tuple[Any, ...] = ()


@dataclass(frozen=True)
class GainExample:
    sample_id: str
    target_profile_id: str
    question: str
    answer: str
    target_image_value: Any
    counterfactual_profile_ids: tuple[str, ...]
    counterfactual_image_values: tuple[Any, ...]


@dataclass(frozen=True)
class RetainExample:
    """A retain image QA used to preserve vanilla visual-answer behavior."""

    sample_id: str
    profile_id: str
    question: str
    answer: str
    image_value: Any


@dataclass(frozen=True)
class AugmentedImage:
    base_image_value: Any
    seed: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate token-level counterfactual visual identity gain on "
            "MLLMU-Bench. Sensitive answer tokens are selected from their "
            "teacher-forced target-vs-counterfactual log-probability gain."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML configuration file. Explicit CLI arguments override it.",
    )
    parser.add_argument(
        "--forget-parquet",
        type=Path,
        default=Path("../data/baseline_data/forget_5/train-00000-of-00001.parquet"),
    )
    parser.add_argument(
        "--retain-parquet",
        type=Path,
        default=Path("../data/baseline_data/retain_95/train-00000-of-00001.parquet"),
    )
    parser.add_argument(
        "--forget-variant-parquet",
        type=Path,
        default=None,
        help=(
            "Optional parquet file or directory containing extra forget target "
            "views keyed by ID, e.g. the Test_Set with an images column. When "
            "set, each forget QA is optimized on these additional views."
        ),
    )
    parser.add_argument(
        "--forget-variant-max-images",
        type=int,
        default=0,
        help=(
            "Maximum number of variant images to use per forget profile. "
            "0 means use all available variants."
        ),
    )
    parser.add_argument(
        "--forget-augmentations-per-image",
        type=int,
        default=0,
        help=(
            "Number of deterministic train-only augmentations to add for each "
            "canonical forget image. This improves view robustness without "
            "using Test_Set pose images."
        ),
    )
    parser.add_argument(
        "--exclude-canonical-forget-image",
        dest="include_canonical_forget_image",
        action="store_false",
        help=(
            "When --forget-variant-parquet is set, train only on variant "
            "images and exclude the original forget parquet image."
        ),
    )
    parser.set_defaults(include_canonical_forget_image=True)
    parser.add_argument(
        "--model-path",
        type=str,
        help="Vanilla/finetuned LLaVA model path. Required unless --self-test.",
    )
    parser.add_argument(
        "--processor-path",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Processor path or Hugging Face ID.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/counterfactual_gain.json"),
    )
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--counterfactual-count", type=int, default=4)
    parser.add_argument(
        "--counterfactual-sampling",
        choices=("random", "nearest"),
        default="random",
        help=(
            "How counterfactual images are selected. 'nearest' uses "
            "--counterfactual-neighbor-map to take the most visually similar "
            "retain profiles for each forget profile."
        ),
    )
    parser.add_argument(
        "--counterfactual-neighbor-map",
        type=Path,
        default=None,
        help=(
            "JSON file mapping forget profile IDs to visually similar retain "
            "profile IDs. Compatible with scripts/build_hard_negative_retain_map.py."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help=(
            "Allowed absolute target-over-counterfactual token gain. "
            "The target range is [-tolerance, +tolerance]."
        ),
    )
    parser.add_argument(
        "--gain-lower-bound",
        type=float,
        default=None,
        help=(
            "Optional lower bound for target-over-counterfactual gain. When set, "
            "sensitive token gains are optimized into "
            "[gain_lower_bound, tolerance] instead of [-tolerance, tolerance]."
        ),
    )
    parser.add_argument(
        "--under-suppression-weight",
        type=float,
        default=1.0,
        help=(
            "Penalty weight for gain < -tolerance. Use 1.0 for symmetric "
            "band loss; use 0.0 to recover the old one-sided loss."
        ),
    )
    parser.add_argument(
        "--token-selection-mode",
        choices=("sensitive", "all"),
        default="sensitive",
        help=(
            "Which teacher-forced answer tokens receive the counterfactual "
            "gain loss. 'sensitive' uses initial positive-gain selection; "
            "'all' applies the same per-token loss to every answer token."
        ),
    )
    parser.add_argument(
        "--forget-loss-mode",
        choices=("token", "sentence"),
        default="token",
        help=(
            "Forget objective used during LoRA optimization. 'token' uses "
            "the selected answer-token gain loss; 'sentence' uses the scalar "
            "full-answer log-probability gain loss."
        ),
    )
    parser.add_argument(
        "--token-gain-threshold",
        type=float,
        default=0.1,
        help="Minimum positive initial gain for a token to be sensitive.",
    )
    parser.add_argument(
        "--token-gain-quantile",
        type=float,
        default=0.75,
        help=(
            "Select tokens at or above this quantile of positive lexical "
            "token gains. Set to 0 to disable quantile filtering."
        ),
    )
    parser.add_argument(
        "--sensitive-token-min",
        type=int,
        default=1,
        help="Minimum sensitive tokens selected per answer.",
    )
    parser.add_argument(
        "--sensitive-token-max-fraction",
        type=float,
        default=0.3,
        help="Maximum fraction of answer tokens selected as sensitive.",
    )
    parser.add_argument(
        "--sensitive-span-expansion",
        choices=("none", "entity", "content"),
        default="content",
        help=(
            "Expand high-gain anchor tokens. 'entity' expands to nearby "
            "entity-like spans. 'content' expands each anchor to its nearby "
            "semantic content phrase using generic boundary words."
        ),
    )
    parser.add_argument(
        "--exclude-gender-questions-from-training",
        dest="exclude_gender_questions_from_training",
        action="store_true",
        help=(
            "Keep gender questions in evaluation, but skip them when building "
            "the counterfactual forgetting training schedule."
        ),
    )
    parser.add_argument(
        "--include-gender-questions-in-training",
        dest="exclude_gender_questions_from_training",
        action="store_false",
        help=(
            "Do not filter gender/sex visual-attribute questions from the "
            "counterfactual forgetting training schedule."
        ),
    )
    parser.set_defaults(exclude_gender_questions_from_training=False)
    parser.add_argument(
        "--gender-question-patterns",
        nargs="+",
        default=list(DEFAULT_GENDER_QUESTION_PATTERNS),
        help=(
            "Regex patterns used to identify gender/sex visual-attribute "
            "questions for optional training-time filtering."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map value; use 'none' to disable it.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Value assigned to CUDA_VISIBLE_DEVICES before loading PyTorch.",
    )
    parser.add_argument(
        "--local-files-only",
        dest="local_files_only",
        action="store_true",
        help="Do not download model or processor files.",
    )
    parser.add_argument(
        "--no-local-files-only",
        dest="local_files_only",
        action="store_false",
        help="Allow model and processor downloads.",
    )
    parser.set_defaults(local_files_only=False)
    parser.add_argument(
        "--train-steps",
        type=int,
        default=0,
        help=(
            "If positive and --train-epochs is 0, attach LoRA and optimize "
            "L_gain for this many randomly sampled steps."
        ),
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=0,
        help=(
            "If positive, shuffle and traverse every loaded Forget VQA once "
            "per epoch. This takes precedence over --train-steps."
        ),
    )
    parser.add_argument(
        "--checkpoint-epochs",
        nargs="+",
        type=int,
        default=[],
        help=(
            "Optional epoch numbers at which to save intermediate LoRA "
            "adapters during epoch-based training."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--language-learning-rate",
        type=float,
        default=None,
        help=(
            "Optional lower learning rate for LoRA parameters under "
            "language_model. Defaults to --learning-rate."
        ),
    )
    parser.add_argument(
        "--projector-learning-rate",
        type=float,
        default=None,
        help=(
            "Optional learning rate for LoRA parameters under "
            "multi_modal_projector. Defaults to --learning-rate."
        ),
    )
    parser.add_argument(
        "--retain-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the retain answer-token distillation loss. Set above 0 "
            "to preserve vanilla image-conditioned behavior on retain QA."
        ),
    )
    parser.add_argument(
        "--retain-samples-per-step",
        type=int,
        default=1,
        help="Number of retain image QA examples distilled at each training step.",
    )
    parser.add_argument(
        "--retain-sampling",
        choices=("random", "hard_negative"),
        default="random",
        help=(
            "How retain examples are sampled. 'hard_negative' uses "
            "--retain-hard-negative-map to prefer visually similar retain "
            "profiles for each forget profile, with random fallback."
        ),
    )
    parser.add_argument(
        "--retain-hard-negative-map",
        type=Path,
        default=None,
        help=(
            "JSON file mapping forget profile IDs to visually similar retain "
            "profile IDs. Produced by scripts/build_hard_negative_retain_map.py."
        ),
    )
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-module-scope",
        choices=(
            "language",
            "vision",
            "projector",
            "vision_projector",
            "all",
            "custom",
        ),
        default="language",
        help=(
            "Which model component receives LoRA adapters. Non-custom scopes "
            "select Linear layers by module path. Use --lora-target-modules "
            "auto to adapt all Linear layers in the selected scope."
        ),
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["auto"],
        help=(
            "LoRA target module filters. Use 'auto' to target all Linear "
            "layers in --lora-module-scope, or pass names like q_proj v_proj. "
            "With --lora-module-scope custom, these are passed directly to PEFT."
        ),
    )
    parser.add_argument(
        "--save-adapter",
        type=Path,
        default=Path("results/counterfactual_gain_adapter"),
    )
    parser.add_argument(
        "--save-adapter-checkpoint-template",
        type=str,
        default=None,
        help=(
            "Optional format string for intermediate adapter directories, "
            "for example '/tmp/adapter_epoch{epoch}'. Defaults to "
            "'{save_adapter}_epoch{epoch}'."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a CPU-only mathematical test without loading data or a model.",
    )
    return parser


def _flatten_yaml_config(
    value: Any,
    output: dict[str, Any],
    prefix: str = "",
) -> None:
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


def _load_yaml_defaults(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML configuration requires PyYAML. Install it with: pip install pyyaml"
        ) from exc

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    flattened: dict[str, Any] = {}
    _flatten_yaml_config(raw, flattened)
    return flattened


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    bootstrap_args, _ = bootstrap.parse_known_args()

    parser = build_parser()
    if bootstrap_args.config is not None:
        yaml_defaults = _load_yaml_defaults(bootstrap_args.config)
        valid_options = {
            action.dest
            for action in parser._actions
            if action.dest not in {"help", "config"}
        }
        unknown = sorted(set(yaml_defaults).difference(valid_options))
        if unknown:
            raise ValueError(
                "Unknown YAML option(s): " + ", ".join(unknown)
            )
        parser.set_defaults(**yaml_defaults)

    args = parser.parse_args()
    if args.retain_loss_weight < 0:
        raise ValueError("--retain-loss-weight must be non-negative.")
    if args.retain_samples_per_step < 1:
        raise ValueError("--retain-samples-per-step must be at least 1.")
    if args.gain_lower_bound is not None and args.gain_lower_bound > args.tolerance:
        raise ValueError("--gain-lower-bound must be less than or equal to --tolerance.")
    if args.counterfactual_sampling == "nearest" and args.counterfactual_neighbor_map is None:
        raise ValueError(
            "--counterfactual-neighbor-map is required with "
            "--counterfactual-sampling nearest."
        )
    if args.language_learning_rate is not None and args.language_learning_rate < 0:
        raise ValueError("--language-learning-rate must be non-negative.")
    if args.projector_learning_rate is not None and args.projector_learning_rate < 0:
        raise ValueError("--projector-learning-rate must be non-negative.")
    if args.checkpoint_epochs is None:
        args.checkpoint_epochs = []
    if isinstance(args.checkpoint_epochs, int):
        args.checkpoint_epochs = [args.checkpoint_epochs]
    args.checkpoint_epochs = sorted({int(epoch) for epoch in args.checkpoint_epochs})
    if any(epoch < 1 for epoch in args.checkpoint_epochs):
        raise ValueError("--checkpoint-epochs must be positive epoch numbers.")
    if args.checkpoint_epochs and args.train_epochs <= 0:
        raise ValueError("--checkpoint-epochs requires --train-epochs.")
    if args.checkpoint_epochs and max(args.checkpoint_epochs) > args.train_epochs:
        raise ValueError("--checkpoint-epochs cannot exceed --train-epochs.")
    for field in (
        "forget_parquet",
        "forget_variant_parquet",
        "retain_parquet",
        "output",
        "save_adapter",
        "retain_hard_negative_map",
        "counterfactual_neighbor_map",
    ):
        value = getattr(args, field)
        if value is not None and not isinstance(value, Path):
            setattr(args, field, Path(value))
    return args


def is_gender_question(
    question: str,
    patterns: Iterable[str] | None = None,
) -> bool:
    """Return True for gender/sex visual-attribute questions.

    These questions are useful diagnostics, but optimizing them toward a
    counterfactual image can suppress generic visual attributes rather than
    identity-specific associations.
    """
    normalized = re.sub(r"\s+", " ", str(question).strip().lower())
    for pattern in patterns or DEFAULT_GENDER_QUESTION_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return True

    # Covers direct yes/no forms like "Is this person male?" without treating
    # relation words such as father/mother as gender questions.
    if re.search(
        r"\b(is|are|was|were)\b[^?]{0,80}\b(male|female|man|woman)\b",
        normalized,
    ):
        return True
    return False


def question_category(
    question: str,
    patterns: Iterable[str] | None = None,
) -> str:
    if is_gender_question(question, patterns):
        return "visual_attribute_gender"
    return "identity_or_other"


def filter_training_examples(
    examples: list[GainExample],
    args: argparse.Namespace,
) -> tuple[list[GainExample], dict[str, Any]]:
    skipped = [
        example
        for example in examples
        if is_gender_question(example.question, args.gender_question_patterns)
    ]
    if args.exclude_gender_questions_from_training:
        kept = [
            example
            for example in examples
            if not is_gender_question(example.question, args.gender_question_patterns)
        ]
    else:
        kept = list(examples)

    summary = {
        "exclude_gender_questions_from_training": bool(
            args.exclude_gender_questions_from_training
        ),
        "total_examples": len(examples),
        "gender_question_examples": len(skipped),
        "training_examples": len(kept),
        "skipped_gender_sample_ids": [example.sample_id for example in skipped],
    }
    if args.exclude_gender_questions_from_training and not kept:
        raise ValueError(
            "All training examples were filtered out as gender questions. "
            "Disable --exclude-gender-questions-from-training or adjust "
            "--gender-question-patterns."
        )
    return kept, summary


def counterfactual_gain_loss(
    target_score,
    baseline_score,
    tolerance: float,
    gain_lower_bound: float | None = None,
):
    import torch

    gain = target_score - baseline_score
    lower_bound = -tolerance if gain_lower_bound is None else gain_lower_bound
    loss = (
        torch.relu(gain - tolerance).square()
        + torch.relu(lower_bound - gain).square()
    )
    return loss, gain


def sensitive_token_gain_loss(
    target_log_probs,
    baseline_log_probs,
    sensitive_mask,
    tolerance: float,
    under_suppression_weight: float = 1.0,
    gain_lower_bound: float | None = None,
):
    """Penalize selected token gains outside the configured gain band."""
    import torch

    token_gains = target_log_probs - baseline_log_probs
    selected_gains = token_gains[sensitive_mask]
    if selected_gains.numel() == 0:
        raise ValueError("Sensitive-token mask must select at least one token.")
    over_loss = torch.relu(selected_gains - tolerance).square()
    lower_bound = -tolerance if gain_lower_bound is None else gain_lower_bound
    under_loss = torch.relu(lower_bound - selected_gains).square()
    loss = (over_loss + under_suppression_weight * under_loss).mean()
    return loss, token_gains


def run_self_test() -> None:
    targets = [-0.2, -1.5]
    baselines = [-2.0, -1.0]
    gains = [
        target - baseline for target, baseline in zip(targets, baselines)
    ]
    tolerance = 0.1
    losses = [max(0.0, abs(gain) - tolerance) ** 2 for gain in gains]
    loss = sum(losses) / len(losses)
    assert all(
        abs(actual - expected) < 1.0e-6
        for actual, expected in zip(gains, [1.8, -0.5])
    )
    assert abs(loss - 1.525) < 1.0e-6

    print("Scalar self-test passed.")
    print(f"gains={gains} loss={loss:.6f}")

    try:
        import torch
    except ImportError:
        print("PyTorch is not installed in this Python environment; gradient test skipped.")
        return

    target_tensor = torch.tensor(targets, requires_grad=True)
    baseline_tensor = torch.tensor(baselines)
    tensor_losses, tensor_gains = counterfactual_gain_loss(
        target_tensor,
        baseline_tensor,
        tolerance=tolerance,
    )
    tensor_loss = tensor_losses.mean()
    tensor_loss.backward()
    assert torch.allclose(tensor_gains, torch.tensor([1.8, -0.5]))
    assert torch.allclose(tensor_loss, torch.tensor(1.525))
    assert target_tensor.grad is not None
    assert target_tensor.grad[0] > 0
    assert target_tensor.grad[1] < 0
    print(f"Gradient self-test passed: target_grad={target_tensor.grad.tolist()}")

    token_targets = torch.tensor([-0.2, -1.5, -0.4], requires_grad=True)
    token_baselines = torch.tensor([-2.0, -1.0, -0.8])
    sensitive_mask = torch.tensor([True, False, True])
    token_loss, token_gains = sensitive_token_gain_loss(
        token_targets,
        token_baselines,
        sensitive_mask,
        tolerance=tolerance,
        under_suppression_weight=1.0,
    )
    token_loss.backward()
    assert torch.allclose(token_gains, torch.tensor([1.8, -0.5, 0.4]))
    assert torch.allclose(token_loss, torch.tensor(1.49))
    assert token_targets.grad is not None
    assert token_targets.grad[0] > 0
    assert token_targets.grad[1] == 0
    assert token_targets.grad[2] > 0
    print(
        "Sensitive-token gradient self-test passed: "
        f"target_grad={token_targets.grad.tolist()}"
    )

    token_targets_under = torch.tensor([-3.0], requires_grad=True)
    token_baselines_under = torch.tensor([-1.0])
    token_loss_under, token_gains_under = sensitive_token_gain_loss(
        token_targets_under,
        token_baselines_under,
        torch.tensor([True]),
        tolerance=tolerance,
        under_suppression_weight=1.0,
    )
    token_loss_under.backward()
    assert torch.allclose(token_gains_under, torch.tensor([-2.0]))
    assert token_loss_under.item() > 0
    assert token_targets_under.grad is not None
    assert token_targets_under.grad[0] < 0
    print(
        "Under-suppression gradient self-test passed: "
        f"target_grad={token_targets_under.grad.tolist()}"
    )


def _parse_metadata(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        parsed = json.loads(value)
    elif isinstance(value, list):
        parsed = value
    elif hasattr(value, "tolist"):
        parsed = value.tolist()
    else:
        raise TypeError(f"Unsupported metadata type: {type(value).__name__}")
    if not isinstance(parsed, list):
        raise TypeError("Profile metadata must decode to a list.")
    return [item for item in parsed if isinstance(item, dict)]


def _profile_id(row: Any, row_index: Any) -> str:
    for key in ("ID", "id", "unique_id", "profile_id"):
        if key in row and row[key] is not None:
            return str(row[key])
    return str(row_index)


def load_profiles(path: Path) -> list[Profile]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read MLLMU-Bench parquet files.") from exc

    try:
        frame = pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Reading parquet requires pyarrow or fastparquet. "
            "Install the repository requirements first."
        ) from exc

    required = {"image", "metadata"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    profiles: list[Profile] = []
    for row_index, row in frame.iterrows():
        qa_pairs: list[tuple[str, str]] = []
        for qa in _parse_metadata(row["metadata"]):
            question = str(qa.get("Question", "")).strip()
            answer = str(qa.get("Answer", "")).strip()
            if question and answer:
                qa_pairs.append((question, answer))
        if qa_pairs:
            profiles.append(
                Profile(
                    profile_id=_profile_id(row, row_index),
                    image_value=row["image"],
                    qa_pairs=tuple(qa_pairs),
                )
            )
    if not profiles:
        raise ValueError(f"No valid profiles found in {path}")
    return profiles


def load_and_combine_parquet(path: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read parquet files.") from exc

    if path.is_dir():
        parquet_files = sorted(path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {path}")
        return pd.concat(
            [pd.read_parquet(file) for file in parquet_files],
            ignore_index=True,
        )
    return pd.read_parquet(path)


def load_forget_variant_images(
    path: Path,
    max_images_per_profile: int,
) -> dict[str, tuple[Any, ...]]:
    """Load extra target views keyed by forget profile ID."""
    if max_images_per_profile < 0:
        raise ValueError("--forget-variant-max-images must be non-negative.")

    frame = load_and_combine_parquet(path)
    if "ID" not in frame.columns:
        raise ValueError(f"{path} is missing required ID column.")
    if "images" not in frame.columns and "image" not in frame.columns:
        raise ValueError(f"{path} must contain an images or image column.")

    variants: dict[str, list[Any]] = {}
    for _, row in frame.iterrows():
        profile_id = str(row["ID"])
        image_values: list[Any] = []
        if "images" in frame.columns and row.get("images") is not None:
            images_value = row["images"]
            if hasattr(images_value, "tolist"):
                images_value = images_value.tolist()
            image_values.extend(list(images_value))
        elif "image" in frame.columns and row.get("image") is not None:
            image_values.append(row["image"])

        if max_images_per_profile > 0:
            remaining = max_images_per_profile - len(variants.get(profile_id, []))
            if remaining <= 0:
                continue
            image_values = image_values[:remaining]
        variants.setdefault(profile_id, []).extend(image_values)

    return {
        profile_id: tuple(image_values)
        for profile_id, image_values in variants.items()
        if image_values
    }


def attach_forget_variants(
    profiles: list[Profile],
    variant_images_by_id: dict[str, tuple[Any, ...]],
) -> list[Profile]:
    attached: list[Profile] = []
    matched = 0
    total_variants = 0
    for profile in profiles:
        variants = variant_images_by_id.get(profile.profile_id, ())
        if variants:
            matched += 1
            total_variants += len(variants)
        attached.append(
            Profile(
                profile_id=profile.profile_id,
                image_value=profile.image_value,
                qa_pairs=profile.qa_pairs,
                variant_image_values=variants,
            )
        )
    print(
        "Forget target variants: "
        f"matched_profiles={matched}/{len(profiles)}, "
        f"variant_images={total_variants}."
    )
    return attached


def decode_image(value: Any):
    from PIL import Image

    if isinstance(value, AugmentedImage):
        return augment_image(
            decode_image(value.base_image_value),
            value.seed,
        )
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(BytesIO(bytes(value))).convert("RGB")
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value).__name__}")


def augment_image(image, seed: int):
    """Apply deterministic identity-preserving augmentation without test poses."""
    from PIL import ImageEnhance

    rng = random.Random(seed)
    width, height = image.size

    # Face/upper-body biased crop: keep the crop center near the upper middle
    # instead of sampling background-heavy random crops.
    crop_ratio = rng.uniform(0.58, 0.78)
    aspect = rng.uniform(0.82, 1.08)
    crop_width = max(1, min(width, int(width * crop_ratio * math.sqrt(aspect))))
    crop_height = max(1, min(height, int(height * crop_ratio / math.sqrt(aspect))))
    center_x = width * rng.uniform(0.45, 0.55)
    center_y = height * rng.uniform(0.32, 0.44)
    left = int(round(center_x - crop_width / 2))
    top = int(round(center_y - crop_height / 2))
    left = max(0, min(left, width - crop_width))
    top = max(0, min(top, height - crop_height))
    image = image.crop((left, top, left + crop_width, top + crop_height))
    image = image.resize((width, height))

    angle = rng.uniform(-8.0, 8.0)
    image = image.rotate(angle, resample=2, fillcolor=(0, 0, 0))

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.90, 1.10))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.90, 1.15))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.90, 1.10))
    image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.90, 1.20))
    return image.convert("RGB")


def build_examples(
    forget_profiles: list[Profile],
    retain_profiles: list[Profile],
    max_samples: int,
    counterfactual_count: int,
    seed: int,
    include_canonical_forget_image: bool = True,
    forget_augmentations_per_image: int = 0,
    counterfactual_sampling: str = "random",
    counterfactual_neighbor_map: dict[str, list[str]] | None = None,
) -> list[GainExample]:
    if counterfactual_count < 1:
        raise ValueError("--counterfactual-count must be at least 1.")
    if forget_augmentations_per_image < 0:
        raise ValueError("--forget-augmentations-per-image must be non-negative.")
    if not retain_profiles:
        raise ValueError("The retain set is empty.")
    if counterfactual_sampling == "nearest" and counterfactual_neighbor_map is None:
        raise ValueError(
            "--counterfactual-neighbor-map is required when "
            "--counterfactual-sampling nearest is used."
        )
    if counterfactual_sampling not in {"random", "nearest"}:
        raise ValueError("--counterfactual-sampling must be 'random' or 'nearest'.")

    rng = random.Random(seed)
    retain_by_profile = {profile.profile_id: profile for profile in retain_profiles}
    candidates: list[tuple[Profile, int, str, str]] = []
    for profile in forget_profiles:
        for qa_index, (question, answer) in enumerate(profile.qa_pairs):
            candidates.append((profile, qa_index, question, answer))
    rng.shuffle(candidates)
    if max_samples > 0:
        candidates = candidates[:max_samples]

    examples: list[GainExample] = []
    for profile, qa_index, question, answer in candidates:
        target_images: list[Any] = []
        if include_canonical_forget_image:
            target_images.append(profile.image_value)
        target_images.extend(profile.variant_image_values)
        try:
            profile_seed = int(profile.profile_id)
        except ValueError:
            profile_seed = sum(ord(char) for char in profile.profile_id)
        for augmentation_index in range(forget_augmentations_per_image):
            target_images.append(
                AugmentedImage(
                    base_image_value=profile.image_value,
                    seed=(
                        seed * 1_000_003
                        + profile_seed * 10_007
                        + qa_index * 101
                        + augmentation_index
                    ),
                )
            )
        if not target_images:
            raise ValueError(
                f"No target images available for forget profile {profile.profile_id}."
            )
        if counterfactual_sampling == "nearest":
            neighbor_ids = counterfactual_neighbor_map.get(profile.profile_id, [])
            neighbor_profiles = [
                retain_by_profile[neighbor_id]
                for neighbor_id in neighbor_ids
                if neighbor_id in retain_by_profile
            ]
            if len(neighbor_profiles) < counterfactual_count:
                raise ValueError(
                    "Not enough nearest counterfactual profiles for forget "
                    f"profile {profile.profile_id}: need {counterfactual_count}, "
                    f"found {len(neighbor_profiles)}."
                )
            counterfactuals = neighbor_profiles[:counterfactual_count]
        elif len(retain_profiles) >= counterfactual_count:
            counterfactuals = rng.sample(retain_profiles, counterfactual_count)
        else:
            counterfactuals = [
                rng.choice(retain_profiles) for _ in range(counterfactual_count)
            ]
        for image_index, target_image_value in enumerate(target_images):
            sample_id = f"{profile.profile_id}_{qa_index:03d}"
            if len(target_images) > 1:
                sample_id = f"{sample_id}_view{image_index:02d}"
            examples.append(
                GainExample(
                    sample_id=sample_id,
                    target_profile_id=profile.profile_id,
                    question=question,
                    answer=answer,
                    target_image_value=target_image_value,
                    counterfactual_profile_ids=tuple(
                        item.profile_id for item in counterfactuals
                    ),
                    counterfactual_image_values=tuple(
                        item.image_value for item in counterfactuals
                    ),
                )
            )
    return examples


def training_step_count(examples: list[GainExample], args: argparse.Namespace) -> int:
    if args.train_epochs > 0:
        return len(examples) * args.train_epochs
    return args.train_steps


def build_training_schedule(
    examples: list[GainExample],
    args: argparse.Namespace,
) -> list[tuple[int, GainExample]]:
    rng = random.Random(args.seed)
    if args.train_epochs > 0:
        schedule: list[tuple[int, GainExample]] = []
        for epoch in range(1, args.train_epochs + 1):
            epoch_examples = list(examples)
            rng.shuffle(epoch_examples)
            schedule.extend((epoch, example) for example in epoch_examples)
        return schedule
    return [
        (0, rng.choice(examples)) for _ in range(args.train_steps)
    ]


def load_hard_negative_map(path: Path) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping = raw.get("simple_map", raw)
    if not isinstance(mapping, dict):
        raise ValueError("Hard-negative retain map must be a JSON object.")

    normalized: dict[str, list[str]] = {}
    for forget_id, retain_items in mapping.items():
        if not isinstance(retain_items, list):
            raise ValueError(
                f"Hard-negative retain map entry {forget_id!r} must be a list."
            )
        retain_ids: list[str] = []
        for item in retain_items:
            if isinstance(item, str):
                retain_ids.append(item)
            elif isinstance(item, dict) and item.get("retain_profile_id") is not None:
                retain_ids.append(str(item["retain_profile_id"]))
            else:
                raise ValueError(
                    f"Unsupported hard-negative item for forget ID {forget_id!r}: "
                    f"{item!r}"
                )
        normalized[str(forget_id)] = retain_ids
    return normalized


def retain_candidates_by_profile(
    retain_profiles: list[Profile],
) -> dict[str, list[RetainExample]]:
    by_profile: dict[str, list[RetainExample]] = {}
    for profile in retain_profiles:
        examples = [
            RetainExample(
                sample_id=f"{profile.profile_id}_{qa_index:03d}",
                profile_id=profile.profile_id,
                question=question,
                answer=answer,
                image_value=profile.image_value,
            )
            for qa_index, (question, answer) in enumerate(profile.qa_pairs)
        ]
        if examples:
            by_profile[profile.profile_id] = examples
    return by_profile


def build_retain_schedule(
    retain_profiles: list[Profile],
    training_schedule: list[tuple[int, GainExample]],
    samples_per_step: int,
    seed: int,
    hard_negative_map: dict[str, list[str]] | None = None,
) -> list[tuple[RetainExample, ...]]:
    total_steps = len(training_schedule)
    if total_steps < 1:
        return []

    candidates = [
        RetainExample(
            sample_id=f"{profile.profile_id}_{qa_index:03d}",
            profile_id=profile.profile_id,
            question=question,
            answer=answer,
            image_value=profile.image_value,
        )
        for profile in retain_profiles
        for qa_index, (question, answer) in enumerate(profile.qa_pairs)
    ]
    if not candidates:
        raise ValueError("No retain QA examples are available for distillation.")

    rng = random.Random(seed + 1)
    if not hard_negative_map:
        rng.shuffle(candidates)
        required = total_steps * samples_per_step
        selected = [candidates[index % len(candidates)] for index in range(required)]
        return [
            tuple(selected[offset : offset + samples_per_step])
            for offset in range(0, required, samples_per_step)
        ]

    by_profile = retain_candidates_by_profile(retain_profiles)
    schedule: list[tuple[RetainExample, ...]] = []
    hard_negative_count = 0
    fallback_count = 0
    for _, forget_example in training_schedule:
        hard_candidates: list[RetainExample] = []
        for retain_profile_id in hard_negative_map.get(
            forget_example.target_profile_id,
            [],
        ):
            hard_candidates.extend(by_profile.get(retain_profile_id, []))
        step_examples: list[RetainExample] = []
        for _ in range(samples_per_step):
            if hard_candidates:
                step_examples.append(rng.choice(hard_candidates))
                hard_negative_count += 1
            else:
                step_examples.append(rng.choice(candidates))
                fallback_count += 1
        schedule.append(tuple(step_examples))

    print(
        "Retain sampling: hard_negative="
        f"{hard_negative_count}, fallback_random={fallback_count}, "
        f"mapped_forget_profiles={len(hard_negative_map)}."
    )
    return schedule


def snapshot_retain_baselines(
    model,
    processor,
    retain_schedule: list[tuple[RetainExample, ...]],
) -> dict[str, dict[str, Any]]:
    """Cache vanilla answer-token log-probabilities before LoRA is attached."""
    baselines: dict[str, dict[str, Any]] = {}
    unique_examples = {
        example.sample_id: example
        for step_examples in retain_schedule
        for example in step_examples
    }
    model.eval()
    for index, example in enumerate(unique_examples.values(), start=1):
        result = full_answer_logprob(
            model,
            processor,
            decode_image(example.image_value),
            example.question,
            example.answer,
            require_grad=False,
        )
        baselines[example.sample_id] = {
            "token_ids": result["token_ids"],
            "token_log_probs": result["token_log_probs"].detach().float().cpu().tolist(),
        }
        print(
            f"[retain teacher {index}/{len(unique_examples)}] "
            f"{example.sample_id}"
        )
    return baselines


def retain_distillation_loss(
    model,
    processor,
    example: RetainExample,
    frozen_baseline: dict[str, Any],
):
    """Match vanilla answer-token log-probabilities on a retain image QA."""
    import torch

    result = full_answer_logprob(
        model,
        processor,
        decode_image(example.image_value),
        example.question,
        example.answer,
        require_grad=True,
    )
    if result["token_ids"] != frozen_baseline["token_ids"]:
        raise RuntimeError(
            f"Retain tokenization changed for {example.sample_id}."
        )
    teacher_log_probs = torch.tensor(
        frozen_baseline["token_log_probs"],
        dtype=result["token_log_probs"].dtype,
        device=result["token_log_probs"].device,
    )
    return torch.nn.functional.mse_loss(
        result["token_log_probs"],
        teacher_log_probs,
    )


def resolve_dtype(name: str):
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_model_and_processor(args: argparse.Namespace):
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    if not args.model_path:
        raise ValueError("--model-path is required unless --self-test is used.")

    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_dtype(args.dtype),
        "low_cpu_mem_usage": True,
        "local_files_only": args.local_files_only,
    }
    if args.device_map.casefold() != "none":
        model_kwargs["device_map"] = args.device_map

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        **model_kwargs,
    )
    processor = AutoProcessor.from_pretrained(
        args.processor_path,
        local_files_only=args.local_files_only,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return model, processor


def _input_device(model):
    import torch

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def _move_inputs(inputs: dict[str, Any], device):
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _common_prefix_length(left, right) -> int:
    limit = min(left.shape[-1], right.shape[-1])
    length = 0
    for index in range(limit):
        if int(left[index]) != int(right[index]):
            break
        length += 1
    return length


def full_answer_logprob(
    model,
    processor,
    image,
    question: str,
    answer: str,
    require_grad: bool,
):
    """Return sentence and per-token teacher-forced answer log-probabilities."""
    import torch

    prompt = f"USER: <image>\n{question}\nASSISTANT: "
    full_text = prompt + answer
    full_inputs = processor(
        text=full_text,
        images=image,
        return_tensors="pt",
    )
    prompt_inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    )

    full_ids_cpu = full_inputs["input_ids"][0]
    prompt_ids_cpu = prompt_inputs["input_ids"][0]
    answer_start = _common_prefix_length(full_ids_cpu, prompt_ids_cpu)
    if answer_start >= full_ids_cpu.shape[-1]:
        raise RuntimeError(
            "Could not locate answer tokens. Check the model chat template."
        )

    device = _input_device(model)
    full_inputs = _move_inputs(dict(full_inputs), device)
    context = torch.enable_grad() if require_grad else torch.no_grad()
    with context:
        outputs = model(**full_inputs)
        log_probs = outputs.logits[:, :-1].float().log_softmax(dim=-1)
        target_ids = full_inputs["input_ids"][:, 1:]
        token_log_probs = log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)

        original_positions = torch.arange(
            1,
            full_inputs["input_ids"].shape[-1],
            device=device,
        )
        answer_mask = original_positions.ge(answer_start).unsqueeze(0)
        if "attention_mask" in full_inputs:
            answer_mask = answer_mask & full_inputs["attention_mask"][:, 1:].bool()

        special_ids = set(processor.tokenizer.all_special_ids)
        if special_ids:
            special_mask = torch.ones_like(answer_mask)
            for special_id in special_ids:
                special_mask &= target_ids.ne(special_id)
            answer_mask &= special_mask

        selected_log_probs = token_log_probs.masked_select(answer_mask)
        selected_token_ids = target_ids.masked_select(answer_mask)
        token_count = selected_log_probs.numel()
        if token_count == 0:
            raise RuntimeError(
                f"No answer tokens remained after masking for question: {question!r}"
            )
        score = selected_log_probs.mean()

    token_ids_cpu = selected_token_ids.detach().cpu().tolist()
    return {
        "score": score,
        "token_log_probs": selected_log_probs,
        "token_ids": token_ids_cpu,
        "tokens": processor.tokenizer.convert_ids_to_tokens(token_ids_cpu),
        "token_texts": [
            processor.tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for token_id in token_ids_cpu
        ],
        "token_count": token_count,
        "answer_start": answer_start,
    }


def _lexical_token_mask(token_texts: list[str]):
    import torch

    return torch.tensor(
        [any(character.isalnum() for character in text) for text in token_texts],
        dtype=torch.bool,
    )


def _is_terminal_punctuation(text: str) -> bool:
    return text in {".", ";", "?", "!", "\n"}


def _is_entity_bridge(text: str) -> bool:
    stripped = text.strip()
    return stripped in {",", "-", "–", "—", "/", "&", "'", "’", "(", ")"}


def _is_entity_particle(text: str) -> bool:
    return text.lower() in {
        "of",
        "the",
        "and",
        "de",
        "del",
        "da",
        "dos",
        "van",
        "von",
        "der",
        "den",
        "la",
        "le",
        "di",
        "du",
        "al",
    }


def _looks_entity_token(text: str) -> bool:
    stripped = text.strip()
    if not stripped or not any(character.isalnum() for character in stripped):
        return False
    if any(character.isdigit() for character in stripped):
        return True
    letters = [character for character in stripped if character.isalpha()]
    if not letters:
        return False
    if letters[0].isupper():
        return True
    if len(letters) > 1 and all(character.isupper() for character in letters):
        return True
    return _is_entity_particle(stripped)


def _starts_new_word(raw_token: str | None, decoded_text: str, index: int) -> bool:
    if index == 0:
        return True
    if decoded_text.startswith(" "):
        return True
    if raw_token is None:
        return False
    return raw_token.startswith(("\u2581", "Ġ"))


def _token_word_groups(
    token_texts: list[str],
    token_strings: list[str] | None = None,
) -> list[list[int]]:
    groups: list[list[int]] = []
    pending_boundary = False
    for index, text in enumerate(token_texts):
        raw_token = token_strings[index] if token_strings is not None else None
        if not any(character.isalnum() for character in text):
            if (
                text.isspace()
                or _is_terminal_punctuation(text.strip())
                or _is_entity_bridge(text.strip())
                or (
                    raw_token is not None
                    and raw_token.startswith(("\u2581", "Ġ"))
                )
            ):
                pending_boundary = True
            continue
        if not groups or pending_boundary or _starts_new_word(raw_token, text, index):
            groups.append([index])
        else:
            groups[-1].append(index)
        pending_boundary = False
    return groups


def _word_text(token_texts: list[str], group: list[int]) -> str:
    return "".join(token_texts[index] for index in group).strip()


def _expand_to_full_words(mask, token_texts: list[str], token_strings=None):
    expanded = mask.clone()
    for group in _token_word_groups(token_texts, token_strings):
        if any(bool(mask[index]) for index in group):
            for index in group:
                expanded[index] = True
    return expanded


_LEFT_BOUNDARY_WORDS = {
    "'",
    "s",
    "'s",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "do",
    "does",
    "did",
    "enjoys",
    "enjoy",
    "likes",
    "like",
    "loves",
    "love",
    "prefers",
    "prefer",
    "works",
    "work",
    "born",
    "located",
    "based",
    "studies",
    "study",
    "studied",
    "attends",
    "attend",
    "attended",
    "graduated",
    "graduates",
    "graduate",
    "lives",
    "live",
    "lived",
    "resides",
    "reside",
    "resided",
}

_RIGHT_BOUNDARY_WORDS = {
    "and",
    "but",
    "while",
    "whereas",
    "although",
    "because",
}

_LEADING_TRIM_WORDS = {
    "a",
    "an",
    "the",
    "in",
    "at",
    "on",
    "to",
    "for",
    "with",
    "as",
}

_NEGATION_WORDS = {
    "no",
    "not",
    "none",
    "without",
    "never",
    "neither",
    "nor",
    "n't",
}


def _normalize_word(word: str) -> str:
    return word.lower().strip()


def _question_context_words(question: str) -> set[str]:
    """Return normalized words that are context supplied by the question.

    If a word appears in the question, it usually describes what is being asked
    about rather than the private answer value itself. This lets the span
    expansion stop at dataset-specific relation words such as father, mother,
    mum, advisor, roommate, etc. without hard-coding them.
    """
    words = {
        _normalize_word(match.group(0))
        for match in re.finditer(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", question)
    }
    expanded = set(words)
    for word in words:
        if word.endswith("'s") and len(word) > 2:
            expanded.add(word[:-2])
        if word.endswith("s") and len(word) > 3:
            expanded.add(word[:-1])
    return expanded


def _is_left_boundary_word(word: str, question_words: set[str] | None = None) -> bool:
    normalized = _normalize_word(word)
    if normalized in _LEFT_BOUNDARY_WORDS:
        return True
    if question_words and normalized in question_words:
        return True
    return False


def _is_right_boundary_word(word: str) -> bool:
    return _normalize_word(word) in _RIGHT_BOUNDARY_WORDS


def _is_predicate_anchor(word: str) -> bool:
    return _normalize_word(word) in {
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "enjoys",
        "enjoy",
        "likes",
        "like",
        "loves",
        "love",
        "prefers",
        "prefer",
        "studies",
        "study",
        "studied",
        "attends",
        "attend",
        "attended",
        "lives",
        "live",
        "lived",
        "resides",
        "reside",
        "resided",
    }


def _trim_span_start(words: list[str], start: int, end: int) -> int:
    while start <= end and _normalize_word(words[start]) in _LEADING_TRIM_WORDS:
        start += 1
    return start


def _negation_start_for_span(words: list[str], start: int, end: int) -> int:
    """Include negation if it defines the same answer fact as the span.

    For "does not have any medical conditions", an anchor on "conditions"
    should expand back to "not have any medical conditions"; otherwise the
    sensitive fact polarity is lost.
    """
    nearest_negation: int | None = None
    for index in range(start, -1, -1):
        word = _normalize_word(words[index])
        if word in _RIGHT_BOUNDARY_WORDS:
            break
        if word in _NEGATION_WORDS or word.endswith("n't"):
            nearest_negation = index
            break
    if nearest_negation is None:
        return start
    return min(nearest_negation, start)


def _expand_sensitive_content_span(
    mask,
    token_texts: list[str],
    token_strings=None,
    question_words: set[str] | None = None,
):
    """Expand high-gain anchors to nearby content phrases.

    The gain threshold only identifies anchor words. The final sensitive mask
    should cover the minimal semantic phrase around each anchor, so teacher
    forcing does not hide low-gain continuation words such as "school" in
    "high school mathematics teacher" or "records" in "vinyl records".
    """
    expanded = _expand_to_full_words(mask, token_texts, token_strings)
    groups = _token_word_groups(token_texts, token_strings)
    if not groups:
        return expanded

    words = [_word_text(token_texts, group) for group in groups]
    selected_group_indices = [
        group_index
        for group_index, group in enumerate(groups)
        if any(bool(expanded[token_index]) for token_index in group)
    ]
    if not selected_group_indices:
        return expanded

    spans: list[tuple[int, int]] = []
    for anchor in selected_group_indices:
        start = anchor
        end = anchor

        # Predicate anchors such as "enjoys" are usually evidence that the
        # following complement is sensitive; the predicate itself is template
        # wording, not the attribute we want to optimize.
        if _is_predicate_anchor(words[anchor]) and anchor + 1 < len(groups):
            start = anchor + 1
            end = anchor + 1

        left = start - 1
        while left >= 0:
            word = _normalize_word(words[left])
            if _is_left_boundary_word(word, question_words):
                break
            # A leading preposition/article is useful inside a phrase
            # ("records from the 1960s") but not before the first content word
            # ("born in Vienna").
            if word in _LEADING_TRIM_WORDS and left == start - 1:
                start = left
                left -= 1
                continue
            start = left
            left -= 1

        right = end + 1
        while right < len(groups):
            word = _normalize_word(words[right])
            if _is_right_boundary_word(word):
                break
            end = right
            right += 1

        start = _negation_start_for_span(words, start, end)
        start = _trim_span_start(words, start, end)
        if start <= end:
            spans.append((start, end))

    if not spans:
        return _expand_sensitive_entity_spans(expanded, token_texts, token_strings)

    for start, end in spans:
        for group in groups[start : end + 1]:
            for token_index in group:
                expanded[token_index] = True

    return expanded


def _expand_sensitive_entity_spans(mask, token_texts: list[str], token_strings=None):
    """Expand selected anchor tokens to nearby entity-like tokens.

    Teacher forcing can make a later entity token look non-sensitive because an
    earlier correct token has already exposed the answer, e.g. after "Vienna"
    the token "Austria" is easy to predict. This expansion keeps the gain-based
    token as the anchor but selects the rest of the same entity-like phrase.
    """
    import torch

    expanded = _expand_to_full_words(mask, token_texts, token_strings)
    lexical = _lexical_token_mask(token_texts)
    selected_indices = torch.where(expanded.cpu())[0].tolist()
    n_tokens = len(token_texts)

    for anchor in selected_indices:
        if not lexical[anchor]:
            continue

        for direction in (-1, 1):
            cursor = anchor + direction
            pending_bridge_indices: list[int] = []
            while 0 <= cursor < n_tokens:
                text = token_texts[cursor].strip()
                if _is_terminal_punctuation(text):
                    break

                if lexical[cursor]:
                    if _looks_entity_token(text):
                        for bridge_index in pending_bridge_indices:
                            # Punctuation helps define the span but usually
                            # should not be optimized as a sensitive token.
                            if lexical[bridge_index]:
                                expanded[bridge_index] = True
                        expanded[cursor] = True
                        pending_bridge_indices = []
                        cursor += direction
                        continue
                    break

                if _is_entity_bridge(text):
                    pending_bridge_indices.append(cursor)
                    cursor += direction
                    continue

                break

    return expanded


def select_sensitive_tokens(
    token_gains,
    token_texts: list[str],
    token_strings: list[str],
    question: str,
    gain_threshold: float,
    gain_quantile: float,
    minimum_count: int,
    maximum_fraction: float,
    span_expansion: str,
):
    """Select high positive-gain lexical tokens with deterministic top-k limits."""
    import torch

    if span_expansion not in {"none", "entity", "content"}:
        raise ValueError(
            "--sensitive-span-expansion must be 'none', 'entity', or 'content'."
        )
    if not 0.0 <= gain_quantile <= 1.0:
        raise ValueError("--token-gain-quantile must be between 0 and 1.")
    if minimum_count < 1:
        raise ValueError("--sensitive-token-min must be at least 1.")
    if not 0.0 < maximum_fraction <= 1.0:
        raise ValueError(
            "--sensitive-token-max-fraction must be greater than 0 and at most 1."
        )

    gains = token_gains.detach().float().cpu()
    lexical = _lexical_token_mask(token_texts)
    candidates = lexical & gains.gt(0)
    positive_gains = gains[candidates]
    question_words = _question_context_words(question)

    cutoff = float(gain_threshold)
    if gain_quantile > 0 and positive_gains.numel() > 0:
        cutoff = max(
            cutoff,
            float(torch.quantile(positive_gains, gain_quantile).item()),
        )
    selected = candidates & gains.ge(cutoff)

    lexical_count = int(lexical.sum().item())
    available_count = max(1, lexical_count)
    max_count = max(
        minimum_count,
        int(math.ceil(available_count * maximum_fraction)),
    )
    max_count = min(max_count, available_count)

    candidate_indices = torch.where(lexical)[0]
    if candidate_indices.numel() == 0:
        candidate_indices = torch.arange(gains.numel())
        max_count = min(max_count, int(candidate_indices.numel()))

    ranked_indices = candidate_indices[
        torch.argsort(gains[candidate_indices], descending=True)
    ]
    selected_indices = torch.where(selected)[0]
    if selected_indices.numel() < minimum_count:
        selected_indices = ranked_indices[:minimum_count]
    elif selected_indices.numel() > max_count:
        selected_indices = ranked_indices[:max_count]

    mask = torch.zeros_like(gains, dtype=torch.bool)
    mask[selected_indices] = True
    if span_expansion == "entity":
        mask = _expand_sensitive_entity_spans(mask, token_texts, token_strings)
    elif span_expansion == "content":
        mask = _expand_sensitive_content_span(
            mask,
            token_texts,
            token_strings,
            question_words=question_words,
        )
    return mask, cutoff


def score_example(
    model,
    processor,
    example: GainExample,
    tolerance: float,
    require_target_grad: bool,
    token_selection_mode: str,
    forget_loss_mode: str,
    token_gain_threshold: float,
    token_gain_quantile: float,
    sensitive_token_min: int,
    sensitive_token_max_fraction: float,
    sensitive_span_expansion: str,
    under_suppression_weight: float = 1.0,
    gain_lower_bound: float | None = None,
    gender_question_patterns: Iterable[str] | None = None,
    frozen_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    target_image = decode_image(example.target_image_value)
    target_result = full_answer_logprob(
        model,
        processor,
        target_image,
        example.question,
        example.answer,
        require_grad=require_target_grad,
    )
    target_score = target_result["score"]
    target_token_log_probs = target_result["token_log_probs"]

    if frozen_baseline is None:
        counterfactual_scores: list[float] = []
        counterfactual_token_log_probs = []
        for image_value in example.counterfactual_image_values:
            counterfactual_result = full_answer_logprob(
                model,
                processor,
                decode_image(image_value),
                example.question,
                example.answer,
                require_grad=False,
            )
            if counterfactual_result["token_ids"] != target_result["token_ids"]:
                raise RuntimeError(
                    "Target and counterfactual tokenization differ for "
                    f"{example.sample_id}."
                )
            counterfactual_scores.append(
                float(counterfactual_result["score"].detach().cpu())
            )
            counterfactual_token_log_probs.append(
                counterfactual_result["token_log_probs"].detach()
            )
        baseline_value = sum(counterfactual_scores) / len(counterfactual_scores)
        baseline_token_log_probs = torch.stack(
            counterfactual_token_log_probs,
            dim=0,
        ).mean(dim=0)
        initial_token_gains = (
            target_token_log_probs.detach() - baseline_token_log_probs
        )
        if token_selection_mode == "all":
            # Ablation: apply the token-level counterfactual objective to every
            # teacher-forced answer token, including punctuation/subword tokens.
            sensitive_mask_cpu = torch.ones_like(
                initial_token_gains,
                dtype=torch.bool,
            )
            selection_cutoff = 0.0
        else:
            sensitive_mask_cpu, selection_cutoff = select_sensitive_tokens(
                initial_token_gains,
                target_result["token_texts"],
                target_result["tokens"],
                example.question,
                gain_threshold=token_gain_threshold,
                gain_quantile=token_gain_quantile,
                minimum_count=sensitive_token_min,
                maximum_fraction=sensitive_token_max_fraction,
                span_expansion=sensitive_span_expansion,
            )
    else:
        counterfactual_scores = []
        baseline_value = float(frozen_baseline["score"])
        if frozen_baseline["token_ids"] != target_result["token_ids"]:
            raise RuntimeError(
                f"Frozen baseline tokenization changed for {example.sample_id}."
            )
        baseline_token_log_probs = torch.tensor(
            frozen_baseline["token_log_probs"],
            dtype=target_token_log_probs.dtype,
            device=target_token_log_probs.device,
        )
        sensitive_mask_cpu = torch.tensor(
            frozen_baseline["sensitive_mask"],
            dtype=torch.bool,
        )
        selection_cutoff = float(frozen_baseline["selection_cutoff"])

    baseline = torch.tensor(
        baseline_value,
        dtype=target_score.dtype,
        device=target_score.device,
    )
    sentence_loss, gain = counterfactual_gain_loss(
        target_score,
        baseline,
        tolerance,
        gain_lower_bound=gain_lower_bound,
    )
    baseline_token_log_probs = baseline_token_log_probs.to(
        target_token_log_probs.device
    )
    sensitive_mask = sensitive_mask_cpu.to(target_token_log_probs.device)
    token_loss, token_gains = sensitive_token_gain_loss(
        target_token_log_probs,
        baseline_token_log_probs,
        sensitive_mask,
        tolerance,
        under_suppression_weight=under_suppression_weight,
        gain_lower_bound=gain_lower_bound,
    )

    token_gains_cpu = token_gains.detach().float().cpu().tolist()
    target_token_log_probs_cpu = (
        target_token_log_probs.detach().float().cpu().tolist()
    )
    baseline_token_log_probs_cpu = (
        baseline_token_log_probs.detach().float().cpu().tolist()
    )
    sensitive_mask_list = sensitive_mask_cpu.tolist()
    token_details = [
        {
            "index": index,
            "token_id": token_id,
            "token": token,
            "text": token_text,
            "target_logprob": target_logprob,
            "baseline_logprob": baseline_logprob,
            "token_gain": token_gain,
            "sensitive": bool(is_sensitive),
        }
        for (
            index,
            token_id,
            token,
            token_text,
            target_logprob,
            baseline_logprob,
            token_gain,
            is_sensitive,
        ) in zip(
            range(target_result["token_count"]),
            target_result["token_ids"],
            target_result["tokens"],
            target_result["token_texts"],
            target_token_log_probs_cpu,
            baseline_token_log_probs_cpu,
            token_gains_cpu,
            sensitive_mask_list,
        )
    ]
    if forget_loss_mode == "sentence":
        loss_tensor = sentence_loss
    elif forget_loss_mode == "token":
        loss_tensor = token_loss
    else:
        raise ValueError("--forget-loss-mode must be 'token' or 'sentence'.")

    return {
        "loss_tensor": loss_tensor,
        "forget_loss_mode": forget_loss_mode,
        "target_score_tensor": target_score,
        "baseline_token_log_probs": baseline_token_log_probs_cpu,
        "sensitive_mask": sensitive_mask_list,
        "token_ids": target_result["token_ids"],
        "sample_id": example.sample_id,
        "target_profile_id": example.target_profile_id,
        "counterfactual_profile_ids": list(example.counterfactual_profile_ids),
        "question": example.question,
        "question_category": question_category(
            example.question,
            patterns=gender_question_patterns,
        ),
        "answer": example.answer,
        "answer_token_count": target_result["token_count"],
        "answer_start_token": target_result["answer_start"],
        "sensitive_token_count": int(sensitive_mask_cpu.sum().item()),
        "sensitive_tokens": [
            detail["text"]
            for detail in token_details
            if detail["sensitive"]
        ],
        "token_selection_cutoff": selection_cutoff,
        "token_details": token_details,
        "target_score": float(target_score.detach().cpu()),
        "counterfactual_scores": counterfactual_scores,
        "baseline_score": baseline_value,
        "identity_gain": float(gain.detach().cpu()),
        "gain_loss": float(sentence_loss.detach().cpu()),
        "sensitive_token_gain": float(
            token_gains.masked_select(sensitive_mask).mean().detach().cpu()
        ),
        "sensitive_token_loss": float(token_loss.detach().cpu()),
    }


def summarize(
    records: Iterable[dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        return {"samples": 0, "tolerance": tolerance}

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)

    gains = [float(row["identity_gain"]) for row in rows]
    return {
        "samples": len(rows),
        "tolerance": tolerance,
        "mean_target_score": mean("target_score"),
        "mean_baseline_score": mean("baseline_score"),
        "mean_identity_gain": mean("identity_gain"),
        "mean_absolute_identity_gain": (
            sum(abs(value) for value in gains) / len(gains)
        ),
        "median_identity_gain": sorted(gains)[len(gains) // 2],
        "positive_gain_rate": sum(value > 0 for value in gains) / len(gains),
        "negative_gain_rate": sum(value < 0 for value in gains) / len(gains),
        "within_tolerance_rate": (
            sum(abs(value) <= tolerance for value in gains) / len(gains)
        ),
        "under_suppression_rate": (
            sum(value < -tolerance for value in gains) / len(gains)
        ),
        "over_gain_rate": (
            sum(value > tolerance for value in gains) / len(gains)
        ),
        "active_loss_rate": (
            sum(abs(value) > tolerance for value in gains) / len(gains)
        ),
        "mean_gain_loss": mean("gain_loss"),
        "mean_sensitive_token_count": mean("sensitive_token_count"),
        "mean_sensitive_token_gain": mean("sensitive_token_gain"),
        "mean_sensitive_token_loss": mean("sensitive_token_loss"),
        "mean_target_perplexity": math.exp(-mean("target_score")),
        "mean_baseline_perplexity": math.exp(-mean("baseline_score")),
    }


def summarize_by_question_category(
    records: Iterable[dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(
            str(row.get("question_category", "unknown")),
            [],
        ).append(row)
    return {
        category: summarize(category_records, tolerance)
        for category, category_records in sorted(grouped.items())
    }


def evaluate(
    model,
    processor,
    examples: list[GainExample],
    args: argparse.Namespace,
    frozen_baselines: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    model.eval()
    for index, example in enumerate(examples, start=1):
        baseline = (
            frozen_baselines.get(example.sample_id)
            if frozen_baselines is not None
            else None
        )
        result = score_example(
            model,
            processor,
            example,
            tolerance=args.tolerance,
            require_target_grad=False,
            token_selection_mode=args.token_selection_mode,
            forget_loss_mode=args.forget_loss_mode,
            token_gain_threshold=args.token_gain_threshold,
            token_gain_quantile=args.token_gain_quantile,
            sensitive_token_min=args.sensitive_token_min,
            sensitive_token_max_fraction=args.sensitive_token_max_fraction,
            sensitive_span_expansion=args.sensitive_span_expansion,
            under_suppression_weight=args.under_suppression_weight,
            gain_lower_bound=args.gain_lower_bound,
            gender_question_patterns=args.gender_question_patterns,
            frozen_baseline=baseline,
        )
        result.pop("loss_tensor")
        result.pop("target_score_tensor")
        records.append(result)
        print(
            f"[{index}/{len(examples)}] {example.sample_id} "
            f"target={result['target_score']:.4f} "
            f"baseline={result['baseline_score']:.4f} "
            f"gain={result['identity_gain']:.4f} "
            f"sensitive={result['sensitive_tokens']} "
            f"token_loss={result['sensitive_token_loss']:.4f}"
        )
    return records, summarize(records, args.tolerance)


def attach_lora(model, args: argparse.Namespace):
    from peft import LoraConfig, get_peft_model

    target_modules = resolve_lora_target_modules(model, args)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(
        f"Attaching LoRA: scope={args.lora_module_scope}, "
        f"targets={len(target_modules) if isinstance(target_modules, list) else target_modules}"
    )
    if isinstance(target_modules, list):
        preview = target_modules[:12]
        print("LoRA target preview:", preview)
        if len(target_modules) > len(preview):
            print(f"... {len(target_modules) - len(preview)} more LoRA targets")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def _lora_scope_keywords(scope: str) -> tuple[str, ...]:
    if scope == "language":
        return ("language_model",)
    if scope == "vision":
        return ("vision_tower",)
    if scope == "projector":
        return ("multi_modal_projector",)
    if scope == "vision_projector":
        return ("vision_tower", "multi_modal_projector")
    if scope == "all":
        return ("vision_tower", "multi_modal_projector", "language_model")
    if scope == "custom":
        return ()
    raise ValueError(f"Unknown LoRA module scope: {scope}")


def _is_lora_linear_candidate(module_name: str, module, scope_keywords: tuple[str, ...]):
    import torch

    if not isinstance(module, torch.nn.Linear):
        return False
    excluded = ("embed_tokens", "embeddings", "lm_head")
    if any(keyword in module_name for keyword in excluded):
        return False
    return any(keyword in module_name for keyword in scope_keywords)


def _matches_lora_filter(module_name: str, filters: list[str]) -> bool:
    if not filters or "auto" in filters:
        return True
    leaf_name = module_name.rsplit(".", 1)[-1]
    for item in filters:
        if leaf_name == item or module_name == item or module_name.endswith(f".{item}"):
            return True
    return False


def resolve_lora_target_modules(model, args: argparse.Namespace):
    if args.lora_module_scope == "custom":
        if not args.lora_target_modules:
            raise ValueError("--lora-target-modules is required for custom LoRA scope.")
        return args.lora_target_modules

    scope_keywords = _lora_scope_keywords(args.lora_module_scope)
    filters = list(args.lora_target_modules or ["auto"])
    target_modules: list[str] = []
    for module_name, module in model.named_modules():
        if not _is_lora_linear_candidate(module_name, module, scope_keywords):
            continue
        if not _matches_lora_filter(module_name, filters):
            continue
        target_modules.append(module_name)

    target_modules = sorted(set(target_modules))
    if not target_modules:
        raise ValueError(
            "No LoRA target modules found for "
            f"scope={args.lora_module_scope!r}, filters={filters!r}. "
            "Use --lora-module-scope custom with explicit target modules, "
            "or use --lora-target-modules auto."
        )
    return target_modules


def build_lora_optimizer(model, args: argparse.Namespace):
    import torch

    language_lr = (
        args.learning_rate
        if args.language_learning_rate is None
        else args.language_learning_rate
    )
    projector_lr = (
        args.learning_rate
        if args.projector_learning_rate is None
        else args.projector_learning_rate
    )
    groups = {
        "vision_tower": {"params": [], "lr": args.learning_rate},
        "multi_modal_projector": {"params": [], "lr": projector_lr},
        "language_model": {"params": [], "lr": language_lr},
        "other": {"params": [], "lr": args.learning_rate},
    }

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "language_model" in name:
            groups["language_model"]["params"].append(parameter)
        elif "vision_tower" in name:
            groups["vision_tower"]["params"].append(parameter)
        elif "multi_modal_projector" in name:
            groups["multi_modal_projector"]["params"].append(parameter)
        else:
            groups["other"]["params"].append(parameter)

    optimizer_groups = []
    for group_name, group in groups.items():
        if not group["params"]:
            continue
        param_count = sum(parameter.numel() for parameter in group["params"])
        print(
            f"Optimizer group {group_name}: "
            f"{param_count} trainable params, lr={group['lr']}",
            flush=True,
        )
        optimizer_groups.append(group)

    if not optimizer_groups:
        raise ValueError("No trainable parameters found after attaching LoRA.")
    return torch.optim.AdamW(optimizer_groups, weight_decay=0.0)


def adapter_training_lrs(args: argparse.Namespace) -> dict[str, Any]:
    language_lr = (
        args.learning_rate
        if args.language_learning_rate is None
        else args.language_learning_rate
    )
    projector_lr = (
        args.learning_rate
        if args.projector_learning_rate is None
        else args.projector_learning_rate
    )
    return {
        "learning_rate": args.learning_rate,
        "language_learning_rate": args.language_learning_rate,
        "projector_learning_rate": args.projector_learning_rate,
        "effective_learning_rates": {
            "vision_tower": args.learning_rate,
            "multi_modal_projector": projector_lr,
            "language_model": language_lr,
            "other": args.learning_rate,
        },
    }


def write_adapter_training_metadata(
    adapter_dir: Path,
    args: argparse.Namespace,
    checkpoint_epoch: int | None = None,
) -> None:
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        print(
            f"Adapter config not found at {adapter_config_path}; "
            "skipping training metadata.",
            flush=True,
        )
        return

    metadata = {
        "training_lrs": adapter_training_lrs(args),
        "training_metadata": {
            "train_epochs": args.train_epochs,
            "train_steps": args.train_steps,
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_epochs": args.checkpoint_epochs,
            "retain_loss_weight": args.retain_loss_weight,
            "retain_samples_per_step": args.retain_samples_per_step,
            "retain_sampling": args.retain_sampling,
            "counterfactual_sampling": args.counterfactual_sampling,
            "counterfactual_neighbor_map": (
                str(args.counterfactual_neighbor_map)
                if args.counterfactual_neighbor_map
                else None
            ),
            "gain_lower_bound": args.gain_lower_bound,
            "forget_loss_mode": args.forget_loss_mode,
            "token_selection_mode": args.token_selection_mode,
            "lora_module_scope": args.lora_module_scope,
            "lora_target_modules": args.lora_target_modules,
        },
    }

    metadata_path = adapter_dir / "adapter_training_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def adapter_checkpoint_path(args: argparse.Namespace, epoch: int) -> Path:
    template = args.save_adapter_checkpoint_template
    if template is None:
        return Path(f"{args.save_adapter}_epoch{epoch}")
    return Path(
        template.format(
            epoch=epoch,
            save_adapter=str(args.save_adapter),
        )
    )


def save_lora_adapter(
    model,
    adapter_dir: Path,
    args: argparse.Namespace,
    checkpoint_epoch: int | None = None,
) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    write_adapter_training_metadata(adapter_dir, args, checkpoint_epoch=checkpoint_epoch)
    if checkpoint_epoch is None:
        print(f"LoRA adapter saved to {adapter_dir}")
    else:
        print(f"LoRA adapter checkpoint epoch {checkpoint_epoch} saved to {adapter_dir}")


def optimize_gain(
    model,
    processor,
    schedule: list[tuple[int, GainExample]],
    baselines: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    retain_schedule: list[tuple[RetainExample, ...]] | None = None,
    retain_baselines: dict[str, dict[str, Any]] | None = None,
):
    import torch

    model = attach_lora(model, args)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = build_lora_optimizer(model, args)
    history: list[dict[str, float]] = []

    model.train()
    total_steps = len(schedule)
    if retain_schedule is not None and len(retain_schedule) != total_steps:
        raise ValueError("Retain schedule length must match the training schedule.")
    checkpoint_epochs = set(args.checkpoint_epochs)
    for step, (epoch, example) in enumerate(schedule, start=1):
        optimizer.zero_grad(set_to_none=True)
        result = score_example(
            model,
            processor,
            example,
            tolerance=args.tolerance,
            require_target_grad=True,
            token_selection_mode=args.token_selection_mode,
            forget_loss_mode=args.forget_loss_mode,
            token_gain_threshold=args.token_gain_threshold,
            token_gain_quantile=args.token_gain_quantile,
            sensitive_token_min=args.sensitive_token_min,
            sensitive_token_max_fraction=args.sensitive_token_max_fraction,
            sensitive_span_expansion=args.sensitive_span_expansion,
            under_suppression_weight=args.under_suppression_weight,
            gain_lower_bound=args.gain_lower_bound,
            gender_question_patterns=args.gender_question_patterns,
            frozen_baseline=baselines[example.sample_id],
        )
        forget_loss = result["loss_tensor"]
        retain_loss = torch.zeros_like(forget_loss)
        if retain_schedule is not None:
            if retain_baselines is None:
                raise ValueError("Retain baselines are required with a retain schedule.")
            step_losses = [
                retain_distillation_loss(
                    model,
                    processor,
                    retain_example,
                    retain_baselines[retain_example.sample_id],
                )
                for retain_example in retain_schedule[step - 1]
            ]
            retain_loss = torch.stack(step_losses).mean()

        loss = forget_loss + args.retain_loss_weight * retain_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        item = {
            "step": step,
            "epoch": epoch,
            "gain_loss": float(loss.detach().cpu()),
            "forget_loss": float(forget_loss.detach().cpu()),
            "retain_distillation_loss": float(retain_loss.detach().cpu()),
            "identity_gain": result["identity_gain"],
            "sensitive_token_gain": result["sensitive_token_gain"],
            "sensitive_token_count": result["sensitive_token_count"],
            "target_score": result["target_score"],
            "baseline_score": result["baseline_score"],
        }
        history.append(item)
        epoch_text = f" epoch={epoch}" if epoch > 0 else ""
        print(
            f"[train {step}/{total_steps}{epoch_text}] {example.sample_id} "
            f"token_gain={item['sensitive_token_gain']:.4f} "
            f"forget_loss={item['forget_loss']:.4f} "
            f"retain_loss={item['retain_distillation_loss']:.4f} "
            f"total_loss={item['gain_loss']:.4f}"
        )
        next_epoch = schedule[step][0] if step < total_steps else None
        if epoch in checkpoint_epochs and next_epoch != epoch:
            save_lora_adapter(
                model,
                adapter_checkpoint_path(args, epoch),
                args,
                checkpoint_epoch=epoch,
            )

    save_lora_adapter(model, args.save_adapter, args)
    return model, history


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")
    if args.self_test:
        run_self_test()
        return

    random.seed(args.seed)
    forget_profiles = load_profiles(args.forget_parquet)
    if args.forget_variant_parquet is not None:
        variant_images_by_id = load_forget_variant_images(
            args.forget_variant_parquet,
            max_images_per_profile=args.forget_variant_max_images,
        )
        forget_profiles = attach_forget_variants(
            forget_profiles,
            variant_images_by_id,
        )
    retain_profiles = load_profiles(args.retain_parquet)
    counterfactual_neighbor_map = None
    if args.counterfactual_sampling == "nearest":
        counterfactual_neighbor_map = load_hard_negative_map(args.counterfactual_neighbor_map)
    examples = build_examples(
        forget_profiles,
        retain_profiles,
        max_samples=args.max_samples,
        counterfactual_count=args.counterfactual_count,
        seed=args.seed,
        include_canonical_forget_image=args.include_canonical_forget_image,
        forget_augmentations_per_image=args.forget_augmentations_per_image,
        counterfactual_sampling=args.counterfactual_sampling,
        counterfactual_neighbor_map=counterfactual_neighbor_map,
    )
    print(
        f"Loaded {len(forget_profiles)} forget profiles, "
        f"{len(retain_profiles)} retain profiles, "
        f"{len(examples)} QA examples."
    )

    model, processor = load_model_and_processor(args)
    initial_records, initial_summary = evaluate(
        model,
        processor,
        examples,
        args,
    )
    baselines = {
        row["sample_id"]: {
            "score": float(row["baseline_score"]),
            "token_ids": row["token_ids"],
            "token_log_probs": row["baseline_token_log_probs"],
            "sensitive_mask": row["sensitive_mask"],
            "selection_cutoff": row["token_selection_cutoff"],
        }
        for row in initial_records
    }
    training_examples, training_filter_summary = filter_training_examples(
        examples,
        args,
    )
    retain_schedule: list[tuple[RetainExample, ...]] | None = None
    retain_baselines: dict[str, dict[str, Any]] | None = None
    training_schedule = build_training_schedule(training_examples, args)
    total_training_steps = len(training_schedule)
    if args.retain_loss_weight > 0 and total_training_steps > 0:
        hard_negative_map = None
        if args.retain_sampling == "hard_negative":
            if args.retain_hard_negative_map is None:
                raise ValueError(
                    "--retain-hard-negative-map is required when "
                    "--retain-sampling hard_negative is used."
                )
            hard_negative_map = load_hard_negative_map(args.retain_hard_negative_map)
        retain_schedule = build_retain_schedule(
            retain_profiles,
            training_schedule=training_schedule,
            samples_per_step=args.retain_samples_per_step,
            seed=args.seed,
            hard_negative_map=hard_negative_map,
        )
        print(
            "Retain constraint: caching vanilla token log-probabilities for "
            f"{sum(len(items) for items in retain_schedule)} sampled retain QA "
            f"examples (weight={args.retain_loss_weight})."
        )
        retain_baselines = snapshot_retain_baselines(
            model,
            processor,
            retain_schedule,
        )
    if args.exclude_gender_questions_from_training:
        print(
            "Training filter: skipped "
            f"{training_filter_summary['gender_question_examples']} "
            "gender/sex visual-attribute QA examples; "
            f"training on {training_filter_summary['training_examples']} "
            f"of {training_filter_summary['total_examples']} examples."
        )
    else:
        print(
            "Training filter: gender/sex visual-attribute QA examples are "
            "kept in training. Set exclude_gender_questions_from_training: "
            "true to skip them."
        )

    payload: dict[str, Any] = {
        "config": {
            "config_file": str(args.config) if args.config else None,
            "forget_parquet": str(args.forget_parquet),
            "forget_variant_parquet": (
                str(args.forget_variant_parquet)
                if args.forget_variant_parquet
                else None
            ),
            "forget_variant_max_images": args.forget_variant_max_images,
            "forget_augmentations_per_image": args.forget_augmentations_per_image,
            "include_canonical_forget_image": args.include_canonical_forget_image,
            "retain_parquet": str(args.retain_parquet),
            "model_path": args.model_path,
            "processor_path": args.processor_path,
            "max_samples": args.max_samples,
            "counterfactual_count": args.counterfactual_count,
            "counterfactual_sampling": args.counterfactual_sampling,
            "counterfactual_neighbor_map": (
                str(args.counterfactual_neighbor_map)
                if args.counterfactual_neighbor_map
                else None
            ),
            "tolerance": args.tolerance,
            "gain_lower_bound": args.gain_lower_bound,
            "under_suppression_weight": args.under_suppression_weight,
            "forget_loss_mode": args.forget_loss_mode,
            "token_selection_mode": args.token_selection_mode,
            "token_gain_threshold": args.token_gain_threshold,
            "token_gain_quantile": args.token_gain_quantile,
            "sensitive_token_min": args.sensitive_token_min,
            "sensitive_token_max_fraction": args.sensitive_token_max_fraction,
            "sensitive_span_expansion": args.sensitive_span_expansion,
            "exclude_gender_questions_from_training": (
                args.exclude_gender_questions_from_training
            ),
            "gender_question_patterns": args.gender_question_patterns,
            "seed": args.seed,
            "dtype": args.dtype,
            "device_map": args.device_map,
            "cuda_visible_devices": args.cuda_visible_devices,
            "local_files_only": args.local_files_only,
            "train_steps": args.train_steps,
            "train_epochs": args.train_epochs,
            "checkpoint_epochs": args.checkpoint_epochs,
            "learning_rate": args.learning_rate,
            "language_learning_rate": args.language_learning_rate,
            "projector_learning_rate": args.projector_learning_rate,
            "retain_loss_weight": args.retain_loss_weight,
            "retain_samples_per_step": args.retain_samples_per_step,
            "retain_sampling": args.retain_sampling,
            "retain_hard_negative_map": (
                str(args.retain_hard_negative_map)
                if args.retain_hard_negative_map
                else None
            ),
            "lora_module_scope": args.lora_module_scope,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": args.lora_target_modules,
            "save_adapter": str(args.save_adapter),
            "save_adapter_checkpoint_template": args.save_adapter_checkpoint_template,
        },
        "training_filter_summary": training_filter_summary,
        "initial_summary": initial_summary,
        "initial_summary_by_question_category": summarize_by_question_category(
            initial_records,
            args.tolerance,
        ),
        "initial_summary_without_gender_questions": summarize(
            [
                row
                for row in initial_records
                if row.get("question_category") != "visual_attribute_gender"
            ],
            args.tolerance,
        ),
        "initial_records": initial_records,
    }

    if args.train_steps > 0 or args.train_epochs > 0:
        model, history = optimize_gain(
        model,
        processor,
        training_schedule,
        baselines,
        args,
            retain_schedule=retain_schedule,
            retain_baselines=retain_baselines,
        )
        final_records, final_summary = evaluate(
            model,
            processor,
            examples,
            args,
            frozen_baselines=baselines,
        )
        payload["training_history"] = history
        payload["final_summary"] = final_summary
        payload["final_summary_by_question_category"] = (
            summarize_by_question_category(final_records, args.tolerance)
        )
        payload["final_summary_without_gender_questions"] = summarize(
            [
                row
                for row in final_records
                if row.get("question_category") != "visual_attribute_gender"
            ],
            args.tolerance,
        )
        payload["final_records"] = final_records

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Results saved to {args.output}")
    if "final_summary" in payload:
        print("Final summary overall:")
        print(json.dumps(payload["final_summary"], indent=2))
        print("Final summary without gender questions:")
        print(json.dumps(payload["final_summary_without_gender_questions"], indent=2))
        print("Final summary by question category:")
        print(json.dumps(payload["final_summary_by_question_category"], indent=2))
    else:
        print("Initial summary overall:")
        print(json.dumps(payload["initial_summary"], indent=2))
        print("Initial summary without gender questions:")
        print(json.dumps(payload["initial_summary_without_gender_questions"], indent=2))
        print("Initial summary by question category:")
        print(json.dumps(payload["initial_summary_by_question_category"], indent=2))


if __name__ == "__main__":
    main()
