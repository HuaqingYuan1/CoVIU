from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a hard-negative retain map by retrieving visually similar "
            "retain identities for each forget identity with a frozen CLIP image encoder."
        )
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
        "--clip-model-path",
        type=str,
        default="../models/clip-vit-base-patch32",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hard_negative_retain_map_clip_top20.json"),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use 'auto', 'cuda', or 'cpu'.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=True,
        help="Load CLIP from local files only.",
    )
    parser.add_argument(
        "--allow-download",
        dest="local_files_only",
        action="store_false",
        help="Allow transformers to download missing CLIP files.",
    )
    return parser


def decode_image(value: Any):
    from PIL import Image

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


def profile_id(row: Any, row_index: int) -> str:
    for key in ("ID", "id", "unique_id", "profile_id"):
        if key in row and row[key] is not None:
            return str(row[key])
    return str(row_index)


def load_profile_images(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    if "image" not in frame.columns:
        raise ValueError(f"{path} is missing the image column.")

    profiles: list[dict[str, Any]] = []
    for row_index, row in frame.iterrows():
        profiles.append(
            {
                "profile_id": profile_id(row, int(row_index)),
                "image": row["image"],
            }
        )
    if not profiles:
        raise ValueError(f"No profiles found in {path}.")
    return profiles


def resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def encode_images(model, processor, profiles: list[dict[str, Any]], batch_size: int, device):
    import torch
    import torch.nn.functional as F

    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(profiles), batch_size):
            batch = profiles[start : start + batch_size]
            images = [decode_image(item["image"]) for item in batch]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            image_features = model.get_image_features(**inputs)
            image_features = F.normalize(image_features.float(), dim=-1)
            embeddings.append(image_features.cpu())
            print(f"[embed] {min(start + batch_size, len(profiles))}/{len(profiles)}")
    return torch.cat(embeddings, dim=0)


def main() -> None:
    args = build_parser().parse_args()

    import torch
    from transformers import CLIPModel, CLIPProcessor

    if args.top_k < 1:
        raise ValueError("--top-k must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    forget_profiles = load_profile_images(args.forget_parquet)
    retain_profiles = load_profile_images(args.retain_parquet)
    device = resolve_device(args.device)

    print(f"forget profiles: {len(forget_profiles)}")
    print(f"retain profiles: {len(retain_profiles)}")
    print(f"clip model: {args.clip_model_path}")
    print(f"device: {device}")

    processor = CLIPProcessor.from_pretrained(
        args.clip_model_path,
        local_files_only=args.local_files_only,
    )
    model = CLIPModel.from_pretrained(
        args.clip_model_path,
        local_files_only=args.local_files_only,
    ).to(device)

    print("encoding forget images...")
    forget_embeddings = encode_images(
        model,
        processor,
        forget_profiles,
        args.batch_size,
        device,
    )
    print("encoding retain images...")
    retain_embeddings = encode_images(
        model,
        processor,
        retain_profiles,
        args.batch_size,
        device,
    )

    similarities = forget_embeddings @ retain_embeddings.T
    top_k = min(args.top_k, len(retain_profiles))
    top_scores, top_indices = torch.topk(similarities, k=top_k, dim=1)

    hard_negative_map: dict[str, list[dict[str, Any]]] = {}
    simple_map: dict[str, list[str]] = {}
    for forget_index, forget_profile in enumerate(forget_profiles):
        forget_id = forget_profile["profile_id"]
        entries = []
        for rank, retain_index in enumerate(top_indices[forget_index].tolist(), start=1):
            retain_id = retain_profiles[retain_index]["profile_id"]
            score = float(top_scores[forget_index, rank - 1].item())
            entries.append(
                {
                    "rank": rank,
                    "retain_profile_id": retain_id,
                    "cosine_similarity": score,
                }
            )
        hard_negative_map[forget_id] = entries
        simple_map[forget_id] = [entry["retain_profile_id"] for entry in entries]

    payload = {
        "method": "clip_image_cosine_similarity",
        "clip_model_path": args.clip_model_path,
        "forget_parquet": str(args.forget_parquet),
        "retain_parquet": str(args.retain_parquet),
        "top_k": top_k,
        "forget_count": len(forget_profiles),
        "retain_count": len(retain_profiles),
        "hard_negative_map": hard_negative_map,
        "simple_map": simple_map,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
