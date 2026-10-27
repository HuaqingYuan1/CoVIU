import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoProcessor, LlavaForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_process.CLEAR_process import train_collate_clear
from data_process.data_preprocess import LLAVA_multimodal_Dataset
from data_process.MLLMU_process import MLLMU_text_Dataset
from data_process.SFRon import Mask_Our


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data_root"]
    forget_ratio = cfg["forget_ratio"]
    cfg.setdefault(
        "forget_parquet",
        os.path.join(data_root, f"forget_{forget_ratio}", "train-00000-of-00001.parquet"),
    )
    cfg.setdefault(
        "retain_parquet",
        os.path.join(data_root, f"retain_{100 - forget_ratio}", "train-00000-of-00001.parquet"),
    )
    cfg.setdefault(
        "full_set_parquet",
        os.path.join(data_root, "Full_Set", "train-00000-of-00001.parquet"),
    )
    cfg.setdefault("output_dir", os.path.join(ROOT, "masks", f"forget{forget_ratio}"))
    return cfg


def build_dataloaders(cfg, processor, device):
    forget_df = pd.read_parquet(cfg["forget_parquet"])
    retain_df = pd.read_parquet(cfg["retain_parquet"])
    full_df = pd.read_parquet(cfg["full_set_parquet"])

    multimodal_forget_dataset = LLAVA_multimodal_Dataset(df=forget_df)
    multimodal_knowledge_dataset = MLLMU_text_Dataset(df=full_df)
    multimodal_remain_dataset = LLAVA_multimodal_Dataset(df=retain_df)

    language_preserve_dataset = ConcatDataset([multimodal_knowledge_dataset, multimodal_remain_dataset])
    vision_preserve_dataset = ConcatDataset([multimodal_remain_dataset])

    batch_size = cfg["dataloader"]["batch_size"]
    shuffle = cfg["dataloader"].get("shuffle", True)
    collate = lambda batch: train_collate_clear(batch, processor, device, True)

    forget_dataloader = DataLoader(
        multimodal_forget_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )
    vision_preserve_dataloader = DataLoader(
        vision_preserve_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )
    language_preserve_dataloader = DataLoader(
        language_preserve_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )
    return forget_dataloader, vision_preserve_dataloader, language_preserve_dataloader


def save_checkpoint(data, folder, filename):
    path = os.path.join(folder, filename)
    tmp_path = path + ".tmp"
    torch.save(data, tmp_path)
    os.replace(tmp_path, path)


def build_mask_result(weight_mask, forget_grad, preserve_grad, weight_only=False):
    if weight_only:
        return {"weight": weight_mask}
    return {
        "weight": weight_mask,
        "forget_grad": forget_grad,
        "preserve_grad": preserve_grad,
    }


def save_mask_result(mg, folder, filename, weight_only=False):
    weight_mask, forget_grad, preserve_grad = mg.get_weight_saliency_mask()
    save_checkpoint(
        build_mask_result(weight_mask, forget_grad, preserve_grad, weight_only),
        folder,
        filename,
    )


def run(cfg, skip_vision=False, weight_only=False):
    if cfg.get("cuda_visible_devices"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda_visible_devices"])

    model = LlavaForConditionalGeneration.from_pretrained(
        cfg["model_path"],
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=os.path.isdir(cfg["model_path"]),
    )
    processor = AutoProcessor.from_pretrained(
        cfg["model_id"],
        local_files_only=os.path.isdir(cfg["model_id"]),
    )
    processor.tokenizer.padding_side = "right"
    print(model)

    mask_cfg = cfg["mask"]
    mg = Mask_Our(model, mask_cfg["lr"])
    forget_dataloader, vision_preserve_dataloader, language_preserve_dataloader = build_dataloaders(
        cfg, processor, model.device
    )

    folder = cfg["output_dir"]
    os.makedirs(folder, exist_ok=True)
    label = mask_cfg["label"]
    threshold = mask_cfg["threshold"]

    if skip_vision:
        print("Skipping vision mask generation.")
    else:
        weight_mask, forget_grad, preserve_grad = mg.prepare_weight_saliency_mask(
            modules=["vision_tower", "multi_modal_projector"],
            forget_loader=forget_dataloader,
            preserve_loader=vision_preserve_dataloader,
            threshold=threshold,
            save_path="",
        )
        save_checkpoint(
            build_mask_result(weight_mask, forget_grad, preserve_grad, weight_only),
            folder,
            f"mllmu_vision_mask_{label}.pt",
        )

    weight_mask, forget_grad, preserve_grad = mg.prepare_weight_saliency_mask(
        modules=["language_model"],
        forget_loader=forget_dataloader,
        preserve_loader=language_preserve_dataloader,
        threshold=threshold,
        save_path="",
    )
    save_checkpoint(
        build_mask_result(weight_mask, forget_grad, preserve_grad, weight_only),
        folder,
        f"mllmu_language_mask_{label}.pt",
    )

    if mask_cfg.get("save_both_mask", False):
        save_mask_result(mg, folder, f"mllmu_both_mask_{label}.pt", weight_only)
    print(f"Saved masks to {folder}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MLLMU saliency masks for MMUnlearner.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config" / "mllmu_gen_mask.yaml"),
        help="Path to yaml config file.",
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip vision mask generation and only generate the language mask.",
    )
    parser.add_argument(
        "--weight-only",
        action="store_true",
        help="Only save the weight mask and omit large gradient tensors.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(load_config(args.config), skip_vision=args.skip_vision, weight_only=args.weight_only)
