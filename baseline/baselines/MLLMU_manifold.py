import sys
import os
import pandas as pd
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from peft import PeftModel
import random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
import torch
import json
import argparse
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, get_scheduler, AdamW, MllamaForConditionalGeneration, AutoTokenizer,Qwen2VLForConditionalGeneration, set_seed as transformers_set_seed
# from qwen_vl_utils import process_vision_info
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from data_process.data_preprocess import LLAVA_multimodal_Dataset, train_collate_fn_mllmu, train_collate_mllmu_ansonly,Vanilla_LLaVA_Dataset
from data_process.CLEAR_process import CLEAR_Dataset, CAPTION_MODE, RECOGNITION_MODE, train_collate_clear, NONE_MODE,train_collate_clear_ansonly
from data_process.MLLMU_process import train_collate_fn_llava_new
from accelerate import Accelerator
import torch
from baselines.config_utils import load_clear_split

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def flatten_yaml_config(value, output, prefix=""):
    if not isinstance(value, dict):
        raise TypeError("The YAML root must be a mapping.")
    for key, item in value.items():
        normalized_key = str(key).replace("-", "_")
        location = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flatten_yaml_config(item, output, location)
            continue
        if normalized_key in output:
            raise ValueError(
                f"Duplicate YAML option {normalized_key!r} at {location}."
            )
        output[normalized_key] = item


def load_yaml_defaults(path):
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
    flatten_yaml_config(raw, flattened)
    return flattened


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)


def build_parser():
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--model_id", type=str, default=None, help="Processor/model ID")
    parser.add_argument("--vanilla_dir", type=str, default=None, help="Base model directory")
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--data_split_dir", type=str, default="../Data_split", help="Data split directory")
    parser.add_argument("--dataset_type", choices=("mllmu", "clear"), default="mllmu")
    parser.add_argument("--clear_data_dir", type=str, default="data/CLEAR")
    parser.add_argument("--forget_split_ratio", type=int, default=5, help="Forget split ratio")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=str2bool, default=False, help="Enable gradient accumulation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Reserved; gradient accumulation is not implemented in this script")
    parser.add_argument("--trainer", type=str2bool, default=False, help="Use HuggingFace Trainer")
    parser.add_argument("--grad_mask_path", type=str, default=None, help="Mask for gradient update")
    parser.add_argument("--mask_dir", type=str, default=None, help="Optional mask directory recorded in trainer_config.json")
    parser.add_argument("--forget_alpha", type=float, default=1.0, help="Forget loss weight")
    parser.add_argument("--ans_only", type=str2bool, default=True, help="Only compute LM loss on answer tokens")
    parser.add_argument("--cuda_visible_devices", type=str, default=None, help="CUDA_VISIBLE_DEVICES value; prefer setting it before launch")
    parser.add_argument("--mask_check_only", type=str2bool, default=False, help="Print mask coverage and exit before training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def parse_args():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()

    parser = build_parser()
    if bootstrap_args.config:
        yaml_defaults = load_yaml_defaults(bootstrap_args.config)
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
    if not args.model_id:
        raise ValueError("--model_id is required, either in YAML or CLI.")
    if not args.vanilla_dir:
        raise ValueError("--vanilla_dir is required, either in YAML or CLI.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be >= 1.")
    if args.gradient_accumulation_steps > 1:
        args.gradient_accumulation = True
    return args


def find_all_linear_names(model):
    print(model)
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["embeddings","embed_tokens","patch_embed"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else ".".join(names[-2:]))

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    # if "qwen" in str(model).lower():
    #     lora_module_names.remove('proj')
    return list(lora_module_names)

# Example usage:
def load_model_and_processor(args):
    """
    Load the model and processor based on the provided model_id.
    Different models may require different loading methods, which are handled with conditional statements.
    """
    if "llava" in args.model_id:
        # Load LLAVA model and processor
        print("Loading LLAVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        print("getting peft model")
        lora_config = LoraConfig(
            r=16, #32
            lora_alpha=16, #8
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    elif "llama" in args.model_id.lower():
        model = MllamaForConditionalGeneration.from_pretrained(
            args.vanilla_dir, 
            device_map="auto",
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            local_files_only=True,
        )

        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
    elif "qwen" in args.model_id.lower():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.vanilla_dir, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    return model, processor


def _base_lora_parameter_name(name):
    """Map a PEFT LoRA parameter name back to the wrapped base weight name."""
    normalized = name
    if normalized.startswith("base_model.model."):
        normalized = normalized[len("base_model.model."):]
    for marker in (".lora_A.default.weight", ".lora_B.default.weight"):
        if marker in normalized:
            return normalized.replace(marker, ".weight")
    return normalized


def _lora_mask_from_base_mask(name, base_mask, grad_shape):
    """Project a full weight mask onto a LoRA A/B gradient mask."""
    if not torch.is_tensor(base_mask) or base_mask.ndim != 2:
        return None

    if ".lora_A.default.weight" in name:
        # LoRA A has shape [rank, in_features]. A column can update any
        # output row, so keep input columns selected by the full mask.
        mask = base_mask.any(dim=0, keepdim=True)
    elif ".lora_B.default.weight" in name:
        # LoRA B has shape [out_features, rank]. Keep output rows selected
        # by the full mask.
        mask = base_mask.any(dim=1, keepdim=True)
    else:
        return None

    if mask.numel() == 0:
        return None
    try:
        return mask.to(dtype=torch.bool).expand(grad_shape).contiguous()
    except RuntimeError:
        return None


def build_trainable_grad_mask(model, grad_mask):
    """Build masks keyed by actual trainable parameter names.

    MMUnlearner masks are generated for base model weights, while this script
    trains PEFT LoRA parameters. This adapts base masks to LoRA A/B shapes once
    before training so the hot loop can use a direct name lookup.
    """
    trainable_grad_mask = {}
    exact_hits = 0
    lora_hits = 0
    misses = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in grad_mask:
            mask = grad_mask[name]
            if torch.is_tensor(mask) and tuple(mask.shape) == tuple(p.shape):
                trainable_grad_mask[name] = mask.to(dtype=torch.bool).cpu()
                exact_hits += 1
            else:
                misses.append((name, "exact_shape_mismatch"))
            continue

        base_name = _base_lora_parameter_name(name)
        base_mask = grad_mask.get(base_name)
        mask = _lora_mask_from_base_mask(name, base_mask, p.shape)
        if mask is not None:
            trainable_grad_mask[name] = mask.cpu()
            lora_hits += 1
        else:
            misses.append((name, base_name))

    total_trainable = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    print(
        "Built trainable grad mask: "
        f"masked={len(trainable_grad_mask)}/{total_trainable}, "
        f"exact={exact_hits}, lora_projected={lora_hits}, "
        f"missed={len(misses)}",
        flush=True,
    )
    if trainable_grad_mask:
        examples = list(trainable_grad_mask.items())[:5]
        print(
            "Trainable mask examples:",
            [(name, tuple(mask.shape), float(mask.float().mean())) for name, mask in examples],
            flush=True,
        )
    if misses:
        print("Trainable mask miss examples:", misses[:5], flush=True)
    return trainable_grad_mask


def apply_trainable_grad_mask(model, trainable_grad_mask):
    masked = 0
    for name, p in model.named_parameters():
        if p.grad is not None and name in trainable_grad_mask:
            p.grad *= trainable_grad_mask[name].to(p.grad.device)
            masked += 1
    return masked


def print_mask_coverage(model, grad_mask):
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    trainable_names = [name for name, _ in trainable]
    exact_hits = [name for name in trainable_names if name in grad_mask]
    base_hits = [
        name for name in trainable_names
        if _base_lora_parameter_name(name) in grad_mask
    ]

    print(
        "Mask coverage on trainable parameters: "
        f"exact={len(exact_hits)}/{len(trainable_names)}, "
        f"base_lora_name={len(base_hits)}/{len(trainable_names)}",
        flush=True,
    )
    if exact_hits:
        print("Exact mask hit examples:", exact_hits[:5], flush=True)
    else:
        print("Exact mask hit examples: []", flush=True)
    if base_hits:
        examples = [
            (name, _base_lora_parameter_name(name))
            for name in base_hits[:5]
        ]
        print("Base-name LoRA mask hit examples:", examples, flush=True)
    else:
        print("Base-name LoRA mask hit examples: []", flush=True)



######################### Accelerate Version #################################
def main(args):
    set_random_seed(args.seed)
    print(f"Seed: {args.seed}", flush=True)
    # Load model and processor
    print("Trainer Status is ", args.trainer)
    model, processor = load_model_and_processor(args)
    print(model)
    tokenizer = processor.tokenizer
    print("Tokenizer Length: ", len(tokenizer))

    current_vocab = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > current_vocab:
        print(f"Expanding vocabulary: {current_vocab} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    # Dataset and Dataloader setup

    if args.dataset_type == "clear":
        forget_folder = os.path.join(args.clear_data_dir, f"forget{args.forget_split_ratio:02d}")
        retain_folder = os.path.join(args.clear_data_dir, f"retain{100 - args.forget_split_ratio}")
    else:
        forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
        retain_folder = os.path.join(args.data_split_dir, f"retain_{100 - args.forget_split_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)

    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")

    # Load DataLoader
    if args.dataset_type == "clear":
        forget_df = load_clear_split(args.clear_data_dir, "forget", args.forget_split_ratio)
        retain_df = load_clear_split(args.clear_data_dir, "retain", args.forget_split_ratio)
        multimodal_forget_dataset = CLEAR_Dataset(data=forget_df, mode=CAPTION_MODE)
        multimodal_retain_dataset = CLEAR_Dataset(data=retain_df, mode=CAPTION_MODE)
    else:
        forget_df = pd.read_parquet(forget_parquet_file)
        retain_df = pd.read_parquet(retain_parquet_file)
        multimodal_forget_dataset = LLAVA_multimodal_Dataset(df=forget_df)
        multimodal_retain_dataset = LLAVA_multimodal_Dataset(df=retain_df)


    collate_fn = train_collate_clear_ansonly if args.ans_only else train_collate_clear
    print(f"Answer-only loss: {args.ans_only}", flush=True)

    if processor:
        train_dataloader_forget = DataLoader(
            multimodal_forget_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, processor, model.device,True)
        )

        train_dataloader_retain = DataLoader(
            multimodal_retain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, processor, model.device,True)
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")
    module_set=set()
    trainable_grad_mask = None
    if args.grad_mask_path:
        grad_data=torch.load(args.grad_mask_path)
        grad_mask=grad_data['weight']
        layer_name_list=list(grad_mask.keys())
        for name in layer_name_list:
            if "proj" in name:
                continue
            elif "fc" in name:
                continue
            elif "linear" in name:
                continue
            elif "mlp" in name:
                continue
            elif "qkv" in name:
                continue
            else:
                grad_mask.pop(name)
        def calc_sparsity(tensor):
            num_zero_elements = tensor.numel() - torch.count_nonzero(tensor)
            total_elements = tensor.numel()
            sparsity = num_zero_elements / total_elements
            return sparsity.item(), total_elements, num_zero_elements

        total_cnt=0
        w_cnt=0
        for k,v in grad_mask.items():
            try: 
                w_sparsity, total_elements, w_num_zero_elements = calc_sparsity(v)
                total_cnt += total_elements
                w_cnt += w_num_zero_elements
                module_set=module_set|set(k.split("."))
            except: 
                pass 
        print("Saliency mask generated!")
        print(f"Total sparsity among weight:{w_cnt/total_cnt*100}")
        print_mask_coverage(model, grad_mask)
        trainable_grad_mask = build_trainable_grad_mask(model, grad_mask)
        if args.mask_check_only:
            print("mask_check_only=True; exiting before training.", flush=True)
            return

    print(f"Fine-tuned modules: {module_set}\n",)

    accelerator = Accelerator()
    accumulation_steps = args.gradient_accumulation_steps if args.gradient_accumulation else 1
    if args.gradient_accumulation:
        print(
            f"Gradient accumulation enabled: {accumulation_steps} micro-batches/update.",
            flush=True,
        )
    else:
        print("Gradient accumulation disabled.", flush=True)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    len_retain_raw=len(train_dataloader_retain)
    len_forget_raw=len(train_dataloader_forget)
    n_iters_raw=len_retain_raw
    forget_freq_raw=max(1, len_retain_raw//len_forget_raw)
    forget_steps_per_epoch=(n_iters_raw + forget_freq_raw - 1)//forget_freq_raw
    if args.gradient_accumulation:
        optimizer_steps_per_epoch=(n_iters_raw + accumulation_steps - 1)//accumulation_steps
    else:
        optimizer_steps_per_epoch=n_iters_raw + forget_steps_per_epoch
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=optimizer_steps_per_epoch * args.num_epochs,
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader_forget, train_dataloader_retain, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader_forget, train_dataloader_retain, lr_scheduler
    )
    len_retain,len_forget=len(train_dataloader_retain),len(train_dataloader_forget)
    n_iters=len(train_dataloader_retain)
    forget_freq=max(1, len(train_dataloader_retain)//len(train_dataloader_forget))
    print(f"Each epoch have {n_iters} iterations, with forget_freq={forget_freq}!")

    # Training loop with gradient ascent
    for epoch in range(args.num_epochs):
        model.train()
        total_loss_forget = 0
        total_loss_retain = 0
        forget_updates = 0
        retain_updates = 0
        train_dataloader_forget_iter=iter(train_dataloader_forget)
        train_dataloader_retain_iter=iter(train_dataloader_retain)
        if args.gradient_accumulation:
            optimizer.zero_grad()
            progress = tqdm(
                range(0,n_iters),
                desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            )
            optimizer_updates = 0
            for step_idx in progress:
                last_forget_loss = None
                combined_loss = None
                if step_idx % forget_freq == 0: # update only in specific forget round
                    try:
                        forget_batch=next(train_dataloader_forget_iter)
                    except StopIteration:
                        train_dataloader_forget_iter=iter(train_dataloader_forget)
                        forget_batch=next(train_dataloader_forget_iter)
                    forget_outputs = model(**forget_batch)
                    raw_forget_loss = forget_outputs.loss
                    forget_objective = -args.forget_alpha*raw_forget_loss
                    combined_loss = forget_objective
                    last_forget_loss = raw_forget_loss.item()
                    total_loss_forget += last_forget_loss
                    forget_updates += 1

                try:
                    retain_batch=next(train_dataloader_retain_iter)
                except StopIteration:
                    train_dataloader_retain_iter=iter(train_dataloader_retain)
                    retain_batch=next(train_dataloader_retain_iter)
                retain_outputs = model(**retain_batch)
                loss_retain = retain_outputs.loss
                if loss_retain>1e+10:
                    torch.save(retain_batch,"Ill_batch.pt")
                    output = model.generate(**retain_batch, max_new_tokens=30, do_sample=False)
                    print(processor.decode(output[0], skip_special_tokens=True))
                    optimizer.zero_grad()
                    exit(-1)
                combined_loss = loss_retain if combined_loss is None else combined_loss + loss_retain
                accelerator.backward(combined_loss / accumulation_steps)
                total_loss_retain += loss_retain.item()
                retain_updates += 1

                should_step = (
                    ((step_idx + 1) % accumulation_steps == 0)
                    or (step_idx + 1 == n_iters)
                )
                if should_step:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if trainable_grad_mask:
                        apply_trainable_grad_mask(model, trainable_grad_mask)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    optimizer_updates += 1

                if (step_idx + 1) % max(1, n_iters // 20) == 0 or step_idx == 0:
                    forget_text = (
                        f"{last_forget_loss:.6f}"
                        if last_forget_loss is not None
                        else "no_update"
                    )
                    print(
                        f"Epoch {epoch + 1} Iteration {step_idx + 1} - "
                        f"Forget Loss: {forget_text} - "
                        f"Retain Loss: {loss_retain.item():.6f} - "
                        f"Optimizer Updates: {optimizer_updates}",
                        flush=True,
                    )
            print(
                f"Epoch {epoch + 1} - Optimizer Updates: {optimizer_updates}",
                flush=True,
            )
        else:
            progress = tqdm(
                range(0,n_iters),
                desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            )
            for step_idx in progress:
                last_forget_loss = None
                if step_idx % forget_freq == 0: # update only in specific forget round
                    try:
                        forget_batch=next(train_dataloader_forget_iter)
                    except StopIteration:
                        train_dataloader_forget_iter=iter(train_dataloader_forget)
                        forget_batch=next(train_dataloader_forget_iter)
                    forget_outputs = model(**forget_batch)
                    raw_forget_loss = forget_outputs.loss
                    loss_forget = -args.forget_alpha*raw_forget_loss  # Gradient ascent
                    accelerator.backward(loss_forget)
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if trainable_grad_mask:
                        apply_trainable_grad_mask(model, trainable_grad_mask)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    last_forget_loss = raw_forget_loss.item()
                    total_loss_forget += last_forget_loss
                    forget_updates += 1

                try:
                    retain_batch=next(train_dataloader_retain_iter)
                except StopIteration:
                    train_dataloader_retain_iter=iter(train_dataloader_retain)
                    retain_batch=next(train_dataloader_retain_iter)
                retain_outputs = model(**retain_batch)
                loss_retain = retain_outputs.loss
                if loss_retain>1e+10:
                    torch.save(retain_batch,"Ill_batch.pt")
                    output = model.generate(**retain_batch, max_new_tokens=30, do_sample=False)
                    print(processor.decode(output[0], skip_special_tokens=True))
                    optimizer.zero_grad()
                    exit(-1)
                # print(retain_batch.keys(),retain_outputs.keys())
                accelerator.backward(loss_retain)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                total_loss_retain += loss_retain.item()
                retain_updates += 1

                if (step_idx + 1) % max(1, n_iters // 20) == 0 or step_idx == 0:
                    forget_text = (
                        f"{last_forget_loss:.6f}"
                        if last_forget_loss is not None
                        else "no_update"
                    )
                    print(
                        f"Epoch {epoch + 1} Iteration {step_idx + 1} - "
                        f"Forget Loss: {forget_text} - "
                        f"Retain Loss: {loss_retain.item():.6f}",
                        flush=True,
                    )
        avg_forget_loss = total_loss_forget / max(1, forget_updates)
        avg_retain_loss = total_loss_retain / max(1, retain_updates)
        print(
            f"Epoch {epoch + 1} - Forget Loss: {avg_forget_loss:.6f} "
            f"({forget_updates} updates)",
            flush=True,
        )
        print(
            f"Epoch {epoch + 1} - Retain Loss: {avg_retain_loss:.6f} "
            f"({retain_updates} updates)",
            flush=True,
        )      

    # Save the final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if not isinstance(unwrapped_model, PeftModel):
        raise TypeError("Expected a PEFT model; refusing to save full model weights.")
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(args.save_dir, safe_serialization=True)
        print(f"LoRA adapter saved to: {args.save_dir}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(
            "CUDA_VISIBLE_DEVICES was set inside Python. Prefer exporting it "
            "in the shell before launch so CUDA sees it before torch import: "
            f"{args.cuda_visible_devices}",
            flush=True,
        )
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
    trainer_config = vars(args).copy()
    trainer_config.update({"save_format": "peft_adapter", "base_model_path": args.vanilla_dir})
    with open(f"{args.save_dir}/trainer_config.json", 'wt') as f:
        json.dump(trainer_config, f, indent=4)
