import sys
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from peft import PeftModel
import random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
import torch
import json
import argparse
from PIL import Image
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, MllamaForConditionalGeneration, AutoTokenizer,Qwen2VLForConditionalGeneration, set_seed as transformers_set_seed
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
    parser = argparse.ArgumentParser(description="Reproduce ViKeR on MLLMU-Bench")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--model_id", type=str, default=None, help="Processor/model ID")
    parser.add_argument("--vanilla_dir", type=str, default=None, help="Base model directory")
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--data_split_dir", type=str, default="../Data_split", help="Data split directory")
    parser.add_argument("--dataset_type", choices=("mllmu", "clear"), default="mllmu")
    parser.add_argument("--clear_data_dir", type=str, default="data/CLEAR")
    parser.add_argument("--forget_split_ratio", type=int, default=10, help="Forget split ratio")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=str2bool, default=False, help="Enable gradient accumulation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Reserved; gradient accumulation is not implemented in this script")
    parser.add_argument("--trainer", type=str2bool, default=False, help="Use HuggingFace Trainer")
    parser.add_argument("--grad_mask_path", type=str, default=None, help="Mask for gradient update")
    parser.add_argument("--mask_dir", type=str, default=None, help="Optional mask directory recorded in trainer_config.json")
    parser.add_argument("--viker_lambda", type=float, default=0.05, help="KL regularization weight for ViKeR")
    parser.add_argument("--viker_num_refs", type=int, default=5, help="Number of irrelevant reference images per forget sample")
    parser.add_argument("--reference_image_dir", type=str, default=None, help="Directory containing CelebA images used as irrelevant ViKeR references")
    parser.add_argument("--ans_only", type=str2bool, default=True, help="Only compute LM loss on answer tokens")
    parser.add_argument("--language_only", type=str2bool, default=True, help="Restrict training to the language model (freeze vision tower and multimodal projector)")
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
    if not args.reference_image_dir:
        raise ValueError("--reference_image_dir is required for paper-faithful ViKeR.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be >= 1.")
    if args.viker_num_refs < 1:
        raise ValueError("--viker_num_refs must be >= 1.")
    if args.gradient_accumulation_steps > 1:
        args.gradient_accumulation = True
    return args


def find_language_linear_names(model):
    """Return exact Linear module paths belonging to LLaVA's language model.

    Using leaf names such as ``q_proj`` is unsafe: PEFT suffix matching would
    also inject adapters into similarly named layers in the vision tower.
    """
    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and "language_model" in name.split(".")
        and not name.endswith("lm_head")
    ]
    if not names:
        raise ValueError("No language-model Linear layers were found for LoRA.")
    return names


def enforce_language_only_trainable(model):
    """Freeze every trainable parameter outside the language-model subtree."""
    frozen = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and "language_model" not in name.split("."):
            parameter.requires_grad_(False)
            frozen.append(name)

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    unexpected = [name for name in trainable if "language_model" not in name.split(".")]
    if unexpected:
        raise RuntimeError(
            "language_only=True but non-language parameters remain trainable: "
            + ", ".join(unexpected[:10])
        )
    if not trainable:
        raise RuntimeError("language_only=True left no trainable parameters.")
    print(
        f"Language-only training verified: {len(trainable)} trainable tensors; "
        f"froze {len(frozen)} non-language tensors.",
        flush=True,
    )

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
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=(
                find_language_linear_names(model)
                if args.language_only
                else "all-linear"
            ),
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


def collate_with_examples(examples, processor, device, ans_only):
    """Build a train batch while keeping raw examples for ViKeR references."""
    collate_fn = train_collate_clear_ansonly if ans_only else train_collate_clear
    return collate_fn(examples, processor, device, True), examples


class IrrelevantImageDataset(Dataset):
    """Unlabelled CelebA images used by ViKeR as visual references."""

    def __init__(self, image_dir):
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(
                f"CelebA reference image directory does not exist: {image_dir}"
            )
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        self.image_paths = sorted(
            os.path.join(root, filename)
            for root, _, filenames in os.walk(image_dir)
            for filename in filenames
            if os.path.splitext(filename)[1].lower() in extensions
        )
        if not self.image_paths:
            raise ValueError(f"No reference images found under: {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]) as image:
            return {"image": image.convert("RGB")}


def make_reference_examples(forget_examples, irrelevant_examples, num_refs):
    if not irrelevant_examples:
        raise ValueError("ViKeR needs at least one irrelevant visual reference.")
    reference_examples = []
    for forget_idx, forget_example in enumerate(forget_examples):
        for ref_idx in range(num_refs):
            irrelevant_example = irrelevant_examples[(forget_idx * num_refs + ref_idx) % len(irrelevant_examples)]
            reference_examples.append(
                {
                    "image": irrelevant_example.get("image"),
                    "question": forget_example.get("question"),
                    "answer": forget_example.get("answer"),
                }
            )
    return reference_examples


def next_reference_examples(reference_iter, reference_dataloader, count):
    """Draw enough irrelevant visual examples for ViKeR references."""
    examples = []
    while len(examples) < count:
        try:
            examples.extend(next(reference_iter))
        except StopIteration:
            reference_iter = iter(reference_dataloader)
            examples.extend(next(reference_iter))
    return examples[:count], reference_iter


def clone_batch_to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def disable_adapter_if_possible(model):
    if hasattr(model, "module") and hasattr(model.module, "disable_adapter"):
        return model.module.disable_adapter()
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return torch.no_grad()


def viker_token_kl_loss(model, forget_batch, reference_batch, num_refs):
    labels = forget_batch["labels"]
    current_outputs = model(**forget_batch)

    with torch.no_grad():
        with disable_adapter_if_possible(model):
            reference_outputs = model(**reference_batch)

    current_logits = current_outputs.logits[:, :-1, :]
    reference_logits = reference_outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    batch_size = current_logits.shape[0]
    if reference_logits.shape[0] != batch_size * num_refs:
        raise ValueError(
            "Reference batch size must equal forget_batch_size * viker_num_refs: "
            f"{reference_logits.shape[0]} != {batch_size} * {num_refs}"
        )

    seq_len = min(current_logits.shape[1], reference_logits.shape[1], shift_labels.shape[1])
    current_logits = current_logits[:, :seq_len, :]
    reference_logits = reference_logits[:, :seq_len, :]
    shift_labels = shift_labels[:, :seq_len]

    current_log_probs = torch.nn.functional.log_softmax(current_logits.float(), dim=-1)
    reference_probs = torch.nn.functional.softmax(reference_logits.float(), dim=-1)
    vocab_size = reference_probs.shape[-1]
    target_probs = reference_probs.view(batch_size, num_refs, seq_len, vocab_size).mean(dim=1)
    target_log_probs = torch.log(target_probs.clamp_min(1e-8))

    token_kl = torch.nn.functional.kl_div(
        current_log_probs,
        target_probs,
        reduction="none",
        log_target=False,
    ).sum(dim=-1)
    token_mask = shift_labels.ne(-100)
    kl_loss = (token_kl * token_mask).sum() / token_mask.sum().clamp_min(1)

    entropy = -(target_probs * target_log_probs).sum(dim=-1)
    mean_entropy = (entropy * token_mask).sum() / token_mask.sum().clamp_min(1)
    return current_outputs.loss, kl_loss, mean_entropy



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

    # Resizing embeddings can create a fresh trainable parameter, so enforce
    # the module boundary only after all vocabulary resizing is complete.
    if args.language_only:
        enforce_language_only_trainable(model)
    
    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    # Dataset and Dataloader setup

    if args.dataset_type == "clear":
        forget_folder = os.path.join(args.clear_data_dir, f"forget{args.forget_split_ratio:02d}")
    else:
        forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    print("Forget Folder: ", forget_folder)
    print("CelebA reference image directory: ", args.reference_image_dir)

    # ViKeR only trains on D_f.  The separate celebrity set supplies unlabelled
    # irrelevant images for gradient-free reference inference; it is not a
    # retain training set and contributes no supervised loss.
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")

    if args.dataset_type == "clear":
        forget_df = load_clear_split(args.clear_data_dir, "forget", args.forget_split_ratio)
        multimodal_forget_dataset = CLEAR_Dataset(data=forget_df, mode=CAPTION_MODE)
    else:
        forget_df = pd.read_parquet(forget_parquet_file)
        multimodal_forget_dataset = LLAVA_multimodal_Dataset(df=forget_df)
    irrelevant_image_dataset = IrrelevantImageDataset(args.reference_image_dir)


    print(f"Answer-only loss: {args.ans_only}", flush=True)

    if processor:
        train_dataloader_forget = DataLoader(
            multimodal_forget_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_with_examples(x, processor, model.device, args.ans_only)
        )

        reference_dataloader = DataLoader(
            irrelevant_image_dataset,
            batch_size=args.batch_size * args.viker_num_refs,
            shuffle=True,
            collate_fn=lambda x: x,
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
    print(f"Gradient accumulation steps: {accumulation_steps}", flush=True)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)

    model, optimizer, train_dataloader_forget = accelerator.prepare(
        model, optimizer, train_dataloader_forget
    )
    n_iters = len(train_dataloader_forget)
    print(f"Each epoch has {n_iters} forget-set batches.", flush=True)

    # Eq. (5): L_ViKeR = -L_NLL(D_f) + lambda * KL(R || Q).
    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad()
        reference_iter = iter(reference_dataloader)
        total_nll = 0.0
        total_kl = 0.0
        total_objective = 0.0
        optimizer_updates = 0

        progress = tqdm(
            enumerate(train_dataloader_forget),
            total=n_iters,
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
        )
        for step_idx, (forget_batch, forget_examples) in progress:
            irrelevant_examples, reference_iter = next_reference_examples(
                reference_iter,
                reference_dataloader,
                len(forget_examples) * args.viker_num_refs,
            )
            reference_examples = make_reference_examples(
                forget_examples,
                irrelevant_examples,
                args.viker_num_refs,
            )
            reference_batch = collate_with_examples(
                reference_examples,
                processor,
                forget_batch["input_ids"].device,
                args.ans_only,
            )[0]
            reference_batch = clone_batch_to_device(
                reference_batch, forget_batch["input_ids"].device
            )

            nll_loss, viker_kl, ref_entropy = viker_token_kl_loss(
                model,
                forget_batch,
                reference_batch,
                args.viker_num_refs,
            )
            objective = -nll_loss + args.viker_lambda * viker_kl
            accelerator.backward(objective / accumulation_steps)

            should_step = (
                (step_idx + 1) % accumulation_steps == 0
                or step_idx + 1 == n_iters
            )
            if should_step:
                if trainable_grad_mask:
                    apply_trainable_grad_mask(model, trainable_grad_mask)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_updates += 1

            total_nll += nll_loss.item()
            total_kl += viker_kl.item()
            total_objective += objective.item()
            progress.set_postfix(
                nll=f"{nll_loss.item():.4f}",
                kl=f"{viker_kl.item():.4f}",
                entropy=f"{ref_entropy.item():.4f}",
                objective=f"{objective.item():.4f}",
            )

        print(
            f"Epoch {epoch + 1}: NLL={total_nll / n_iters:.6f}, "
            f"KL={total_kl / n_iters:.6f}, "
            f"ViKeR={total_objective / n_iters:.6f}, "
            f"optimizer_updates={optimizer_updates}",
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
