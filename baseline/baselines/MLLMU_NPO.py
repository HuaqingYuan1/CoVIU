import sys
import os
import pandas as pd
from torch.utils.data import DataLoader, random_split, RandomSampler
from tqdm import tqdm
from peft import PeftModel
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
import torch
import json
import argparse
import torch
import math
from transformers import LlavaForConditionalGeneration, AutoProcessor, get_scheduler, AdamW, MllamaForConditionalGeneration, AutoTokenizer,Qwen2VLForConditionalGeneration
# from qwen_vl_utils import process_vision_info
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from data_process.data_preprocess import LLAVA_multimodal_Dataset, train_collate_fn_mllmu, train_collate_mllmu_ansonly,Vanilla_LLaVA_Dataset
from data_process.CLEAR_process import CLEAR_Dataset, CAPTION_MODE, RECOGNITION_MODE, train_collate_clear, NONE_MODE,train_collate_clear_ansonly
from accelerate import Accelerator
import torch
import torch.nn.functional as F
try:
    from baselines.config_utils import add_dataset_args, load_clear_split, parse_args_with_yaml, str2bool, subsample_dataset
except ModuleNotFoundError:
    from config_utils import add_dataset_args, load_clear_split, parse_args_with_yaml, str2bool, subsample_dataset


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
        # LoRA configuration
        lora_config = LoraConfig(
            r=16, #32
            lora_alpha=16, #8
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )

        print("getting peft model")
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
        # LoRA configuration
        lora_config = LoraConfig(
            r=16, #32
            lora_alpha=16, #8
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )

        print("getting peft model")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

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
        # LoRA configuration
        lora_config = LoraConfig(
            r=16, #32
            lora_alpha=16, #8
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )

        print("getting peft model")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    return model, processor



######################### Accelerate Version #################################
def main(args):
    # Load model and processor
    print("Trainer Status is ", args.trainer)
    model, processor = load_model_and_processor(args)

    if "llava" in args.model_id:
        # Load LLAVA model and processor
        print("Loading Oracle LLAVA model...")
        oracle_model = LlavaForConditionalGeneration.from_pretrained(
            args.oracle_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    elif "llama" in args.model_id.lower():
        # Load LLAVA Next model and processor
        print("Loading Oracle llama model...")
        oracle_model = MllamaForConditionalGeneration.from_pretrained(
            args.oracle_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    elif "qwen" in args.model_id.lower():
        # Load LLAVA Next model and processor
        print("Loading Oracle qwen model...")
        oracle_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.oracle_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )


    print("Processor Tokenizer Length: ", len(processor.tokenizer)) #128257
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print("Tokenizer Length: ", len(tokenizer))

    # Resize token embeddings to match the tokenizer
    # if args.models_id.startswith("meta-llama") == False:
    #     model.resize_token_embeddings(len(processor.tokenizer))

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    if args.dataset_type == "clear":
        forget_folder = os.path.join(args.clear_data_dir, f"forget{args.forget_split_ratio:02d}")
        retain_folder = os.path.join(args.clear_data_dir, f"retain{100 - args.forget_split_ratio}")
    else:
        forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
        retain_folder = os.path.join(args.data_split_dir, f"retain_{100 - args.forget_split_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)

    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_folder, args.forget_file)
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")

    # Load DataLoader
    if args.dataset_type == "clear":
        forget_df = load_clear_split(args.clear_data_dir, "forget", args.forget_split_ratio)
        retain_df = load_clear_split(args.clear_data_dir, "retain", args.forget_split_ratio)
    else:
        forget_df = pd.read_parquet(forget_parquet_file)
        retain_df = pd.read_parquet(retain_parquet_file)
    retain_df = subsample_dataset(retain_df, args.retain_fraction, args.retain_max_samples, args.seed)
    print(f"Forget samples: {len(forget_df)}")
    print(f"Retain samples: {len(retain_df)}")

    if args.dataset_type == "clear":
        multimodal_forget_dataset = CLEAR_Dataset(data=forget_df, mode=CAPTION_MODE)
        multimodal_retain_dataset = CLEAR_Dataset(data=retain_df, mode=CAPTION_MODE)
    else:
        multimodal_forget_dataset = LLAVA_multimodal_Dataset(df=forget_df)
        multimodal_retain_dataset = LLAVA_multimodal_Dataset(df=retain_df)

    if args.dataset_type == "clear":
        train_collate_function = train_collate_clear_ansonly if args.ans_only else train_collate_clear
    else:
        train_collate_function = train_collate_mllmu_ansonly if args.ans_only else train_collate_fn_mllmu

    if processor:
        train_dataloader = DataLoader(
            multimodal_forget_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_function(x, processor, model.device,True)
        )
        retain_sampler = None
        retain_shuffle = True
        if args.retain_dynamic_sample:
            retain_sampler = RandomSampler(
                multimodal_retain_dataset,
                replacement=True,
                num_samples=len(multimodal_forget_dataset),
            )
            retain_shuffle = False
        train_dataloader_retain = DataLoader(
            multimodal_retain_dataset,
            batch_size=args.batch_size,
            shuffle=retain_shuffle,
            sampler=retain_sampler,
            collate_fn=lambda x: train_collate_function(x, processor, model.device,True)
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    # Accelerator setup
    accelerator = Accelerator()
    accumulation_steps = 1
    if args.gradient_accumulation:
        print("Gradient accumulation enabled.")
        accumulation_steps = args.gradient_accumulation_steps
        model.gradient_checkpointing_enable()
    else:
        print("Gradient accumulation disabled.")

    optimizer = AdamW(model.parameters(), lr=args.lr)

    n_iters = (
        len(train_dataloader_retain)
        if args.retain_full_epoch and args.retain_weight > 0 and not args.retain_dynamic_sample
        else len(train_dataloader)
    )
    optimizer_steps_per_epoch = math.ceil(n_iters / accumulation_steps)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=optimizer_steps_per_epoch * args.num_epochs,
    )

    oracle_model, model, optimizer, train_dataloader, train_dataloader_retain, lr_scheduler = accelerator.prepare(
        oracle_model, model, optimizer, train_dataloader, train_dataloader_retain, lr_scheduler
    )

    for epoch in range(args.num_epochs):
        forget_iter = iter(train_dataloader)
        retain_iter = iter(train_dataloader_retain)
        model.train()
        total_loss = 0
        total_npo_loss = 0
        total_retain_loss = 0
        progress_bar = tqdm(range(n_iters), desc=f"Epoch {epoch + 1}")

        if args.gradient_accumulation:
            optimizer.zero_grad()
            for step in progress_bar:
                try:
                    forget_batch = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(train_dataloader)
                    forget_batch = next(forget_iter)

                outputs = model(**forget_batch)
                current_loss = outputs.loss

                with torch.no_grad():
                    oracle_outputs = oracle_model(**forget_batch)
                    oracle_loss = oracle_outputs.loss

                neg_log_ratios = current_loss - oracle_loss
                npo_loss = -F.logsigmoid(args.beta * neg_log_ratios).mean() * 2 / args.beta
                loss = npo_loss

                retain_loss = torch.zeros_like(loss)
                if args.retain_weight > 0:
                    try:
                        retain_batch = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(train_dataloader_retain)
                        retain_batch = next(retain_iter)
                    retain_outputs = model(**retain_batch)
                    retain_loss = retain_outputs.loss
                    loss = loss + args.retain_weight * retain_loss

                accelerator.backward(loss / accumulation_steps)

                if (step + 1) % accumulation_steps == 0 or (step + 1) == n_iters:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                total_loss += loss.item()
                total_npo_loss += npo_loss.item()
                total_retain_loss += retain_loss.item()
                progress_bar.set_postfix(loss=total_loss / (step + 1))

        else:
            for step in progress_bar:
                try:
                    forget_batch = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(train_dataloader)
                    forget_batch = next(forget_iter)

                # Forward pass with the current model to get the loss
                outputs = model(**forget_batch)
                current_loss = outputs.loss

                # Forward pass with the oracle model to get the oracle loss
                with torch.no_grad():
                    oracle_outputs = oracle_model(**forget_batch)
                    oracle_loss = oracle_outputs.loss

                # Compute neg_log_ratios and NPO loss
                neg_log_ratios = current_loss - oracle_loss
                npo_loss = -F.logsigmoid(args.beta * neg_log_ratios).mean() * 2 / args.beta
                loss = npo_loss

                retain_loss = torch.zeros_like(loss)
                if args.retain_weight > 0:
                    try:
                        retain_batch = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(train_dataloader_retain)
                        retain_batch = next(retain_iter)
                    retain_outputs = model(**retain_batch)
                    retain_loss = retain_outputs.loss
                    loss = loss + args.retain_weight * retain_loss

                # Backward pass and optimization
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                total_loss += loss.item()
                total_npo_loss += npo_loss.item()
                total_retain_loss += retain_loss.item()
                progress_bar.set_postfix(loss=total_loss / (step + 1))

        print(
            f"Epoch {epoch + 1} Loss: {total_loss / n_iters} "
            f"- NPO Loss: {total_npo_loss / n_iters} "
            f"- Retain Loss: {total_retain_loss / n_iters}"
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
    # Argument parser for different options
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--model_id", type=str, default=None, help="Pretrained model ID")
    parser.add_argument("--vanilla_dir", type=str, default=None, help="Pretrained model ID")
    parser.add_argument("--oracle_model_id", type=str, default=None, help="Oracle model ID")
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--data_split_dir", type=str, default="../Data_split", help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--forget_split_ratio", type=int, default=5, help="Directory to save the model")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=str2bool, default=False, help="Enable gradient accumulation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--trainer", type=str2bool, default=False, help="Use HuggingFace Trainer")
    parser.add_argument("--forget_file", type=str, default="train-00000-of-00001.parquet", help="Forget file name")
    parser.add_argument("--ans_only", type=str2bool, default=False, help="Answer only for training")
    parser.add_argument("--retain_weight", type=float, default=0.0, help="Weight for retain-set SFT loss")
    parser.add_argument("--retain_fraction", type=float, default=1.0, help="Fraction of retain data to use")
    parser.add_argument("--retain_max_samples", type=int, default=0, help="If > 0, cap retain data to this many samples")
    parser.add_argument("--retain_full_epoch", type=str2bool, default=False, help="Train for one retain dataloader epoch and cycle forget data")
    parser.add_argument("--retain_dynamic_sample", type=str2bool, default=False, help="Randomly sample retain batches from the full retain set for the same number of samples as forget data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for retain subsampling")
    add_dataset_args(parser)

    args = parse_args_with_yaml(
        parser,
        required=("model_id", "vanilla_dir", "oracle_model_id"),
    )
    main(args)
    trainer_config = vars(args).copy()
    trainer_config.update({"save_format": "peft_adapter", "base_model_path": args.vanilla_dir})
    with open(f"{args.save_dir}/trainer_config.json", 'wt') as f:
        json.dump(trainer_config, f, indent=4)
