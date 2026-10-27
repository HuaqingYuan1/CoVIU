import sys
import os
import math
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from tqdm import tqdm
from peft import PeftModel
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
import torch
import json
import argparse
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, get_scheduler, AdamW, MllamaForConditionalGeneration, AutoTokenizer,Qwen2VLForConditionalGeneration
# from qwen_vl_utils import process_vision_info
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from data_process.data_preprocess import LLAVA_multimodal_Dataset, train_collate_fn_mllmu, train_collate_mllmu_ansonly,Vanilla_LLaVA_Dataset
from data_process.CLEAR_process import CLEAR_Dataset, CAPTION_MODE, RECOGNITION_MODE, train_collate_clear, NONE_MODE,train_collate_clear_ansonly
from accelerate import Accelerator
import torch
try:
    from baselines.config_utils import add_dataset_args, load_clear_split, parse_args_with_yaml, str2bool
except ModuleNotFoundError:
    from config_utils import add_dataset_args, load_clear_split, parse_args_with_yaml, str2bool


class ForgetVQAToQADataset(Dataset):
    def __init__(self, df):
        self.dataset = []
        for _, row in df.iterrows():
            try:
                metadata = json.loads(row["metadata"])
            except json.JSONDecodeError:
                continue
            name = self._extract_name(metadata)
            if not name:
                continue
            for qa_pair in metadata:
                question = qa_pair.get("Question") or qa_pair.get("Additional_question") or ""
                answer = qa_pair.get("Answer") or qa_pair.get("Additional_answer") or ""
                question = self._rewrite_with_name(question, name)
                answer = self._rewrite_with_name(answer, name)
                self._append_text_qa(question, answer)

    @staticmethod
    def _extract_name(metadata):
        for qa_pair in metadata:
            question = (qa_pair.get("Question") or "").lower()
            answer = qa_pair.get("Answer") or ""
            if "name" not in question or not answer:
                continue
            for marker in (
                " is ",
                " is named ",
                " is called ",
                "name associated with this profile is ",
            ):
                if marker in answer:
                    name = answer.split(marker)[-1].strip().strip(".")
                    if name:
                        return name
        return None

    @staticmethod
    def _rewrite_with_name(text, name):
        replacements = (
            ("this person in the image", name),
            ("the person in the image", name),
            ("this individual in the image", name),
            ("the individual in the image", name),
            ("this person", name),
            ("the person", name),
            ("this individual", name),
            ("the individual", name),
            ("this profile", name),
        )
        rewritten = text
        for old, new in replacements:
            rewritten = rewritten.replace(old, new)
            rewritten = rewritten.replace(old.capitalize(), new)
        return rewritten

    def _append_text_qa(self, question, answer):
        if question and answer:
            self.dataset.append({
                "image": None,
                "question": question,
                "answer": answer,
            })

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


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

    # dataset = Vanilla_LLaVA_Dataset_baseline(json_dir=profile_dir, image_dir=image_base_path, flatten=False)
    # print(f"Dataset size (profiles): {len(dataset)}")

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
    if args.retain_forget_vqa_as_qa and args.dataset_type == "mllmu":
        forget_qa_retain_dataset = ForgetVQAToQADataset(df=forget_df)
        multimodal_retain_dataset = ConcatDataset([
            multimodal_retain_dataset,
            forget_qa_retain_dataset,
        ])
        print(
            "Retain dataset includes baseline retain split plus "
            f"{len(forget_qa_retain_dataset)} text-only QA samples rewritten from forget VQA metadata."
        )

    if args.dataset_type == "clear":
        train_collate_function = train_collate_clear_ansonly if args.ans_only else train_collate_clear
    else:
        train_collate_function = train_collate_mllmu_ansonly if args.ans_only else train_collate_fn_mllmu


    if processor is not None:
        train_dataloader_forget = DataLoader(
            multimodal_forget_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_function(x, processor, model.device, True)
        )

        train_dataloader_retain = DataLoader(
            multimodal_retain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_function(x, processor,model.device, True)
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    accelerator = Accelerator()
    if args.gradient_accumulation:
        print("Gradient accumulation enabled.")
        accumulation_steps = args.accumulation_steps
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False
    else:
        print("Gradient accumulation disabled.")
        accumulation_steps = 1

    optimizer = AdamW(model.parameters(), lr=args.lr)
    updates_per_epoch = math.ceil(len(train_dataloader_retain) / accumulation_steps)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=updates_per_epoch * args.num_epochs,
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader_forget, train_dataloader_retain, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader_forget, train_dataloader_retain, lr_scheduler
    )

    len_retain, len_forget = len(train_dataloader_retain), len(train_dataloader_forget)
    n_iters = len(train_dataloader_retain)
    forget_freq = max(1, len(train_dataloader_retain) // len(train_dataloader_forget))
    print(f"Each epoch have {n_iters} iterations, with forget_freq={forget_freq}!")

    # Training loop with gradient ascent
    for epoch in range(args.num_epochs):
        train_data_forget = enumerate(train_dataloader_forget)
        train_data_retain = enumerate(train_dataloader_retain)  # to iterate manually
        model.train()
        total_loss_forget = 0
        total_loss_retain = 0
        optimizer.zero_grad()
        for iter in tqdm(range(0, n_iters)):
            step_loss = None
            raw_forget_loss = None

            if iter % forget_freq == 0:  # update only in specific forget round
                try:
                    _, forget_batch = next(train_data_forget)
                except StopIteration:
                    train_data_forget = enumerate(train_dataloader_forget)
                    _, forget_batch = next(train_data_forget)
                forget_outputs = model(**forget_batch)
                raw_forget_loss = forget_outputs.loss
                step_loss = -raw_forget_loss  # Gradient ascent on forget data.
                total_loss_forget += raw_forget_loss.item()

            _, retain_batch = next(train_data_retain)
            retain_outputs = model(**retain_batch)
            loss_retain = retain_outputs.loss
            step_loss = loss_retain if step_loss is None else step_loss + loss_retain
            total_loss_retain += loss_retain.item()

            backward_loss = step_loss
            if args.gradient_accumulation:
                backward_loss = backward_loss / accumulation_steps
            if not backward_loss.requires_grad:
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                raise RuntimeError(
                    "Training loss is detached from the graph. "
                    f"loss={backward_loss.item()}, trainable_params={trainable_params}"
                )
            accelerator.backward(backward_loss)

            should_step = (
                not args.gradient_accumulation
                or (iter + 1) % accumulation_steps == 0
                or (iter + 1) == n_iters
            )
            if should_step:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if raw_forget_loss is not None:
                print(f"Iteration {iter + 1} - Forget Loss: {raw_forget_loss.item()}")
            print(f"Iteration {iter + 1} - Retain Loss: {loss_retain.item()}")

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
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--data_split_dir", type=str, default="../Data_split", help="Directory to save the model")
    parser.add_argument("--forget_split_ratio", type=int, default=5, help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=str2bool, default=False, help="Enable gradient accumulation")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--trainer", type=str2bool, default=False, help="Use HuggingFace Trainer")
    parser.add_argument("--ans_only", type=str2bool, default=False, help="Answer only for training")
    parser.add_argument("--retain_forget_vqa_as_qa", type=str2bool, default=False, help="Add text-only QA samples rewritten from forget VQA metadata to retain training.")
    add_dataset_args(parser)

    args = parse_args_with_yaml(parser, required=("model_id", "vanilla_dir"))

    main(args)
    trainer_config = vars(args).copy()
    trainer_config.update({"save_format": "peft_adapter", "base_model_path": args.vanilla_dir})
    with open(f"{args.save_dir}/trainer_config.json", 'wt') as f:
        json.dump(trainer_config, f, indent=4)
