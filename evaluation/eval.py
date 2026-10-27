"""Shared MLLMU-Bench zero-shot tasks and metrics for CoVIU and all baselines.

Task prompts, answer extraction, and metrics originate from the original CoVIU zero-shot evaluator.
Use run_eval.py for YAML configuration, validation, and model selection.
"""

import os
import json
import random
from PIL import Image
from tqdm import tqdm
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer, Idefics2ForConditionalGeneration
import pandas as pd
import random
import json
from PIL import Image
from io import BytesIO
# from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer, Idefics2ForConditionalGeneration, MllamaProcessor, MllamaForConditionalGeneration
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
import argparse
import fnmatch
import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


def load_and_combine_parquet_files(directory):
    # Get all Parquet files in the directory
    parquet_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]

    # Read and concatenate all Parquet files
    combined_df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
    return combined_df

def save_ids_to_json(parquet_file, output_folder, filename="ids.json"):
    """
    Extract IDs from a Parquet file and save them to a JSON file in the specified folder.

    Args:
        parquet_file (str): Path to the Parquet file containing the data.
        output_folder (str): Path to the folder where the JSON file will be saved.
        filename (str): Name of the JSON file. Defaults to "ids.json".
    """
    # Load the Parquet file into a DataFrame
    df = pd.read_parquet(parquet_file)

    # Extract the unique IDs
    ids = df['ID'].unique().tolist()

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Construct the full path to the JSON file
    output_json_file = os.path.join(output_folder, filename)

    # Save the IDs to a JSON file
    with open(output_json_file, 'w') as f:
        json.dump(ids, f)

    print(f"Saved IDs to {output_json_file}")

def compute_bleu(ground_truth, predicted_answer):
    """
    Compute the BLEU score between a ground truth and predicted answer using simple whitespace tokenization.

    Args:
        ground_truth (str): The correct reference answer.
        predicted_answer (str): The predicted answer from the model.

    Returns:
        float: The BLEU score.
    """
    # Use .split() to tokenize based on spaces
    reference = [ground_truth.split()]  # Reference needs to be a list of tokenized words
    hypothesis = predicted_answer.split()  # Hypothesis (predicted answer) is also tokenized

    # Use smoothing to handle cases where BLEU score could be 0 for short texts
    smoothing_function = SmoothingFunction().method1

    # Compute the BLEU score
    bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)

    return bleu_score


def format_accuracy_result(image_correct, image_total, text_correct, text_total):
    image_accuracy = (image_correct / image_total) * 100 if image_total > 0 else 0.0
    text_accuracy = (text_correct / text_total) * 100 if text_total > 0 else 0.0
    overall_correct = image_correct + text_correct
    overall_total = image_total + text_total
    overall_accuracy = (overall_correct / overall_total) * 100 if overall_total > 0 else 0.0
    return {
        "image_textual": {
            "correct": image_correct,
            "total": image_total,
            "accuracy": image_accuracy,
        },
        "pure_text": {
            "correct": text_correct,
            "total": text_total,
            "accuracy": text_accuracy,
        },
        "overall": {
            "correct": overall_correct,
            "total": overall_total,
            "accuracy": overall_accuracy,
        },
    }


def print_accuracy_result(task_name, result):
    print(f"{task_name} | Image-Textual: {result['image_textual']['correct']}/{result['image_textual']['total']} "
          f"({result['image_textual']['accuracy']:.2f}%)")
    print(f"{task_name} | Pure Text: {result['pure_text']['correct']}/{result['pure_text']['total']} "
          f"({result['pure_text']['accuracy']:.2f}%)")
    print(f"{task_name} | Overall: {result['overall']['correct']}/{result['overall']['total']} "
          f"({result['overall']['accuracy']:.2f}%)")


def build_accuracy_summary(results_data):
    summary = {}
    for split_name, split_results in results_data.items():
        summary[split_name] = {}
        split_overall_correct = 0
        split_overall_total = 0
        for task_name in ("classification", "fill_in_the_blank"):
            task_result = split_results.get(task_name, {})
            if not task_result:
                continue
            image = task_result.get("image_textual", {})
            text = task_result.get("pure_text", {})
            overall = task_result.get("overall")
            if not overall:
                overall = format_accuracy_result(
                    image.get("correct", 0),
                    image.get("total", 0),
                    text.get("correct", 0),
                    text.get("total", 0),
                )["overall"]
            summary[split_name][task_name] = {
                "image_textual": image,
                "pure_text": text,
                "overall": overall,
            }
            split_overall_correct += overall.get("correct", 0)
            split_overall_total += overall.get("total", 0)
        if split_overall_total > 0:
            summary[split_name]["overall"] = {
                "correct": split_overall_correct,
                "total": split_overall_total,
                "accuracy": (split_overall_correct / split_overall_total) * 100,
            }
    return summary


def _safe_accuracy(correct, total):
    return (correct / total) * 100 if total > 0 else 0.0


def _weighted_modality_acc_for_split(split_results):
    vqa_correct = 0
    vqa_total = 0
    qa_correct = 0
    qa_total = 0

    for task_name in ("classification", "fill_in_the_blank"):
        task_result = split_results.get(task_name, {})

        image = task_result.get("image_textual", {})
        vqa_correct += image.get("correct", 0)
        vqa_total += image.get("total", 0)

        text = task_result.get("pure_text", {})
        qa_correct += text.get("correct", 0)
        qa_total += text.get("total", 0)

    vqa_acc = _safe_accuracy(vqa_correct, vqa_total)
    qa_acc = _safe_accuracy(qa_correct, qa_total)
    return {
        "VQA Acc": vqa_acc,
        "QA Acc": qa_acc,
        "VQA correct": vqa_correct,
        "VQA total": vqa_total,
        "QA correct": qa_correct,
        "QA total": qa_total,
        "Gap (QA - VQA)": qa_acc - vqa_acc,
        "MinAcc": min(vqa_acc, qa_acc),
    }


def build_focus_metrics(results_data):
    split_alias = {
        "Forget Set Results": "Forget",
        "Test Set Results": "Test",
        "Retain Set (shared dataset) Results": "Retain(shared)",
        "Retain Set (real person) Results": "Retain(real person)",
    }

    by_split = {}
    for split_name, alias in split_alias.items():
        if split_name not in results_data:
            continue
        by_split[alias] = _weighted_modality_acc_for_split(results_data[split_name])

    table_columns = {}
    if "Forget" in by_split:
        table_columns["Forget VQA Acc"] = by_split["Forget"]["VQA Acc"]
        table_columns["Forget QA Acc"] = by_split["Forget"]["QA Acc"]
        table_columns["Forget Gap = QA - VQA"] = by_split["Forget"]["Gap (QA - VQA)"]
    if "Test" in by_split:
        table_columns["Test VQA Acc"] = by_split["Test"]["VQA Acc"]
        table_columns["Test QA Acc"] = by_split["Test"]["QA Acc"]
        table_columns["Test Gap = QA - VQA"] = by_split["Test"]["Gap (QA - VQA)"]
    if "Retain(shared)" in by_split:
        table_columns["Retain(shared) VQA Acc"] = by_split["Retain(shared)"]["VQA Acc"]
        table_columns["Retain(shared) QA Acc"] = by_split["Retain(shared)"]["QA Acc"]
        table_columns["Retain(shared) MinAcc = min(VQA, QA)"] = by_split["Retain(shared)"]["MinAcc"]
    if "Retain(real person)" in by_split:
        table_columns["Retain(real person) VQA Acc"] = by_split["Retain(real person)"]["VQA Acc"]
        table_columns["Retain(real person) QA Acc"] = by_split["Retain(real person)"]["QA Acc"]
        table_columns["Retain(real person) MinAcc = min(VQA, QA)"] = by_split["Retain(real person)"]["MinAcc"]

    return {
        "metric_definition": (
            "VQA Acc and QA Acc are weighted by correct/total over "
            "classification + fill_in_the_blank for each evaluated split."
        ),
        "by_split": by_split,
        "table_columns": table_columns,
    }


def print_accuracy_summary(summary):
    print("\n================ Accuracy Summary (Image-Textual / Pure Text / Overall) ================")
    for split_name, tasks in summary.items():
        print(f"\n[{split_name}]")
        for task_name, modalities in tasks.items():
            if task_name == "overall":
                overall = modalities
                print(
                    f"  {'overall':18s} | "
                    f"Combined: {overall['correct']:4d}/{overall['total']:<4d} ({overall['accuracy']:6.2f}%)"
                )
                continue
            image = modalities["image_textual"]
            text = modalities["pure_text"]
            overall = modalities["overall"]
            print(
                f"  {task_name:18s} | "
                f"Image-Textual: {image['correct']:4d}/{image['total']:<4d} ({image['accuracy']:6.2f}%) | "
                f"Pure Text: {text['correct']:4d}/{text['total']:<4d} ({text['accuracy']:6.2f}%) | "
                f"Overall: {overall['correct']:4d}/{overall['total']:<4d} ({overall['accuracy']:6.2f}%)"
            )
    print("============================================================================\n")


def evaluate_from_ids(id_json_file, question_folder, filename_pattern="*"):
    """
    Load IDs from the JSON file and find their corresponding evaluation question files with a specific filename pattern,
    then return a list of the loaded JSON files.
    """
    with open(id_json_file, 'r') as f:
        ids = json.load(f)

    json_files = []
    for filename in sorted(os.listdir(question_folder)):
        for id_ in ids:
            if filename.startswith(id_) and fnmatch.fnmatch(filename, filename_pattern):
                file_path = os.path.join(question_folder, filename)
                with open(file_path, 'r') as f:
                    json_files.append(json.load(f))
                break

    return json_files


def formulate_prompt_with_options(question, options):
    """
    Formulate the prompt by combining the question and its options.

    Args:
        question (str): The question text.
        options (dict): The options for the question (e.g., {"A": "Option A", "B": "Option B"}).

    Returns:
        str: The formulated prompt combining the question and options.
    """
    # Combine the question with the options
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    prompt = f"{question}\n{options_str}"
    return prompt


def formulate_prompt_with_options_llama(question, options):
    """
    Formulate the prompt by combining the question and its options.

    Args:
        question (str): The question text.
        options (dict): The options for the question (e.g., {"A": "Option A", "B": "Option B"}).

    Returns:
        str: The formulated prompt combining the question and options.
    """
    # Combine the question with the options
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    prompt = f"{question}\n####Choices:\n{options_str}"
    return prompt
def split_dataset(original_dataset, forget_percentage=0.3):
    forget_set_size = int(len(original_dataset) * forget_percentage)
    retain_set_size = len(original_dataset) - forget_set_size
    forget_set, retain_set = train_test_split(original_dataset, test_size=retain_set_size, random_state=42)
    return forget_set, retain_set

def load_json_files(question_folder):
    """
    Load all JSON files from the given folder.
    """
    json_files = []
    for filename in sorted(os.listdir(question_folder)):
        if filename.endswith(".json"):
            with open(os.path.join(question_folder, filename), 'r') as f:
                json_files.append(json.load(f))
    return json_files

def load_image(image_folder, image_id):
    """
    Load an image, trying both .png and .jpg extensions.
    """
    possible_extensions = ['.png', '.jpg', '.jpeg']
    for ext in possible_extensions:
        image_path = os.path.join(image_folder, f"{image_id}{ext}")
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                return image
            except Exception as e:
                print(f"Error loading image at {image_path}: {e}")
                return None
    print(f"Image not found for ID: {image_id}")
    return None


def load_random_test_image(image_folder, image_id):
    """
    Load a random image from a folder in 'test' mode.

    Args:
        image_folder: The folder where the image_id folder is stored.
        image_id: The ID of the folder containing multiple images.

    Returns:
        image: The randomly selected image (or None if not found or error occurs).
    """
    # In 'test' mode, image_id is a folder containing multiple images
    image_dir = os.path.join(image_folder, image_id)

    if not os.path.isdir(image_dir):
        print(f"Image folder not found for ID: {image_id}")
        return None

    # List the images inside the folder
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Filter to match the expected naming format: {image_id}_poseX_gen1.png
    image_files = [f for f in image_files if f.startswith(image_id) and 'pose' in f]

    if not image_files:
        print(f"No valid images found in folder: {image_dir}")
        return None

    # Randomly select one image from the folder
    selected_image = random.choice(image_files)
    image_path = os.path.join(image_dir, selected_image)

    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Randomly selected image: {selected_image}")
        return image
    except Exception as e:
        print(f"Error loading image at {image_path}: {e}")
        return None

def evaluate_classification(parquet_file, few_shot_parquet_file, processor, tokenizer, model, args, id_list_file=None, mode="default", forget_parquet_file=None):
    """
    Evaluate classification task in zero-shot mode.

    Args:
        parquet_file: Path to the main Parquet file for evaluation.
        few_shot_parquet_file: Unused. Kept for compatibility with existing callers.
        processor: The processor for handling image and text inputs.
        tokenizer: The tokenizer for decoding model outputs.
        model: The model to use for classification.
        args: Arguments object containing model ID and other configurations.
        id_list_file: (Optional) Path to the JSON file containing the list of IDs. Default is None.
        mode: Evaluation split/mode name. Default is 'default'.
        forget_parquet_file: (Optional) Path to the forget Parquet file to filter IDs for test mode.

    Returns:
        dict: A dictionary with accuracy scores.
    """
    print("################################## Classification Task Starts ##############################################")
    print(f"############################## Evaluating {mode} Mode #########################################" )

    # Load the ID list from the JSON file if provided
    if id_list_file:
        with open(id_list_file, 'r') as f:
            id_list = json.load(f)
    elif mode == "test" and forget_parquet_file:
        # Load IDs from the forget Parquet file for filtering in test mode
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
        # If no id_list_file is provided, load all IDs from the main Parquet file
        df = pd.read_parquet(parquet_file)
        id_list = df['ID'].unique().tolist()

    print(f"Loaded {len(id_list)} IDs from {id_list_file if id_list_file else 'parquet_file'}")

    total_image_textual_correct = 0
    total_image_textual_questions = 0
    total_pure_text_correct = 0
    total_pure_text_questions = 0

    print("Zero-shot classification: no in-context examples will be loaded.")

    # Load evaluation samples
    if mode == "test":
        if os.path.isdir(parquet_file):  # Check if it's a directory containing multiple Parquet files
            df = load_and_combine_parquet_files(parquet_file)
        else:
            df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]

    # Process each evaluation sample
    for _, row in eval_samples.iterrows():
        classification_questions = row["Classification_Task"]

        # Randomly select one image if in test mode
        if mode == "test" and "images" in row:
            image_data = random.choice(row["images"])["bytes"]
        else:
            image_data = row["image"]["bytes"]

        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Iterate through each image-textual question
        print("########################## Processing Image-Textual Questions ########################## ")
        for idx, question_data in enumerate(classification_questions.get("Image_Textual_Questions", [])):
            question = question_data["Question"]
            options = question_data["Options"]
            correct_answer = question_data["Correct_Answer"]
            question_with_options = formulate_prompt_with_options(question, options)

            prompt = (f"USER: <image>\n{question_with_options}\n"
                      f"Just give ONE letter representing the answer directly.\nASSISTANT:")

            # Model specific logic for generating answers
            if args.model_id.startswith("HuggingFaceM4"):
                inputs = processor(images=[image], text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = processor.decode(outputs[0][2:], skip_special_tokens=True)
            elif args.model_id.startswith("llava"):
                inputs = processor(images=[image], text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = processor.decode(outputs[0][2:], skip_special_tokens=True)

            assistant_response = generated_text.split("ASSISTANT:")[1].strip() if "ASSISTANT:" in generated_text else generated_text.strip()
            predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() in options else None
            if predicted_answer == correct_answer:
                total_image_textual_correct += 1
            total_image_textual_questions += 1
            print("Prompt: ", prompt)
            print("Model Answer: ", predicted_answer)
            print("Correct Answer: ", correct_answer)
            print("The model answer is: ", predicted_answer == correct_answer)
            print("\n")

        # Process Pure_Text_Questions
        print("########################## Processing Pure-textual Questions ########################## ")
        for idx, question_data in enumerate(classification_questions.get("Pure_Text_Questions", [])):
            question = question_data["Question"]
            options = question_data["Options"]
            correct_answer = question_data["Correct_Answer"]
            question_with_options = formulate_prompt_with_options(question, options)

            prompt = (
                f"USER:\n{question_with_options}\n"
                f"Just give ONE letter representing the answer directly.\nASSISTANT:"
            )


            # Model specific logic
            if args.model_id.startswith("HuggingFaceM4"):
                inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
                generated_text = tokenizer.decode(outputs[0][2:], skip_special_tokens=True)
            elif args.model_id.startswith("llava"):
                inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = tokenizer.decode(outputs[0][2:], skip_special_tokens=True)

            assistant_response = generated_text.split("ASSISTANT:")[1].strip() if "ASSISTANT:" in generated_text else generated_text.strip()
            predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() in options else None
            if predicted_answer == correct_answer:
                total_pure_text_correct += 1
            total_pure_text_questions += 1

            print("Prompt: ", prompt)
            print("Model Answer: ", predicted_answer)
            print("Correct Answer: ", correct_answer)
            print("The model answer is: ", predicted_answer == correct_answer)
            print("\n")

    # Calculate accuracy
    image_textual_accuracy = (total_image_textual_correct / total_image_textual_questions) * 100 if total_image_textual_questions > 0 else 0
    pure_text_accuracy = (total_pure_text_correct / total_pure_text_questions) * 100 if total_pure_text_questions > 0 else 0

    result = format_accuracy_result(
        total_image_textual_correct,
        total_image_textual_questions,
        total_pure_text_correct,
        total_pure_text_questions,
    )
    print_accuracy_result("Classification", result)
    return result


# def evaluate_fill_in_the_blank(json_files, image_folder, processor, tokenizer, model, args, id_list_file=None, mode="default"):
def evaluate_fill_in_the_blank(parquet_file, few_shot_parquet_file, processor, tokenizer, model, args, id_list_file=None, mode="default", forget_parquet_file=None):
    """
    Evaluate fill-in-the-blank task in zero-shot mode.

    Args:
        parquet_file: Path to the main Parquet file for evaluation.
        few_shot_parquet_file: Unused. Kept for compatibility with existing callers.
        processor: The processor for handling image and text inputs.
        tokenizer: The tokenizer for decoding model outputs.
        model: The model to use for classification.
        args: Arguments object containing model ID and other configurations.
        id_list_file: (Optional) Path to the JSON file containing the list of IDs. Default is None.
        mode: Evaluation split/mode name. Default is 'default'.
        forget_parquet_file: (Optional) Path to the forget Parquet file to filter IDs for test mode.

    Returns:
        dict: A dictionary with accuracy scores.
    """
    print(
        "################################## Fill-in-the-blank Task Starts ##############################################")

    print(f"Evaluating {mode} Mode")
    # Load the ID list from the JSON file if provided
    if id_list_file:
        with open(id_list_file, 'r') as f:
            id_list = json.load(f)
    elif mode == "test" and forget_parquet_file:
        # Load IDs from the forget Parquet file for filtering in test mode
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
        # If no id_list_file is provided, load all IDs from the Parquet file
        df = pd.read_parquet(parquet_file)
        id_list = df['ID'].unique().tolist()

    print(f"Loaded {len(id_list)} IDs from {id_list_file if id_list_file else 'parquet_file'}")

    total_image_textual_correct = 0
    total_image_textual_questions = 0
    total_pure_text_correct = 0
    total_pure_text_questions = 0

    print("Zero-shot fill-in-the-blank: no in-context examples will be loaded.")

    # Load evaluation samples
    # Load the test set with multiple Parquet files if mode is "test"
    if mode == "test":
        if os.path.isdir(parquet_file):  # Check if it's a directory containing multiple Parquet files
            df = load_and_combine_parquet_files(parquet_file)
        else:
            df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]

    # Process each evaluation sample
    for _, row in eval_samples.iterrows():
        fill_in_the_blank_questions = row["Mask_Task"]

        # Randomly select one image if in test mode
        if mode == "test" and "images" in row:
            image_data = random.choice(row["images"])["bytes"]
        else:
            image_data = row["image"]["bytes"]

        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Iterate through each question in Mask_Task.
        for idx, question_entry in enumerate(fill_in_the_blank_questions):
            question = question_entry["Question"]
            ground_truth = question_entry["Ground_Truth"]
            question_type = question_entry["Type"]
            question = question.replace("__", "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."

            prompt = (f"USER: "
                      f"<image>\n{question}\nASSISTANT:" if question_type == "Image_Textual" else
                      f"USER:\n{question}\nASSISTANT:")

            # Model specific logic
            if args.model_id.startswith("HuggingFaceM4"):
                inputs = processor(images=[image] if question_type == "Image_Textual" else None,
                                   text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = processor.decode(outputs[0][2:], skip_special_tokens=True)

            elif args.model_id.startswith("llava"):
                inputs = processor(images=[image] if question_type == "Image_Textual" else None,
                                   text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = processor.decode(outputs[0][2:], skip_special_tokens=True)

            # Process the answer
            if "ASSISTANT:" in generated_text:
                assistant_response = generated_text.split("ASSISTANT:")[1].strip()
            elif "Answer:" in generated_text:
                assistant_response = generated_text.split("Answer:")[1].strip()
            else:
                assistant_response = generated_text.strip()

            print("Prompt: ", prompt)
            print("Model Answer: ", assistant_response)
            print("Correct Answer: ", ground_truth)
            print("The model answer is: ", ground_truth.lower() in assistant_response.lower())
            print("\n")
            # Evaluate if the generated answer contains the correct ground truth
            if question_type == "Image_Textual":
                if ground_truth.lower() in assistant_response.lower():
                    total_image_textual_correct += 1
                total_image_textual_questions += 1
            elif question_type == "Pure_Text":
                if ground_truth.lower() in assistant_response.lower():
                    total_pure_text_correct += 1
                total_pure_text_questions += 1

    # Calculate accuracy
    result = format_accuracy_result(
        total_image_textual_correct,
        total_image_textual_questions,
        total_pure_text_correct,
        total_pure_text_questions,
    )
    print_accuracy_result("Fill-in-the-blank", result)
    return result

def evaluate_generation(parquet_file, processor, tokenizer, model, args, mode="default", forget_parquet_file=None):
    """
    Evaluate the generation tasks using the ROUGE and BLEU scores.

    Args:
        parquet_file: Path to the main Parquet file for evaluation.
        processor: The processor for handling text and images (e.g., from Hugging Face).
        tokenizer: The tokenizer for decoding model outputs.
        model: The model for answering the generation questions.
        args: Arguments object containing model ID and other configurations.
        file_name: Name of the file to save the evaluation results.
        mode: Mode to control which evaluation setup to use. Default is 'default'.
        forget_parquet_file: (Optional) Path to the forget Parquet file to filter IDs for test mode.

    Returns:
        dict: A dictionary containing average ROUGE and BLEU scores for Image_Textual and Pure_Text questions.
    """
    print("################################## Generation Task Starts ##############################################")

    # Initialize ROUGE scorer
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Initialize variables to store scores and question counts for both question types
    total_rouge1_img = total_rouge2_img = total_rougeL_img = total_bleu_img = total_image_textual_questions = 0
    total_rouge1_text = total_rouge2_text = total_rougeL_text = total_bleu_text = total_pure_text_questions = 0

    # Initialize list to store the results
    results = {
        "Generation_Questions": []
    }

    # Load the ID list from the forget Parquet file for filtering if mode is "test"
    if mode == "test" and forget_parquet_file:
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
        # Load all IDs from the Parquet file if no filtering is needed
        df = pd.read_parquet(parquet_file)
        id_list = df['ID'].unique().tolist()

    # Load evaluation samples
    if mode == "test":
        if os.path.isdir(parquet_file):  # Check if it's a directory containing multiple Parquet files
            df = load_and_combine_parquet_files(parquet_file)
        else:
            df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]

    # Loop through each person's data in the evaluation samples
    for _, row in tqdm(eval_samples.iterrows(), total=len(eval_samples)):
        image_id = row["ID"]
        generation_questions = row["Generation_Task"]

        # Randomly select one image if in test mode and multiple images are available
        if mode == "test" and "images" in row:
            image_data = random.choice(row["images"])["bytes"]
        else:
            image_data = row["image"]["bytes"]

        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Process each generation question
        for question_data in generation_questions:
            question_type = question_data["Type"]
            question = question_data["Question"]
            ground_truth = question_data["Ground_Truth"]

            if question_type == "Image_Textual":
                prompt = f"USER: <image>\n{question}\nAnswer the question based on your trained knowledge in one sentence accurately in ENGLISH.\nASSISTANT: "

                if args.model_id.startswith("HuggingFaceM4"):
                    inputs = processor(images=[image], text=prompt, return_tensors="pt").to("cuda")
                elif args.model_id.startswith("llava"):
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
                elif args.model_id.startswith("meta-llama"):
                    llama_prompt = f"<|image|><|begin_of_text|>### Question:{question}\n### Answer:"
                    inputs = processor(images=image, text=llama_prompt, return_tensors="pt")
                else:
                    raise ValueError("Model ID not supported")

                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_answer = processor.decode(outputs[0][2:], skip_special_tokens=True)

            else:  # Pure_Text case
                if args.model_id.startswith("meta-llama"):
                    llama_prompt = f"<|begin_of_text|>### Question: {question}\n### Answer:"
                    inputs = processor(text=llama_prompt, return_tensors="pt").to("cuda")
                else:
                    prompt = f"USER: {question}\nAnswer the question based on your trained knowledge in one sentence in ENGLISH.\nASSISTANT:"
                    inputs = tokenizer(prompt, return_tensors='pt').to("cuda")

                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_answer = tokenizer.decode(outputs[0][2:], skip_special_tokens=True)

            # Process the generated answer
            if "ASSISTANT:" in generated_answer:
                predicted_answer = generated_answer.split("ASSISTANT:")[1].strip()
            elif "Answer:" in generated_answer:
                predicted_answer = generated_answer.split("Answer:")[1].strip()
            else:
                predicted_answer = generated_answer.strip()

            # Print debug information
            print("###### Generation Question: ######", question)
            print("###### Generation Prompt: ######", prompt)
            print("###### Generation ASSISTANT: ######", predicted_answer)
            print("###### Generation Ground Truth: ######", ground_truth)

            # Save results for this question
            results["Generation_Questions"].append({
                "image_id": image_id,
                "question type": question_type,
                "question": question,
                "generated_answer": predicted_answer,
                "ground_truth": ground_truth
            })

            # Calculate ROUGE and BLEU scores
            bleu_score = compute_bleu(ground_truth, predicted_answer)
            rouge_scores = rouge_scorer_obj.score(ground_truth, predicted_answer)

            if question_type == "Image_Textual":
                # Accumulate scores for Image_Textual questions
                total_bleu_img += bleu_score
                total_rouge1_img += rouge_scores['rouge1'].fmeasure
                total_rouge2_img += rouge_scores['rouge2'].fmeasure
                total_rougeL_img += rouge_scores['rougeL'].fmeasure
                total_image_textual_questions += 1
            else:
                # Accumulate scores for Pure_Text questions
                total_bleu_text += bleu_score
                total_rouge1_text += rouge_scores['rouge1'].fmeasure
                total_rouge2_text += rouge_scores['rouge2'].fmeasure
                total_rougeL_text += rouge_scores['rougeL'].fmeasure
                total_pure_text_questions += 1

    # Save the results to a JSON file
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    with open(f'{args.output_folder}/{mode}_generation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Calculate and print average ROUGE and BLEU scores
    avg_scores = {}
    if total_image_textual_questions > 0:
        avg_scores.update({
            "Average ROUGE-1 (Image_Textual)": total_rouge1_img / total_image_textual_questions,
            "Average ROUGE-2 (Image_Textual)": total_rouge2_img / total_image_textual_questions,
            "Average ROUGE-L (Image_Textual)": total_rougeL_img / total_image_textual_questions,
            "Average BLEU (Image_Textual)": total_bleu_img / total_image_textual_questions
        })

    if total_pure_text_questions > 0:
        avg_scores.update({
            "Average ROUGE-1 (Pure_Text)": total_rouge1_text / total_pure_text_questions,
            "Average ROUGE-2 (Pure_Text)": total_rouge2_text / total_pure_text_questions,
            "Average ROUGE-L (Pure_Text)": total_rougeL_text / total_pure_text_questions,
            "Average BLEU (Pure_Text)": total_bleu_text / total_pure_text_questions
        })

    total_generation_questions = total_image_textual_questions + total_pure_text_questions
    if total_generation_questions > 0:
        avg_scores["Average ROUGE-L (Overall)"] = (
            (total_rougeL_img + total_rougeL_text) / total_generation_questions
        )

    for metric, score in avg_scores.items():
        print(f"{metric}: {score}")

    return avg_scores




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"Random seed: {seed}", flush=True)


def normalize_eval_list(eval_list):
    aliases = {
        "forget": "forget",
        "test": "test",
        "retain": "retain_shared",
        "retain_shared": "retain_shared",
        "shared": "retain_shared",
        "real": "retain_celebrity",
        "celebrity": "retain_celebrity",
        "retain_celebrity": "retain_celebrity",
        "retain_real": "retain_celebrity",
    }
    normalized = []
    for item in eval_list:
        key = str(item).strip().lower()
        if key == "all":
            return ["forget", "test", "retain_shared", "retain_celebrity"]
        if key not in aliases:
            raise ValueError(
                f"Unknown eval split {item!r}. Valid choices: "
                "forget, test, retain, retain_shared, real, retain_celebrity, all."
            )
        value = aliases[key]
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("--eval_list must contain at least one split.")
    return normalized


def load_merge_and_optionally_save_adapter(model, processor, tokenizer, args):
    adapter_path = getattr(args, "adapter_path", None)
    if not adapter_path:
        print("Adapter: none. Evaluating cache_path model.")
        return model

    if PeftModel is None:
        raise ImportError(
            "--adapter_path was provided, but peft is not installed. "
            "Install with: pip install peft"
        )
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    print(f"Loading PEFT adapter from: {adapter_path}", flush=True)
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=False,
    )
    model.eval()
    print(f"PEFT adapter loaded: {adapter_path}", flush=True)
    print(f"Model class after adapter load: {type(model).__name__}", flush=True)
    if hasattr(model, "active_adapters"):
        print(f"Active adapters: {model.active_adapters}", flush=True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params during eval/merge: {trainable_params} / {total_params}", flush=True)

    if getattr(args, "no_merge_adapter", False):
        print(
            "Warning: adapter is attached but not merged. "
            "For LLaVA generation, merged mode is recommended.",
            flush=True,
        )
    else:
        print("Merging PEFT adapter into base model...", flush=True)
        model = model.merge_and_unload()
        model.eval()
        print(f"Model class after merge: {type(model).__name__}", flush=True)

    save_path = getattr(args, "save_merged_model_path", None)
    if save_path:
        if getattr(args, "no_merge_adapter", False):
            raise ValueError("--save_merged_model_path requires merged weights. Remove --no_merge_adapter.")
        os.makedirs(save_path, exist_ok=True)
        print(f"Saving merged full model to: {save_path}", flush=True)
        model.save_pretrained(save_path, safe_serialization=True)
        if processor is not None:
            processor.save_pretrained(save_path)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_path)
        print(f"Merged model saved to: {save_path}", flush=True)

    return model


def main(args):
    set_seed(args.seed)
    eval_splits = normalize_eval_list(args.eval_list)
    print("Eval splits:", eval_splits)
    # Construct folder paths for "forget" and "retain"
    forget_folder = os.path.join(args.data_split_folder, f"forget_{args.forget_ratio}")
    retain_folder = os.path.join(args.data_split_folder, f"retain_{100 - args.forget_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)
    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")
    # real_paraquet_file = os.path.join(args.celebrity_data, f"train-00000-of-00001.parquet")

    processor_source = args.processor_id or args.model_id
    local_processor = os.path.isdir(processor_source)
    processor = AutoProcessor.from_pretrained(
        processor_source,
        local_files_only=local_processor,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        processor_source,
        local_files_only=local_processor,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("The existing MLLMU evaluation pipeline requires a CUDA GPU.")
    torch.cuda.empty_cache()
    model = LlavaForConditionalGeneration.from_pretrained(
        args.cache_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = load_merge_and_optionally_save_adapter(model, processor, tokenizer, args)
    model.eval()
    # Loading adapters with different ranks can consume different RNG streams.
    # Reset after loading so evaluation sampling starts from the same state.
    set_seed(args.seed)

    results_data = {}

    if "forget" in eval_splits:
        print("### Evaluating Forget Set ###")
        forget_fill_in_the_blank_result = evaluate_fill_in_the_blank(
            parquet_file=forget_parquet_file,
            few_shot_parquet_file=args.few_shot_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=args,
            mode="forget",
        )

        forget_classification_result = evaluate_classification(
            parquet_file=forget_parquet_file,
            few_shot_parquet_file=args.few_shot_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=args,
            mode="forget",
        )

        forget_generation_result = evaluate_generation(
            parquet_file=forget_parquet_file,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=args,
            mode="forget",
        )
        print("Forget Set Results:")
        print(forget_classification_result)
        print(forget_generation_result)
        print(forget_fill_in_the_blank_result)
        results_data["Forget Set Results"] = {
            "fill_in_the_blank": forget_fill_in_the_blank_result,
            "classification": forget_classification_result,
            "generation": forget_generation_result,
        }

    if "test" in eval_splits:
        print("### Evaluating Test Set ###")
        test_classification_result = evaluate_classification(
            parquet_file=args.test_data,
            few_shot_parquet_file=args.few_shot_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=args,
            mode="test",
            forget_parquet_file=forget_parquet_file,
        )

        test_fill_in_the_blank_result = evaluate_fill_in_the_blank(
            parquet_file=args.test_data,
            few_shot_parquet_file=args.few_shot_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=args,
            mode="test",
            forget_parquet_file=forget_parquet_file,
        )

        test_generation_result = evaluate_generation(
            parquet_file=args.test_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=args,
            mode="test",
            forget_parquet_file=forget_parquet_file,
        )
        print("Test Set Results:")
        print(test_fill_in_the_blank_result)
        print(test_classification_result)
        print(test_generation_result)
        results_data["Test Set Results"] = {
            "fill_in_the_blank": test_fill_in_the_blank_result,
            "classification": test_classification_result,
            "generation": test_generation_result,
        }

    if "retain_shared" in eval_splits:
        print("### Evaluating Retain Shared Set ###")
        retain_fill_in_the_blank_result = evaluate_fill_in_the_blank(
            parquet_file=retain_parquet_file,
            few_shot_parquet_file=args.few_shot_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=args,
            mode="retain_shared",
        )

        retain_classification_result = evaluate_classification(
            parquet_file=retain_parquet_file,
            few_shot_parquet_file=args.few_shot_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=args,
            mode="retain_shared",
        )

        retain_generation_result = evaluate_generation(
            parquet_file=retain_parquet_file,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=args,
            mode="retain_shared",
        )
        print("Retain Set (shared dataset) Results:")
        print(retain_fill_in_the_blank_result)
        print(retain_classification_result)
        print(retain_generation_result)
        results_data["Retain Set (shared dataset) Results"] = {
            "fill_in_the_blank": retain_fill_in_the_blank_result,
            "classification": retain_classification_result,
            "generation": retain_generation_result,
        }

    if "retain_celebrity" in eval_splits:
        print("### Evaluating Real Celebrity Set ###")
        real_fill_in_the_blank_result = evaluate_fill_in_the_blank(
            parquet_file=args.celebrity_data,
            few_shot_parquet_file=args.few_shot_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=args,
            mode="retain_celebrity",
        )

        real_classification_result = evaluate_classification(
            parquet_file=args.celebrity_data,
            few_shot_parquet_file=args.few_shot_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=args,
            mode="retain_celebrity",
        )

        real_generation_result = evaluate_generation(
            parquet_file=args.celebrity_data,
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            args=args,
            mode="retain_celebrity",
        )
        print("Retain Set (real person) Results:")
        print(real_fill_in_the_blank_result)
        print(real_classification_result)
        print(real_generation_result)
        results_data["Retain Set (real person) Results"] = {
            "fill_in_the_blank": real_fill_in_the_blank_result,
            "classification": real_classification_result,
            "generation": real_generation_result,
        }

    # Output results
    output_file = f'{args.output_folder}/{args.output_file}_final_evaluation_results.json'
    accuracy_summary_file = f'{args.output_folder}/{args.output_file}_accuracy_summary.json'
    focus_metrics_file = f'{args.output_folder}/{args.output_file}_focus_metrics.json'

    accuracy_summary = build_accuracy_summary(results_data)
    focus_metrics = build_focus_metrics(results_data)

    # Write the results to a local JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)

    with open(accuracy_summary_file, 'w', encoding='utf-8') as f:
        json.dump(accuracy_summary, f, ensure_ascii=False, indent=4)

    with open(focus_metrics_file, 'w', encoding='utf-8') as f:
        json.dump(focus_metrics, f, ensure_ascii=False, indent=4)

    print_accuracy_summary(accuracy_summary)
    print(f"Results saved to {output_file}")
    print(f"Accuracy summary saved to {accuracy_summary_file}")
    print(f"Focus metrics saved to {focus_metrics_file}")


if __name__ == "__main__":
    from run_eval import main as run
    run()


