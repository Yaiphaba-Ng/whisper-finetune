import os
import sys
import argparse
import torch  # Ensure torch is imported at the top
import yaml  # Add at the top with other imports
import random  # Added for evaluation sampling
from datetime import datetime
import json
import gradio as gr

# Load Hugging Face token from .env file and set as environment variable
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN", None)
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    print("HF Token loaded from .env and set as environment variable.")
    from huggingface_hub import login
    login(token=hf_token)
else:
    print("HF_TOKEN not found in .env file.")

# Early parse for GPU
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", dest="gpu_device", type=str, default=None, help="GPU device id to use (default: all available)")
args, unknown = parser.parse_known_args()

gpu_device = args.gpu_device

# If not set via CLI, try to get from config file
if gpu_device is None and args.config is not None and os.path.exists(args.config):
    with open(args.config, 'r') as f:
        config_yaml = yaml.safe_load(f)
    gpu_device = config_yaml.get('gpu_device', None)

if gpu_device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    print(f"Set CUDA_VISIBLE_DEVICES to {gpu_device} (will appear as device 0 in torch)")
    # Always use device 0 in torch if CUDA_VISIBLE_DEVICES is set
    torch.cuda.set_device(0)

from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
    WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
)
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # Ensure attention_mask is set for input_features
        if "attention_mask" not in batch:
            batch["attention_mask"] = torch.ones(batch["input_features"].shape[:-1], dtype=torch.long)
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

def finetune_whisper(
    training_args_dict=None,
    **kwargs
):
    # Unpack all config values from training_args_dict and kwargs
    config = {**kwargs, **(training_args_dict or {})}
    # Use a single 'lang' parameter for both model and dataset language
    lang = config.get('lang')
    model_name = config.get('model_name')
    model_cache = config.get('model_cache')
    whisper_pretrained = config.get('whisper_pretrained') or (f"openai/{model_name}" if model_name else None)
    checkpoint_name = config.get('checkpoint_name') or (f"{model_name}-{lang}" if model_name and lang else None)
    checkpoint_dir = config.get('checkpoint_dir') or (f"./checkpoints/{checkpoint_name}" if checkpoint_name else None)
    max_steps = config.get('max_steps')
    # Update config dict with derived values so printout is correct
    config['whisper_pretrained'] = whisper_pretrained
    config['checkpoint_name'] = checkpoint_name
    config['checkpoint_dir'] = checkpoint_dir
    # General config keys
    general_keys = [
        'lang','dataset_name','dataset_cache','model_name','model_cache',
        'whisper_pretrained','checkpoint_name','checkpoint_dir'
    ]
    print("\n===== Whisper Fine-tuning: General Configuration =====")
    for k in general_keys:
        print(f"{k}: {config.get(k)}")
    print("\n===== Whisper Fine-tuning: Training Arguments =====")
    for k, v in config.items():
        if k not in general_keys:
            print(f"{k}: {v}")
    print("====================================================\n")

    # Assign all variables from config
    dataset_name = config.get('dataset_name')
    dataset_cache = config.get('dataset_cache')
    model_name = config.get('model_name')
    model_cache = config.get('model_cache')
    whisper_pretrained = config.get('whisper_pretrained') or f"openai/{model_name}"
    checkpoint_name = config.get('checkpoint_name') or f"{model_name}-{lang}"
    checkpoint_dir = config.get('checkpoint_dir') or f"./checkpoints/{checkpoint_name}"
    max_steps = config.get('max_steps')

    # Load dataset
    common_voice = DatasetDict()
    common_voice["train"] = load_dataset(
        dataset_name,
        lang,
        split="train+validation",
        cache_dir=dataset_cache,
        trust_remote_code=True
    )
    common_voice["test"] = load_dataset(
        dataset_name,
        lang,
        split="test",
        cache_dir=dataset_cache,
        trust_remote_code=True
    )
    # Remove extra columns
    common_voice = common_voice.remove_columns([
        "accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"
    ])
    # Feature extractor, tokenizer, processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_pretrained, cache_dir=model_cache)
    tokenizer = WhisperTokenizer.from_pretrained(whisper_pretrained, language=lang, task="transcribe", cache_dir=model_cache)
    processor = WhisperProcessor.from_pretrained(whisper_pretrained, language=lang, task="transcribe", cache_dir=model_cache)
    # Resample audio
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    # Prepare dataset
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch
    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)
    # Model
    model = WhisperForConditionalGeneration.from_pretrained(whisper_pretrained, cache_dir=model_cache)
    # Move generation parameters from model.config to model.generation_config to silence warning
    gen_keys = [
        'max_length', 'min_length', 'do_sample', 'early_stopping', 'num_beams', 'temperature',
        'top_k', 'top_p', 'repetition_penalty', 'length_penalty', 'no_repeat_ngram_size',
        'encoder_no_repeat_ngram_size', 'bad_words_ids', 'num_return_sequences',
        'forced_bos_token_id', 'forced_eos_token_id', 'remove_invalid_values',
        'exponential_decay_length_penalty', 'suppress_tokens', 'begin_suppress_tokens'
    ]
    for k in gen_keys:
        if hasattr(model.config, k):
            setattr(model.generation_config, k, getattr(model.config, k))
            # Optionally, remove from model.config to avoid confusion
            # delattr(model.config, k)
    model.generation_config.language = lang
    model.generation_config.task = "transcribe"
    # model.generation_config.forced_decoder_ids = None  # (commented out as per your previous change)
    model.config.use_cache = False  # For gradient checkpointing compatibility
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    # Metrics
    metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
    # Set/override with function arguments
    training_args_dict.setdefault('output_dir', checkpoint_dir)
    training_args_dict.setdefault('per_device_train_batch_size', 16)
    training_args_dict.setdefault('gradient_accumulation_steps', 1)
    training_args_dict.setdefault('learning_rate', 1e-5)
    training_args_dict.setdefault('warmup_steps', 500)
    training_args_dict.setdefault('max_steps', max_steps)
    training_args_dict.setdefault('gradient_checkpointing', True)
    training_args_dict.setdefault('fp16', True)
    training_args_dict.setdefault('eval_strategy', 'steps')
    training_args_dict.setdefault('per_device_eval_batch_size', 8)
    training_args_dict.setdefault('predict_with_generate', True)
    training_args_dict.setdefault('generation_max_length', 225)
    training_args_dict.setdefault('save_steps', 1000)
    training_args_dict.setdefault('eval_steps', 1000)
    training_args_dict.setdefault('logging_steps', 25)
    training_args_dict.setdefault('report_to', ["tensorboard"])
    training_args_dict.setdefault('load_best_model_at_end', True)
    training_args_dict.setdefault('metric_for_best_model', "wer")
    training_args_dict.setdefault('greater_is_better', False)
    training_args_dict.setdefault('push_to_hub', True)
    # Type-cast known numeric training arguments to correct types
    numeric_casts = {
        'learning_rate': float,
        'warmup_steps': int,
        'max_steps': int,
        'per_device_train_batch_size': int,
        'gradient_accumulation_steps': int,
        'per_device_eval_batch_size': int,
        'generation_max_length': int,
        'save_steps': int,
        'eval_steps': int,
        'logging_steps': int,
    }
    for k, cast in numeric_casts.items():
        if k in training_args_dict and training_args_dict[k] is not None:
            try:
                training_args_dict[k] = cast(training_args_dict[k])
            except Exception:
                pass  # Leave as is if conversion fails
    # Remove custom arguments not supported by Seq2SeqTrainingArguments
    training_args_dict.pop('delay_between_batches_sec', None)
    training_args = Seq2SeqTrainingArguments(**training_args_dict)
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )
    # Custom: Get delay between batches from config (in seconds)
    delay_between_batches_sec = float(config.get('delay_between_batches_sec', 0))
    import time
    # Custom: Patch trainer to add delay after each training step
    if delay_between_batches_sec > 0:
        orig_training_step = trainer.training_step
        def delayed_training_step(*args, **kwargs):
            result = orig_training_step(*args, **kwargs)
            time.sleep(delay_between_batches_sec)
            return result
        trainer.training_step = delayed_training_step
    # Train the model, resume if checkpoint exists, else start fresh
    import os
    checkpoint_dir_to_check = training_args.output_dir
    checkpoint_found = False
    if os.path.isdir(checkpoint_dir_to_check):
        # Look for a subdirectory like checkpoint-xxxx
        for entry in os.listdir(checkpoint_dir_to_check):
            if entry.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_dir_to_check, entry)):
                checkpoint_found = True
                break
    if checkpoint_found:
        print(f"Resuming training from last checkpoint in {checkpoint_dir_to_check}...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print(f"No valid checkpoint found in {checkpoint_dir_to_check}. Starting training from scratch...")
        trainer.train()
    # Evaluate on test set and print final WER
    eval_results = trainer.evaluate()
    print("Training complete. Best model saved at:", training_args.output_dir)
    print(f"Final WER on test set: {eval_results.get('eval_wer', 'N/A')}")

    # Push to hub logic (from notebook)
    push_kwargs = {
        "dataset_tags": f"{dataset_name}",
        "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
        "dataset_args": f"config: {lang}, split: test",
        "language": f"{lang}",  # Use ISO 639-1 code for Hugging Face Hub
        "model_name": f"{checkpoint_name} - Fine-tuned",  # a 'pretty' name for our model
        "finetuned_from": whisper_pretrained,
        "tasks": "automatic-speech-recognition",
    }
    print("Pushing model and processor to the Hugging Face Hub...")
    trainer.push_to_hub(**push_kwargs)
    processor.save_pretrained(training_args.output_dir)
    feature_extractor.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print("Push to hub complete.")

def finetune_custom_dataset(
    custom_dataset_path,
    lang,
    model_name,
    model_cache,
    whisper_pretrained,
    checkpoint_name,
    checkpoint_dir,
    max_steps,
    training_args_dict
):
    """
    Fine-tune Whisper on a custom dataset (TSV with 'path' and 'sentence' columns).
    """
    import pandas as pd
    from datasets import Dataset, DatasetDict, Audio
    from transformers import (
        WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
        WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
    )
    import evaluate
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    import torch
    import numpy as np
    import soundfile as sf
    # Load TSV
    df = pd.read_csv(custom_dataset_path, sep='\t')
    # Remove rows with missing values
    df = df.dropna(subset=['path', 'sentence'])
    # Shuffle and split (90% train, 10% test)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_train = int(0.9 * len(df))
    df_train = df.iloc[:n_train]
    df_test = df.iloc[n_train:]
    # Convert to HuggingFace Datasets
    ds_train = Dataset.from_pandas(df_train, preserve_index=False)
    ds_test = Dataset.from_pandas(df_test, preserve_index=False)
    dataset = DatasetDict({"train": ds_train, "test": ds_test})
    # Add audio column (load audio from file)
    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))
    # Load processor, tokenizer, etc.
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_pretrained, cache_dir=model_cache)
    tokenizer = WhisperTokenizer.from_pretrained(whisper_pretrained, language=lang, task="transcribe", cache_dir=model_cache)
    processor = WhisperProcessor.from_pretrained(whisper_pretrained, language=lang, task="transcribe", cache_dir=model_cache)
    # Prepare dataset
    def prepare_dataset(batch):
        audio = batch["path"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch
    dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names, num_proc=1)
    # Model
    model = WhisperForConditionalGeneration.from_pretrained(whisper_pretrained, cache_dir=model_cache)
    model.generation_config.language = lang
    model.generation_config.task = "transcribe"
    model.config.use_cache = False
    # Data collator
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            if "attention_mask" not in batch:
                batch["attention_mask"] = torch.ones(batch["input_features"].shape[:-1], dtype=torch.long)
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    # Metrics
    metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}
    # Set/override with function arguments
    training_args_dict.setdefault('output_dir', checkpoint_dir)
    training_args_dict.setdefault('per_device_train_batch_size', 16)
    training_args_dict.setdefault('gradient_accumulation_steps', 1)
    training_args_dict.setdefault('learning_rate', 1e-5)
    training_args_dict.setdefault('warmup_steps', 500)
    training_args_dict.setdefault('max_steps', max_steps)
    training_args_dict.setdefault('gradient_checkpointing', True)
    training_args_dict.setdefault('fp16', True)
    training_args_dict.setdefault('eval_strategy', 'steps')
    training_args_dict.setdefault('per_device_eval_batch_size', 8)
    training_args_dict.setdefault('predict_with_generate', True)
    training_args_dict.setdefault('generation_max_length', 225)
    training_args_dict.setdefault('save_steps', 1000)
    training_args_dict.setdefault('eval_steps', 1000)
    training_args_dict.setdefault('logging_steps', 25)
    training_args_dict.setdefault('report_to', ["tensorboard"])
    training_args_dict.setdefault('load_best_model_at_end', True)
    training_args_dict.setdefault('metric_for_best_model', "wer")
    training_args_dict.setdefault('greater_is_better', False)
    training_args_dict.setdefault('push_to_hub', True)
    training_args_dict.setdefault('save_total_limit', 3)
    # Type-cast known numeric training arguments to correct types
    numeric_casts = {
        'learning_rate': float,
        'warmup_steps': int,
        'max_steps': int,
        'per_device_train_batch_size': int,
        'gradient_accumulation_steps': int,
        'per_device_eval_batch_size': int,
        'generation_max_length': int,
        'save_steps': int,
        'save_total_limit': int,
        'eval_steps': int,
        'logging_steps': int,
    }
    for k, cast in numeric_casts.items():
        if k in training_args_dict and training_args_dict[k] is not None:
            try:
                training_args_dict[k] = cast(training_args_dict[k])
            except Exception:
                pass
    training_args = Seq2SeqTrainingArguments(**training_args_dict)
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )
    # Train
    import os
    checkpoint_dir_to_check = training_args.output_dir
    checkpoint_found = False
    if os.path.isdir(checkpoint_dir_to_check):
        for entry in os.listdir(checkpoint_dir_to_check):
            if entry.startswith('checkpoint-') and os.path.isdir(os.path.join(checkpoint_dir_to_check, entry)):
                checkpoint_found = True
                break
    if checkpoint_found:
        print(f"Resuming training from last checkpoint in {checkpoint_dir_to_check}...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print(f"No valid checkpoint found in {checkpoint_dir_to_check}. Starting training from scratch...")
        trainer.train()
    # Evaluate on test set and print final WER
    eval_results = trainer.evaluate()
    print("Training complete. Best model saved at:", training_args.output_dir)
    print(f"Final WER on test set: {eval_results.get('eval_wer', 'N/A')}")
    # Save processor, feature_extractor, tokenizer
    processor.save_pretrained(training_args.output_dir)
    feature_extractor.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print("Custom dataset fine-tuning complete.")

def evaluate_checkpoint(
    checkpoint_path,
    dataset_name,
    lang,
    dataset_cache,
    model_cache=None,
    is_pretrained=False
):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
    import torch
    from datasets import load_dataset, Audio
    import evaluate
    # Load processor, feature extractor, tokenizer, and model
    if is_pretrained:
        processor = WhisperProcessor.from_pretrained(checkpoint_path, language=lang, task="transcribe")
        feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint_path)
        tokenizer = WhisperTokenizer.from_pretrained(checkpoint_path, language=lang, task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
    else:
        processor = WhisperProcessor.from_pretrained(checkpoint_path)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint_path)
        tokenizer = WhisperTokenizer.from_pretrained(checkpoint_path)
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Load test set
    test_set = load_dataset(
        dataset_name,
        lang,
        split="test",
        cache_dir=dataset_cache,
        trust_remote_code=True
    )
    test_set = test_set.cast_column("audio", Audio(sampling_rate=16000))
    # Ask for eval mode
    print("Select evaluation mode:")
    print("1. Complete test set (all samples)")
    print("2. 10 random samples")
    print("3. Gradio ASR demo (interactive UI)")
    print("4. Transcribe all audio files in a directory (with timestamps, real-time append)")
    eval_mode = input("Enter 1, 2, 3, or 4: ").strip()
    if eval_mode == "1":
        samples = list(test_set)
        eval_mode_str = "all"
        # ...existing code for test set evaluation...
        inputs = [feature_extractor(s["audio"]["array"], sampling_rate=16000).input_features[0] for s in samples]
        input_features = torch.tensor(inputs).unsqueeze(1) if len(inputs[0].shape) == 1 else torch.tensor(inputs)
        input_features = input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        pred_str = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        label_str = [s["sentence"] for s in samples]
        file_paths = [os.path.basename(s["audio"]["path"]) if "path" in s["audio"] else "" for s in samples]
        metric = evaluate.load("wer")
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        print(f"\nWER: {wer:.2f}%\n")
        # Save results as .tsv in ./evals, recreating checkpoint path structure
        if is_pretrained:
            model_name_for_dir = checkpoint_path.replace('/', '_')
            evals_dir = os.path.join("evals", "pretrained", model_name_for_dir)
        else:
            rel_ckpt_path = os.path.relpath(checkpoint_path, start=os.getcwd())
            evals_dir = os.path.join("evals", rel_ckpt_path)
        os.makedirs(evals_dir, exist_ok=True)
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        tsv_path = os.path.join(evals_dir, f"eval_{dt_str}.tsv")
        json_path = os.path.join(evals_dir, f"eval_{dt_str}.json")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("path\treference\thypothesis\n")
            for p, ref, hyp in zip(file_paths, label_str, pred_str):
                f.write(f"{p}\t{ref}\t{hyp}\n")
        eval_config = {
            "wer": wer,
            "language": lang,
            "eval_mode": eval_mode_str,
            "dataset_name": dataset_name,
            "model_name": checkpoint_path if is_pretrained else None,
            "checkpoint_path": None if is_pretrained else checkpoint_path,
            "num_samples": len(samples),
            "datetime": dt_str,
        }
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(eval_config, jf, indent=2)
        print(f"Evaluation log saved to: {tsv_path}\nConfig saved to: {json_path}")
    elif eval_mode == "2":
        samples = random.sample(list(test_set), 10)
        eval_mode_str = "10_samples"
        # ...existing code for 10-sample evaluation...
        inputs = [feature_extractor(s["audio"]["array"], sampling_rate=16000).input_features[0] for s in samples]
        input_features = torch.tensor(inputs).unsqueeze(1) if len(inputs[0].shape) == 1 else torch.tensor(inputs)
        input_features = input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        pred_str = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        label_str = [s["sentence"] for s in samples]
        file_paths = [os.path.basename(s["audio"]["path"]) if "path" in s["audio"] else "" for s in samples]
        metric = evaluate.load("wer")
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        print(f"\nWER: {wer:.2f}%\n")
        if is_pretrained:
            model_name_for_dir = checkpoint_path.replace('/', '_')
            evals_dir = os.path.join("evals", "pretrained", model_name_for_dir)
        else:
            rel_ckpt_path = os.path.relpath(checkpoint_path, start=os.getcwd())
            evals_dir = os.path.join("evals", rel_ckpt_path)
        os.makedirs(evals_dir, exist_ok=True)
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        tsv_path = os.path.join(evals_dir, f"eval_{dt_str}.tsv")
        json_path = os.path.join(evals_dir, f"eval_{dt_str}.json")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("path\treference\thypothesis\n")
            for p, ref, hyp in zip(file_paths, label_str, pred_str):
                f.write(f"{p}\t{ref}\t{hyp}\n")
        eval_config = {
            "wer": wer,
            "language": lang,
            "eval_mode": eval_mode_str,
            "dataset_name": dataset_name,
            "model_name": checkpoint_path if is_pretrained else None,
            "checkpoint_path": None if is_pretrained else checkpoint_path,
            "num_samples": len(samples),
            "datetime": dt_str,
        }
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(eval_config, jf, indent=2)
        print(f"Evaluation log saved to: {tsv_path}\nConfig saved to: {json_path}")
    elif eval_mode == "3":
        gradio_transcribe_interface(
            checkpoint_path=checkpoint_path,
            lang=lang,
            model_cache=model_cache,
            is_pretrained=is_pretrained
        )
        return
    elif eval_mode == "4":
        import glob
        import soundfile as sf
        import numpy as np
        print("Enter directory containing mp3 files:")
        audio_dir = input("Directory path: ").strip()
        if not os.path.isdir(audio_dir):
            print(f"Directory not found: {audio_dir}")
            return
        mp3_files = glob.glob(os.path.join(audio_dir, "*.mp3"))
        if not mp3_files:
            print(f"No mp3 files found in {audio_dir}")
            return
        print(f"Found {len(mp3_files)} mp3 files. Starting transcription...")
        for mp3_path in mp3_files:
            base = os.path.splitext(os.path.basename(mp3_path))[0]
            out_txt = os.path.join(audio_dir, base + ".txt")
            print(f"Transcribing {mp3_path} -> {out_txt}")
            # Load audio
            audio_array, sr = sf.read(mp3_path)
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            # Resample if needed
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                sr = 16000
            # Chunking parameters
            max_chunk_sec = 30
            max_chunk_samples = max_chunk_sec * 16000
            overlap_sec = 1
            overlap_samples = overlap_sec * 16000
            total_samples = len(audio_array)
            start = 0
            with open(out_txt, "w", encoding="utf-8") as outf:
                while start < total_samples:
                    end = min(start + max_chunk_samples, total_samples)
                    chunk = audio_array[start:end]
                    # Prepare input for chunk
                    inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
                    input_features = inputs.input_features.to(device)
                    # Transcribe chunk
                    with torch.no_grad():
                        outputs = model.generate(
                            input_features,
                            task="transcribe",
                            language=lang
                        )
                    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                    # Write transcript to file (append in real time)
                    for line in transcription.split("\n"):
                        outf.write(line.strip() + "\n")
                        outf.flush()
                    # Move to next chunk with overlap
                    if end == total_samples:
                        break
                    start = end - overlap_samples
            print(f"Done: {out_txt}")
        print("All files transcribed.")
        return
    else:
        print("Invalid selection.")
        return

def gradio_transcribe_interface(
    checkpoint_path,
    lang,
    model_cache=None,
    is_pretrained=False
):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import torch
    import evaluate
    import numpy as np
    # Load processor and model
    processor = WhisperProcessor.from_pretrained(checkpoint_path)
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    metric = evaluate.load("wer")
    def transcribe_and_score(audio, reference_text):
        if audio is None:
            return "No audio provided.", "-"
        # Gradio provides audio as (sr, np.ndarray)
        if isinstance(audio, tuple):
            sr, audio_np = audio
        else:
            audio_np = audio
            sr = 16000
        # Convert to mono if stereo
        if len(audio_np.shape) > 1:
            audio_np = np.mean(audio_np, axis=1)
        # Convert to float32 if not already
        if not np.issubdtype(audio_np.dtype, np.floating):
            # If int16, scale to [-1, 1]
            if audio_np.dtype == np.int16:
                audio_np = audio_np.astype(np.float32) / 32768.0
            else:
                audio_np = audio_np.astype(np.float32)
        # Resample if needed
        if sr != 16000:
            import librosa
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
            sr = 16000
        inputs = processor(audio_np, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features, task="transcribe", language=lang)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        if reference_text and reference_text.strip():
            wer = 100 * metric.compute(predictions=[transcription], references=[reference_text.strip()])
            return transcription, f"{wer:.2f}%"
        else:
            return transcription, "-"
    demo = gr.Interface(
        fn=transcribe_and_score,
        inputs=[
            gr.Audio(sources=["upload", "microphone"], type="numpy", label="Audio (upload or record)"),
            gr.Textbox(lines=2, label="Reference Text (optional)", placeholder="Paste ground truth here if you want WER computed")
        ],
        outputs=[
            gr.Textbox(label="Transcription"),
            gr.Textbox(label="WER")
        ],
        title="Whisper ASR Evaluation",
        description="Upload or record an audio file. Optionally, provide the ground truth text to compute WER."
    )
    demo.launch()

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for multilingual ASR")
    parser.add_argument("-l", "--lang", dest="lang", type=str, default=None, help="Language code (ISO 639-1/2/3) for both dataset and model (e.g., 'as' for Assamese)")
    parser.add_argument("-dn", "--name", dest="dataset_name", type=str, default=None, help="Dataset name (default: mozilla-foundation/common_voice_11_0)")
    parser.add_argument("-dc", "--cache", dest="dataset_cache", type=str, default=None, help="Dataset cache directory (default: ./datasets)")
    parser.add_argument("-mn", "--model-name", dest="model_name", type=str, default=None, help="Model name (default: whisper-medium)")
    parser.add_argument("-mc", "--model-cache", dest="model_cache", type=str, default=None, help="Model cache directory (default: ./models)")
    parser.add_argument("-wp", "--whisper-pretrained", dest="whisper_pretrained", type=str, default=None, help="Whisper pretrained model (default: openai/<model_name>)")
    parser.add_argument("-ck", "--checkpoint-name", dest="checkpoint_name", type=str, default=None, help="Checkpoint name (default: <model_name>-<model_lang>)")
    parser.add_argument("-cd", "--checkpoint-dir", dest="checkpoint_dir", type=str, default=None, help="Checkpoint directory (default: ./checkpoints/<checkpoint_name>)")
    parser.add_argument('-t', '--max-steps', dest='max_steps', type=int, default=None, help='Max training steps (default: 4000)')
    parser.add_argument('--config', dest='config', type=str, default='config.yaml', help='YAML config file for all script and training arguments (default: config.yaml)')
    parser.add_argument('-E', '--eval', dest='eval_mode', action='store_true', help='Run in evaluation mode')
    parser.add_argument('-ch', '--checkpoint', dest='eval_checkpoint', type=str, default=None, help='Checkpoint directory path for evaluation')
    parser.add_argument('-pt', '--pretrained', dest='pretrained_model', type=str, default=None, help='HuggingFace model name to evaluate (e.g., openai/whisper-medium)')
    parser.add_argument('--gradio', dest='gradio_mode', action='store_true', help='Launch Gradio ASR demo')
    args = parser.parse_args()

    # Determine mode
    if args.eval_mode:
        mode = "eval"
    elif args.gradio_mode:
        mode = "gradio"
    else:
        mode = "train"

    config = args.config
    config_dict = {}
    if config is not None and os.path.exists(config):
        with open(config, 'r') as f:
            config_dict = yaml.safe_load(f)
        print(f"Loaded all arguments from config: {config}")

    # Use config values as defaults if not set by CLI
    lang = args.lang if args.lang is not None else config_dict.get('lang', 'as')
    dataset_name = args.dataset_name if args.dataset_name is not None else config_dict.get('dataset_name', 'mozilla-foundation/common_voice_11_0')
    dataset_cache = args.dataset_cache if args.dataset_cache is not None else config_dict.get('dataset_cache', './datasets')
    model_name = args.model_name if args.model_name is not None else config_dict.get('model_name', 'whisper-medium')
    model_cache = args.model_cache if args.model_cache is not None else config_dict.get('model_cache', './models')
    whisper_pretrained = args.whisper_pretrained if args.whisper_pretrained is not None else config_dict.get('whisper_pretrained', f"openai/{model_name}")
    checkpoint_name = args.checkpoint_name if args.checkpoint_name is not None else config_dict.get('checkpoint_name', f"{model_name}-{lang}")
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else config_dict.get('checkpoint_dir', f"./checkpoints/{checkpoint_name}")
    max_steps = args.max_steps if args.max_steps is not None else config_dict.get('max_steps', 4000)

    if mode == "train":
        # Prepare training_args_dict for Seq2SeqTrainingArguments
        training_args_dict = {k: v for k, v in config_dict.items() if k not in [
            'lang','dataset_name','dataset_cache','model_name','model_cache','whisper_pretrained','checkpoint_name','checkpoint_dir','training_max_steps','custom_dataset']}

        # If custom_dataset is set, use custom fine-tuning logic
        if config_dict.get('custom_dataset'):
            custom_dataset_path = config_dict['custom_dataset']
            # Infer dataset folder name for checkpoint naming
            dataset_folder = os.path.basename(os.path.dirname(custom_dataset_path.rstrip('/\\')))
            custom_checkpoint_name = f"{dataset_folder}-{model_name}-{lang}"
            custom_checkpoint_dir = os.path.join("./checkpoints", custom_checkpoint_name)
            finetune_custom_dataset(
                custom_dataset_path=custom_dataset_path,
                lang=lang,
                model_name=model_name,
                model_cache=model_cache,
                whisper_pretrained=whisper_pretrained,
                checkpoint_name=custom_checkpoint_name,
                checkpoint_dir=custom_checkpoint_dir,
                max_steps=max_steps,
                training_args_dict=training_args_dict
            )
        else:
            finetune_whisper(
                lang=lang,
                dataset_name=dataset_name,
                dataset_cache=dataset_cache,
                model_name=model_name,
                model_cache=model_cache,
                whisper_pretrained=whisper_pretrained,
                checkpoint_name=checkpoint_name,
                checkpoint_dir=checkpoint_dir,
                max_steps=max_steps,
                training_args_dict=training_args_dict
            )


    elif mode == "eval":
        if args.pretrained_model:
            # Evaluate HuggingFace pretrained model
            pretrained_name = args.pretrained_model
            print(f"Evaluating HuggingFace pretrained model: {pretrained_name}")
            evaluate_checkpoint(
                checkpoint_path=pretrained_name,
                dataset_name=dataset_name,
                lang=lang,
                dataset_cache=dataset_cache,
                model_cache=model_cache,
                is_pretrained=True
            )
        elif not args.eval_checkpoint and not args.pretrained_model:
            # If -E is passed without -ch or -pt, resolve whisper_pretrained as openai/<model_name> if not set
            if config_dict.get('whisper_pretrained'):
                pretrained_name = config_dict['whisper_pretrained']
            else:
                pretrained_name = f"openai/{model_name}"
            print(f"Evaluating resolved whisper_pretrained model: {pretrained_name}")
            evaluate_checkpoint(
                checkpoint_path=pretrained_name,
                dataset_name=dataset_name,
                lang=lang,
                dataset_cache=dataset_cache,
                model_cache=model_cache,
                is_pretrained=True
            )
        else:
            checkpoint_path = args.eval_checkpoint if args.eval_checkpoint else input("Enter checkpoint directory path to evaluate: ").strip()
            evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                dataset_name=dataset_name,
                lang=lang,
                dataset_cache=dataset_cache,
                model_cache=model_cache
            )
    elif mode == "gradio" or args.gradio_mode:
        checkpoint_path = args.eval_checkpoint if args.eval_checkpoint else checkpoint_dir
        gradio_transcribe_interface(
            checkpoint_path=checkpoint_path,
            lang=lang,
            model_cache=model_cache,
            is_pretrained=False
        )
        return

if __name__ == "__main__":
    main()