import os
import sys
import argparse
import torch  # Ensure torch is imported at the top
import yaml  # Add at the top with other imports

# Early parse for GPU
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", dest="gpu_device", type=str, default=None, help="GPU device id to use (default: all available)")
parser.add_argument('--config', dest='config', type=str, default='config.yaml', help='YAML config file for all script and training arguments (default: config.yaml)')
args, unknown = parser.parse_known_args()

if args.gpu_device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    print(f"Set CUDA_VISIBLE_DEVICES to {args.gpu_device} (will appear as device 0 in torch)")
    # Always use device 0 in torch if CUDA_VISIBLE_DEVICES is set
    torch.cuda.set_device(0)

# Load Hugging Face token from config.yaml
config = args.config
config_dict = {}
if config is not None and os.path.exists(config):
    with open(config, 'r') as f:
        config_dict = yaml.safe_load(f)
    print(f"Loaded all arguments from config: {config}")

hf_token = config_dict.get('hf_token', None)
print(f"HF Token: {hf_token}")
from huggingface_hub import login
login(token=hf_token)

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
    # Assign all variables from config, applying derived defaults first
    dataset_lang = config.get('dataset_lang')
    dataset_name = config.get('dataset_name')
    dataset_cache = config.get('dataset_cache')
    model_lang = config.get('model_lang')
    model_name = config.get('model_name')
    model_cache = config.get('model_cache')
    whisper_pretrained = config.get('whisper_pretrained') or (f"openai/{model_name}" if model_name else None)
    checkpoint_name = config.get('checkpoint_name') or (f"{model_name}-{model_lang}" if model_name and model_lang else None)
    checkpoint_dir = config.get('checkpoint_dir') or (f"./checkpoints/{checkpoint_name}" if checkpoint_name else None)
    max_steps = config.get('max_steps')
    gpu_device = config.get('gpu_device')
    hf_token = config.get('hf_token')
    # Update config dict with derived values so printout is correct
    config['whisper_pretrained'] = whisper_pretrained
    config['checkpoint_name'] = checkpoint_name
    config['checkpoint_dir'] = checkpoint_dir
    # General config keys
    general_keys = [
        'dataset_lang','dataset_name','dataset_cache','model_lang','model_name','model_cache',
        'whisper_pretrained','checkpoint_name','checkpoint_dir','gpu_device','hf_token'
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
    dataset_lang = config.get('dataset_lang')
    dataset_name = config.get('dataset_name')
    dataset_cache = config.get('dataset_cache')
    model_lang = config.get('model_lang')
    model_name = config.get('model_name')
    model_cache = config.get('model_cache')
    whisper_pretrained = config.get('whisper_pretrained') or f"openai/{model_name}"
    checkpoint_name = config.get('checkpoint_name') or f"{model_name}-{model_lang}"
    checkpoint_dir = config.get('checkpoint_dir') or f"./checkpoints/{checkpoint_name}"
    max_steps = config.get('max_steps')
    gpu_device = config.get('gpu_device')
    hf_token = config.get('hf_token')

    # Load dataset
    common_voice = DatasetDict()
    common_voice["train"] = load_dataset(
        dataset_name,
        dataset_lang,
        split="train+validation",
        cache_dir=dataset_cache,
        trust_remote_code=True
    )
    common_voice["test"] = load_dataset(
        dataset_name,
        dataset_lang,
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
    tokenizer = WhisperTokenizer.from_pretrained(whisper_pretrained, language=model_lang, task="transcribe", cache_dir=model_cache)
    processor = WhisperProcessor.from_pretrained(whisper_pretrained, language=model_lang, task="transcribe", cache_dir=model_cache)
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
    model.generation_config.language = model_lang
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
    # Train the model, resuming from last checkpoint if available
    trainer.train(resume_from_checkpoint=True)
    # Evaluate on test set and print final WER
    eval_results = trainer.evaluate()
    print("Training complete. Best model saved at:", training_args.output_dir)
    print(f"Final WER on test set: {eval_results.get('eval_wer', 'N/A')}")

    # Push to hub logic (from notebook)
    push_kwargs = {
        "dataset_tags": f"{dataset_name}",
        "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
        "dataset_args": f"config: {dataset_lang}, split: test",
        "language": f"{model_lang}",
        "model_name": f"{checkpoint_name} - Fine-tuned",  # a 'pretty' name for our model
        "finetuned_from": whisper_pretrained,
        "tasks": "automatic-speech-recognition",
    }
    print("Pushing model and processor to the Hugging Face Hub...")
    trainer.push_to_hub(**push_kwargs)
    processor.save_pretrained(training_args.output_dir)
    print("Push to hub complete.")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for multilingual ASR")
    parser.add_argument("-dl", "--lang", dest="dataset_lang", type=str, default=None, help="Dataset language (default: as)")
    parser.add_argument("-dn", "--name", dest="dataset_name", type=str, default=None, help="Dataset name (default: mozilla-foundation/common_voice_11_0)")
    parser.add_argument("-dc", "--cache", dest="dataset_cache", type=str, default=None, help="Dataset cache directory (default: ./datasets)")
    parser.add_argument("-ml", "--model-lang", dest="model_lang", type=str, default=None, help="Model language (default: assamese)")
    parser.add_argument("-mn", "--model-name", dest="model_name", type=str, default=None, help="Model name (default: whisper-medium)")
    parser.add_argument("-mc", "--model-cache", dest="model_cache", type=str, default=None, help="Model cache directory (default: ./models)")
    parser.add_argument("-wp", "--whisper-pretrained", dest="whisper_pretrained", type=str, default=None, help="Whisper pretrained model (default: openai/<model_name>)")
    parser.add_argument("-ck", "--checkpoint-name", dest="checkpoint_name", type=str, default=None, help="Checkpoint name (default: <model_name>-<model_lang>)")
    parser.add_argument("-cd", "--checkpoint-dir", dest="checkpoint_dir", type=str, default=None, help="Checkpoint directory (default: ./checkpoints/<checkpoint_name>)")
    parser.add_argument('-t', '--max-steps', dest='max_steps', type=int, default=None, help='Max training steps (default: 4000)')
    parser.add_argument("-g", "--gpu", dest="gpu_device", type=str, default=None, help="GPU device id to use (default: all available)")
    parser.add_argument('--config', dest='config', type=str, default='config.yaml', help='YAML config file for all script and training arguments (default: config.yaml)')
    args = parser.parse_args()

    config = args.config
    config_dict = {}
    if config is not None and os.path.exists(config):
        with open(config, 'r') as f:
            config_dict = yaml.safe_load(f)
        print(f"Loaded all arguments from config: {config}")

    # Use config values as defaults if not set by CLI
    dataset_lang = args.dataset_lang if args.dataset_lang is not None else config_dict.get('dataset_lang', 'as')
    dataset_name = args.dataset_name if args.dataset_name is not None else config_dict.get('dataset_name', 'mozilla-foundation/common_voice_11_0')
    dataset_cache = args.dataset_cache if args.dataset_cache is not None else config_dict.get('dataset_cache', './datasets')
    model_lang = args.model_lang if args.model_lang is not None else config_dict.get('model_lang', 'assamese')
    model_name = args.model_name if args.model_name is not None else config_dict.get('model_name', 'whisper-medium')
    model_cache = args.model_cache if args.model_cache is not None else config_dict.get('model_cache', './models')
    whisper_pretrained = args.whisper_pretrained if args.whisper_pretrained is not None else config_dict.get('whisper_pretrained', f"openai/{model_name}")
    checkpoint_name = args.checkpoint_name if args.checkpoint_name is not None else config_dict.get('checkpoint_name', f"{model_name}-{model_lang}")
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else config_dict.get('checkpoint_dir', f"./checkpoints/{checkpoint_name}")
    max_steps = args.max_steps if args.max_steps is not None else config_dict.get('max_steps', 4000)
    gpu_device = args.gpu_device if args.gpu_device is not None else config_dict.get('gpu_device', None)
    hf_token = config_dict.get('hf_token', None)

    # Prepare training_args_dict for Seq2SeqTrainingArguments
    training_args_dict = {k: v for k, v in config_dict.items() if k not in [
        'dataset_lang','dataset_name','dataset_cache','model_lang','model_name','model_cache','whisper_pretrained','checkpoint_name','checkpoint_dir','training_max_steps','gpu_device','hf_token']}

    finetune_whisper(
        dataset_lang=dataset_lang,
        dataset_name=dataset_name,
        dataset_cache=dataset_cache,
        model_lang=model_lang,
        model_name=model_name,
        model_cache=model_cache,
        whisper_pretrained=whisper_pretrained,
        checkpoint_name=checkpoint_name,
        checkpoint_dir=checkpoint_dir,
        max_steps=max_steps,
        gpu_device=gpu_device,
        hf_token=hf_token,
        training_args_dict=training_args_dict
    )

if __name__ == "__main__":
    main()