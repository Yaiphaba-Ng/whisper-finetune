# Whisper Fine-Tuning: Detailed Guide

## Overview
This project provides a robust, configurable script for fine-tuning OpenAI's Whisper model for multilingual automatic speech recognition (ASR) using Hugging Face Transformers. All configuration is handled via a single YAML file (`config.yaml`) and/or command-line arguments. Training is fully resumable and supports GPU selection.

---

## 1. Directory Structure

```
whisper-finetune/
├── config.yaml                # Main configuration file for all script and training arguments
├── whisper_finetune_script.py # Main training script
├── run_training.sh            # Bash script to launch training (Linux/WSL)
├── datasets/                  # Downloaded and processed datasets
│   └── ...
├── checkpoints/               # Model checkpoints (auto-created)
│   └── <checkpoint_name>/
│       └── ...
├── models/                    # Model cache (Hugging Face format)
│   └── ...
└── ...
```

- **datasets/**: Contains downloaded and processed datasets. Subfolders are created per dataset/config.
- **checkpoints/**: Contains all training checkpoints. Each run creates a subfolder named after the checkpoint (e.g., `whisper-medium-assamese`).
- **models/**: Hugging Face model cache. Stores downloaded model weights and tokenizer files.

---

## 2. Configuration: `config.yaml`
All script and training arguments are set in `config.yaml`. Example:

```yaml
# General configuration
# (Set these to match your dataset and model preferences)
# For language codes, see: https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
lang: as  # ISO 639-1/2/3 code for both dataset and model language
dataset_name: mozilla-foundation/common_voice_11_0
dataset_cache: ./datasets
model_name: whisper-medium
model_cache: ./models
gpu_device: 1
whisper_pretrained: null  # If null, defaults to openai/<model_name>
checkpoint_name: null     # If null, defaults to <model_name>-<lang>
checkpoint_dir: null      # If null, defaults to ./checkpoints/<checkpoint_name>
hf_token: <your_hf_token> # Required: Hugging Face access token

# Training arguments (passed to Hugging Face Seq2SeqTrainingArguments)
per_device_train_batch_size: 16
gradient_accumulation_steps: 1
learning_rate: 1e-5
warmup_steps: 500
max_steps: 4000
gradient_checkpointing: true
fp16: true
eval_strategy: steps
per_device_eval_batch_size: 8
predict_with_generate: true
generation_max_length: 225
save_steps: 1000
eval_steps: 1000
logging_steps: 25
report_to:
  - tensorboard
load_best_model_at_end: true
metric_for_best_model: wer
greater_is_better: false
push_to_hub: true
```

> **Note:** For a list of language codes, see: https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes

---

## 3. Command-Line Arguments
All arguments can be overridden from the CLI. Example:

```
python whisper_finetune_script.py --config config.yaml -g 0 --max-steps 8000 --model-name whisper-large --lang as
```

**Key CLI arguments:**
- `--config`: Path to YAML config file (default: `config.yaml`)
- `-g`, `--gpu`: GPU device id to use (overrides config)
- `-l`, `--lang`: Language code (ISO 639-1/2/3) for both dataset and model (e.g., `as` for Assamese)
- `-dn`, `--name`: Dataset name
- `-dc`, `--cache`: Dataset cache directory
- `-mn`, `--model-name`: Model name
- `-mc`, `--model-cache`: Model cache directory
- `-wp`, `--whisper-pretrained`: Pretrained model (Hugging Face hub path)
- `-ck`, `--checkpoint-name`: Checkpoint name
- `-cd`, `--checkpoint-dir`: Checkpoint directory
- `-t`, `--max-steps`: Max training steps

---

## 4. Running Training

### Bash Script: `run_training.sh`
```bash
#!/bin/bash
nohup python whisper_finetune_script.py -g 1 > train.log 2>&1 &
```
- Launches training on GPU 1 in the background, logging output to `train.log`.
- Edit the `-g 1` to select a different GPU.
- You can also run the script directly from the command line with any CLI overrides you need.

---

## 5. Checkpoints & Resuming Training
- Checkpoints are saved in `checkpoints/<checkpoint_name>/` every `save_steps` steps.
- Training will automatically resume from the last checkpoint if you rerun the script with the same config and checkpoint directory.
- To start fresh, delete the checkpoint directory or specify a new `checkpoint_name`.

---

## 6. Training Arguments Explained

| Argument                      | Description                                                                 | Effect of Changing Value                                  |
|-------------------------------|-----------------------------------------------------------------------------|-----------------------------------------------------------|
| per_device_train_batch_size   | Batch size per GPU during training                                          | Larger = faster, but more memory needed                   |
| gradient_accumulation_steps   | Number of steps to accumulate gradients before optimizer step               | Higher = effective batch size increases, slower updates   |
| learning_rate                 | Initial learning rate                                                       | Higher = faster learning, but risk of instability         |
| warmup_steps                  | Steps to linearly increase LR from 0 to set value                           | More = slower ramp-up, can help stability                 |
| max_steps                     | Total number of training steps                                              | More = longer training, better results (to a point)       |
| gradient_checkpointing        | Enable memory-saving gradient checkpointing                                 | True = less memory, slightly slower                       |
| fp16                          | Use mixed-precision (FP16) training                                         | True = faster, less memory (if supported by GPU)          |
| eval_strategy                 | When to run evaluation (e.g., 'steps', 'epoch')                             | 'steps' = more frequent eval, 'epoch' = once per epoch    |
| per_device_eval_batch_size    | Batch size per GPU during evaluation                                        | Larger = faster eval, more memory                         |
| predict_with_generate         | Use generation for evaluation                                               | True = more accurate eval, slower                         |
| generation_max_length         | Max tokens to generate during evaluation                                    | Higher = can decode longer outputs, more compute          |
| save_steps                    | Save checkpoint every N steps                                               | Lower = more frequent saves, more disk usage              |
| eval_steps                    | Evaluate every N steps                                                      | Lower = more frequent eval, more compute                  |
| logging_steps                 | Log metrics every N steps                                                   | Lower = more frequent logs, more disk/console output      |
| report_to                     | Where to report logs (e.g., 'tensorboard')                                  | Add 'wandb' for Weights & Biases, etc.                   |
| load_best_model_at_end        | Load best model (by metric) after training                                  | True = best model is used for final eval/push             |
| metric_for_best_model         | Metric to select best model                                                 | Usually 'wer' for ASR                                     |
| greater_is_better             | Whether higher metric is better                                             | False for WER (lower is better)                           |
| push_to_hub                   | Push final model to Hugging Face Hub                                        | True = uploads model, False = local only                  |

---

## 7. Notes & Best Practices
- Always set your Hugging Face token in `config.yaml` (`hf_token:`).
- For new experiments, change `checkpoint_name` to avoid overwriting previous runs.
- Monitor training with TensorBoard (if enabled in `report_to`).
- Adjust batch size and accumulation steps to fit your GPU memory.
- Resume interrupted training simply by rerunning the script.

---

## 8. Example: Quick Start

1. Edit `config.yaml` with your dataset/model/token.
2. Make the training script executable (only needed once after git clone):
   ```bash
   chmod +x run_training.sh
   ```
3. Run:
   ```bash
   bash run_training.sh
   # or
   nohup python whisper_finetune_script.py -g 1 > train.log 2>&1 &
   # To monitor training log in real time:
   tail -f train.log
   ```
4. Monitor progress in `train.log` or with TensorBoard.

---

## 9. Troubleshooting: Checkpoint Resume Warnings

- If you see a message like:
  ```
  there were missing keys in the checkpoint model loaded: ['proj_out.weight']
  ```
  This means the checkpoint does not perfectly match the current model architecture. This can happen if:
  - You changed the model type, size, or config between runs.
  - The checkpoint was created with a different code version.
  - The checkpoint is partially saved or corrupted.

- **What happens?**
  - Training will continue, and missing weights will be randomly initialized.
  - If you did not change the model config, you can usually ignore this warning.
  - If you see unexpected results, start fresh by deleting the checkpoint directory.

- **Best practice:**
  - Always resume with the same model config and code as the checkpoint was created with.
  - Double-check your `model_name`, `whisper_pretrained`, and all config values before resuming.

For further customization, see the Hugging Face [Seq2SeqTrainingArguments documentation](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments).
