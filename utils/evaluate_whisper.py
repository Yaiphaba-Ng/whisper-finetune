import os
import sys
import argparse
import torch
import yaml
import pandas as pd
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate

def evaluate_whisper(config_dict):
    # Extract configuration
    lang = config_dict.get('lang', 'as')
    dataset_name = config_dict.get('dataset_name', 'mozilla-foundation/common_voice_11_0')
    dataset_cache = config_dict.get('dataset_cache', './datasets')
    model_name = config_dict.get('model_name', 'whisper-medium')
    model_cache = config_dict.get('model_cache', './models')
    checkpoint_name = config_dict.get('checkpoint_name') or f"{model_name}-{lang}"
    checkpoint_dir = config_dict.get('checkpoint_dir') or f"./checkpoints/{checkpoint_name}"

    print("\n===== Whisper Evaluation Configuration =====")
    print(f"Language: {lang}")
    print(f"Dataset: {dataset_name}")
    print(f"Checkpoint: {checkpoint_dir}")
    print("==========================================\n")

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = load_dataset(
        dataset_name,
        lang,
        split="test",
        cache_dir=dataset_cache,
        trust_remote_code=True
    )
    
    # Take first 10 samples
    test_dataset = test_dataset.select(range(min(10, len(test_dataset))))
    
    # Remove unnecessary columns and resample audio
    test_dataset = test_dataset.remove_columns([
        "accent", "age", "client_id", "down_votes", "gender", 
        "locale", "path", "segment", "up_votes"
    ])
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))    # Load model and processor
    print(f"Loading model from {checkpoint_dir}...")
    processor = WhisperProcessor.from_pretrained(checkpoint_dir)
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir)

    if torch.cuda.is_available():
        model = model.to("cuda")
        print("Model moved to GPU")
    
    # Prepare metric
    metric = evaluate.load("wer")
    
    # Process and evaluate samples
    results = []
    print("\nProcessing samples...")
    
    for idx, item in enumerate(test_dataset):
        # Prepare audio
        processed_audio = processor(
            item["audio"]["array"],
            sampling_rate=item["audio"]["sampling_rate"],
            return_tensors="pt"
        )
        input_features = processed_audio.input_features
        if torch.cuda.is_available():
            input_features = input_features.to("cuda")
        # Generate prediction
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                task="transcribe",
                language=lang
            )
        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        # Store results
        results.append({
            'id': idx + 1,
            'reference': item['sentence'],
            'hypothesis': transcription,
        })
        
        # Print progress
        print(f"\nSample {idx + 1}:")
        print(f"Reference: {item['sentence']}")
        print(f"Hypothesis: {transcription}")
    
    # Calculate WER
    wer = 100 * metric.compute(
        predictions=[r['hypothesis'] for r in results],
        references=[r['reference'] for r in results]
    )
    print(f"\nWord Error Rate (WER): {wer:.2f}%")
    
    # Save results to TSV
    output_file = os.path.join(checkpoint_dir, 'evaluation_results.tsv')
    df = pd.DataFrame(results)
    df.to_csv(output_file, sep='\t', index=False)
    print(f"\nResults saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Whisper model")
    parser.add_argument('--config', dest='config', type=str, default='config.yaml',
                      help='YAML config file (default: config.yaml)')
    parser.add_argument('-c', '--checkpoint', dest='checkpoint_dir', type=str, required=True,
                      help='Path to the checkpoint directory containing the fine-tuned model')
    args = parser.parse_args()

    # Load configuration
    config_dict = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        print(f"Loaded configuration from: {args.config}")
    
    # Override checkpoint directory from command line
    config_dict['checkpoint_dir'] = args.checkpoint_dir
    
    evaluate_whisper(config_dict)

if __name__ == "__main__":
    main()
