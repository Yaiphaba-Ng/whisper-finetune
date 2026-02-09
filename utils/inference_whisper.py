import os
import argparse
import torch
import gradio as gr
import glob
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
import torchaudio

# Default model directory (where checkpoints and tokenizer are saved)
default_model_dir = './checkpoints/lamzing-whisper-small-mni'

def find_latest_checkpoint(model_dir):
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(model_dir, d))]
    if not checkpoints:
        return model_dir  # fallback to main dir if no sub-checkpoints
    latest = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
    return os.path.join(model_dir, latest)

def load_model_and_processor(model_dir):
    checkpoint_dir = find_latest_checkpoint(model_dir)
    processor = WhisperProcessor.from_pretrained(model_dir)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
    tokenizer = WhisperTokenizer.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir)
    model.eval()
    return model, processor, feature_extractor, tokenizer

def transcribe_audio(model, processor, audio_path, device='cpu'):
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    input_features = processor.feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def gradio_interface(model, processor, device):
    def transcribe_gr(audio):
        if audio is None:
            return "No audio provided."
        import numpy as np
        sr, data = audio
        # Ensure data is float32 in range [-1, 1] for all filetypes
        if data.dtype != np.float32:
            # Handles int16, int32, etc.
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        # If stereo, convert to mono by averaging channels
        if len(data.shape) > 1:
            data = np.mean(data, axis=-1)
        if sr != 16000:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        input_features = processor.feature_extractor(data, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
    iface = gr.Interface(
        fn=transcribe_gr,
        inputs=gr.Audio(type="numpy", label="Upload Audio (16kHz preferred)"),
        outputs=gr.Textbox(label="Transcription"),
        title="Whisper Custom Checkpoint Transcription",
        description="Upload an audio file to transcribe using your fine-tuned Whisper model."
    )
    iface.launch()

def batch_transcribe(model, processor, input_dir, output_dir, device):
    audio_files = glob.glob(os.path.join(input_dir, '*.wav'))
    if not audio_files:
        print(f"No .wav files found in {input_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    for audio_path in audio_files:
        transcript = transcribe_audio(model, processor, audio_path, device=device)
        base = os.path.splitext(os.path.basename(audio_path))[0]
        out_path = os.path.join(output_dir, base + '.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        print(f"Transcribed {audio_path} -> {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Whisper Custom Checkpoint Inference")
    parser.add_argument('-md', '--model_dir', type=str, default=default_model_dir, help='Path to model checkpoint directory (with tokenizer/processor)')
    parser.add_argument('--mode', type=str, choices=['g', 'b'], required=True, help='Inference mode: g for gradio or b for batch')
    parser.add_argument('-i', '--input_dir', type=str, help='Input directory for batch mode (contains .wav files)')
    parser.add_argument('-o', '--output_dir', type=str, help='Directory to save transcripts in batch mode (default: input_dir)')
    parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cpu or cuda)')
    args = parser.parse_args()

    if args.mode == 'batch':
        if not args.input_dir:
            print("-i or --input_dir is required for batch mode.")
            return
        if not args.output_dir:
            args.output_dir = args.input_dir

    model, processor, feature_extractor, tokenizer = load_model_and_processor(args.model_dir)
    model.to(args.device)

    if args.mode == 'g':
        gradio_interface(model, processor, args.device)
    elif args.mode == 'b':
        batch_transcribe(model, processor, args.input_dir, args.output_dir, args.device)

if __name__ == '__main__':
    main() 